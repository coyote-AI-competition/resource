import socketio
import eventlet
import time
import json
import random
import coyote
import os
from collections import defaultdict


class RoomManager:
    def __init__(self):
        self.rooms = {}  # {room_id: Roomインスタンス}

    def create_room(self, room_id):
        if room_id not in self.rooms:
            self.rooms[room_id] = Room(room_id)

    def get_room(self, room_id):
        return self.rooms.get(room_id)

    def close_room(self, room_id):
        self.rooms.pop(room_id, None)


class Room:
    def __init__(self, room_id):
        self.room_id = room_id
        self.game_num = 0
        self.round_num = 0
        self.players = {}  #  {sid: {"life": int, "card": int, "name": str, "win_num": int, "is_ai" : bool, "others_card_sum" : 0}, ...]}
        self.observers = []  # [sid, sid, ...]
        self.active_players = []  # [(sid, name, is_ai), ...]
        self.death_players = []  # [(sid, name, is_ai), ...]
        self.start_players = []  # [sid, ...]
        self.logs = {"round_info": []}  # logの形式を変更
        self.current_turn_index = 0  # 誰のターンか管理
        self.current_game_index = 0  # 何ゲーム目か管理
        self.initial_life = 3
        self.deck = coyote.game.Deck()
        # 変数はライブラリ化しない？
        self.card_sum = 0  # calc_card_sumで使用する
        self.hold_cards = []  # [card, card, ...]
        self.is_shuffle_card = False
        self.is_double_card = False
        self.last_action = 0
        self.is_started = False  # サーバークラスではなくルームクラスに存在すべき
        self.legal_action = []
        self.players_card_sum = 0  # 全プレイヤーの合計値
        self.death_order = defaultdict(
            lambda: defaultdict(int)
        )  # 死亡したプレイヤーのランキング
        # self.cashed_cards = self.deck.cashed_cards.copy()

    def draw_card(self):
        return coyote.game.server_draw_card(self.deck)

    def deck_top_show(self):
        return coyote.game.server_top_show_card(self.deck)

    def convert_card(self, data, Is_othersum):
        return coyote.game.convert_card(self, data, Is_othersum, self.deck)

    # #場のカードの合計値を計算する
    def calc_card_sum(self, true_cards):
        return coyote.game.calc_card_sum(self, true_cards)

    def set_game_num(self, num):
        self.game_num = num

    def add_current_game_index(self):
        self.current_game_index += 1

    def add_current_turn_index(self):
        self.current_turn_index += 1

    def add_player(self, sid, player_name, is_ai=False):
        self.players[sid] = {
            "life": 3,
            "card": 0,
            "name": player_name,
            "win_num": 0,
            "is_ai": is_ai,
            "others_card_sum": 0,
        }
        self.active_players.append((sid, player_name, is_ai))

    def remove_player(self, player_sid):
        self.players.pop(player_sid, None)
        self.active_players = [
            (sid, name, is_ai)
            for sid, name, is_ai in self.active_players
            if sid != player_sid
        ]

    def start_new_round(self, round_num):
        # deck を入れる
        deck = self.deck.cards.copy()

        self.round_num = round_num
        self.logs["round_info"].append(
            {
                "round_count": self.round_num,
                "player_info": [
                    {
                        "sid": sid,
                        "card": player_data["card"],
                        "life": player_data["life"],
                    }
                    for sid, player_data in self.players.items()
                ],
                "deck": deck,
                "turn_info": [],
            }
        )
        self.current_turn_index = 0

    def get_next_turn_player(self):
        if self.active_players:
            return (
                self.active_players[self.current_turn_index % len(self.active_players)][
                    0
                ],
                self.active_players[self.current_turn_index % len(self.active_players)][
                    2
                ],
            )  # sid,is_ai を返す
        return None

    # lifeが0になったプレイヤーをactive_playersから削除
    def death_player(self, player_sid):  # TODO
        player_name = self.players[player_sid]["name"]
        self.active_players = [
            (sid, name, is_ai)
            for sid, name, is_ai in self.active_players
            if sid != player_sid
        ]
        self.death_players = [
            (sid, name, is_ai)
            for sid, name, is_ai in self.active_players
            if sid == player_sid
        ]
        self.death_order[player_sid][len(self.active_players) + 1] += 1
        print(f"{player_name} has died!")

    # 仮
    def reduce_player_life(self, sid, amount=1):
        if sid in self.players:
            self.players[sid]["life"] -= amount
            # ライフが0になったらアクティブプレイヤーから削除
            if self.players[sid]["life"] <= 0:
                self.death_player(sid)
                return True  # プレイヤーが死亡した
            return False  # プレイヤーはまだ生きている
        return None  # プレイヤーが見つからない


class Log:
    def output(log_data, filename=None):
        log_folder = "./log/"
        retry_count = 0

        # ログフォルダが存在しない場合は作成
        if not os.path.exists(log_folder):
            os.makedirs(log_folder)

        while retry_count < 3:
            try:
                filename = str(time.time()) if filename == None else filename
                filefullpath = log_folder + filename + ".json"
                f = open(filefullpath, "x", encoding="UTF-8")
                json.dump(log_data, f)
                f.close()
                return f"Successfully saved the log file as {filefullpath}."
            except FileExistsError:
                retry_count += 1
                time.sleep(0.1)
                return "Retry save the log file"
            except Exception as e:
                return f"Failed to write to the log file: {e}"
        return "Failed to write to the log file after multiple attempts."


class Server(RoomManager):
    def __init__(self, port=5000):
        self.sio = socketio.Server(cors_allowed_origins="*")
        self.app = socketio.WSGIApp(self.sio)
        self.rooms = RoomManager()
        self.sio.on("connect", self.on_connect)  # 接続確認用
        self.sio.on("disconnect", self.on_disconnect)  # 切断確認用
        self.sio.on("leave_room", self.on_leave_room)
        self.sio.on("join_room", self.join_room)
        self.sio.on("game_start", self.game_start)
        self.sio.on("round_start", self.round_start)
        self.sio.on("turn_start", self.turn_start)
        self.sio.on("turn_handling", self.turn_handling)
        self.sio.on("round_end", self.round_end)
        self.sio.on("result", self.result)

        # roomの管理用
        self.sio.on("send_room_info", self.send_room_info)
        self.sio.on("send_rooms", self.send_rooms)
        self.sio.on("log", self.log)
        self.port = port

    def ob_emit(self, func_name, post_data, room_id):
        room = self.rooms.get_room(room_id)
        if room:
            for to_player_sid in room.observers:
                self.sio.emit(func_name, post_data, to=to_player_sid)

    def on_connect(self, sid, environ):
        """
        接続確認用
        """
        print(f"New connection {sid}")
        self.sio.emit("connected", {"sid": sid}, to=sid)

    def on_disconnect(self, sid):
        """
        切断確認用
        """
        print(f"Disconnected {sid}")
        # ルームIDとプレイヤー名を取得するためのロジックを追加
        room_id = None
        player_name = None
        for room in self.rooms.rooms.values():
            for player_sid_, player_data in room.players.items():
                if player_sid_ == sid:
                    room_id = room.room_id
                    player_name = player_data["name"]
                    break
            if room_id and player_name:
                break
        if room_id and player_name:
            self.on_leave_room(
                sid, {"room_id": room_id, "player_name": player_name, "player_sid": sid}
            )
        else:
            print(f"Room ID and player name not found for {sid}")

    def on_leave_room(self, sid, data):
        """
        ルームから退出する
        """
        room_id = data.get("room_id")
        player_name = data.get("player_name")
        player_sid = sid
        # Roomインスタンスを取得
        room = self.rooms.get_room(room_id)

        if room:
            room.remove_player(player_sid)

            player_list = [(p_name, p_sid) for p_sid, p_name, _ in room.active_players]
            post_data = {
                "header": "room_left",
                "msg": f"{player_name} has left the room {room_id}",
                "player_list": player_list,
                "room_id": room_id,
            }
            self.sio.emit("room_left", post_data, room=room_id)
            self.ob_emit("room_left", post_data, room_id)
            # ルームに誰もいなくなったらルームを削除
            print(f"{player_name} has left the room {room_id}")
            if len(room.active_players) == 0:
                self.rooms.close_room(room_id)
                self.sio.emit(
                    "close_room",
                    {
                        "header": "close_room",
                        "msg": f"{room_id} closed",
                        "room_id": room_id,
                    },
                    room=room_id,
                )
                print(f"Close room {room_id}")
            self.sio.leave_room(sid, room_id)

    def join_room(self, sid, data):
        """
        ルームに参加する
        {
            "room_id": "string",
            "player_name": "nunupy"
        }
        """
        room_id = data["room_id"]
        player_name = data["player_name"]
        is_ai = data.get("is_ai", False)
        is_observer = data.get("is_observer", False)

        # すでにルームが存在するか確認
        room = self.rooms.get_room(room_id)
        print("room_id", room_id)
        print("check room", room)
        print("check room.is_started", room.is_started if room else "No room")
        if (
            room and sid not in room.start_players and room.is_started
        ):  # Changed is_start to is_started
            is_observer = True
        print("run join room")
        if not is_observer:
            if room is None:
                print(f"Create new room {room_id}")
                self.rooms.create_room(room_id)
                room = self.rooms.get_room(room_id)

            room.add_player(sid, player_name, is_ai)  # type: ignore #ignore
            player_sid_list = self.rooms.get_room(room_id).players.keys()
            player_name_list = [
                (self.rooms.get_room(room_id).players[player_sid]["name"], player_sid)
                for player_sid in player_sid_list
            ]
            self.sio.enter_room(sid, room_id)
            # ルームにいる全員に通知
            post_data = {
                "header": "join_room",
                "msg": f"{player_name} has joined the room!",
                "room_id": room_id,
                "player_list": player_name_list,
            }
            self.sio.emit("room_message", post_data, room=room_id)
            self.ob_emit("room_message", post_data, room_id)
            print(f"{player_name} has joined the room {room_id}")
        else:
            if room is None:
                print("Falid to join room")
                self.sio.emit(
                    "room_message",
                    {"header": "join_room", "msg": f"There is NO ROOM id: {room_id}"},
                    to=sid,
                )
            else:
                player_sid_list = self.rooms.get_room(room_id).players.keys()
                player_name_list = [
                    (
                        self.rooms.get_room(room_id).players[player_sid]["name"],
                        player_sid,
                    )
                    for player_sid in player_sid_list
                ]
                # self.sio.enter_room(sid, room_id)
                room.observers.append(sid)
                self.sio.emit(
                    "room_message",
                    {
                        "header": "join_room",
                        "msg": f"Success to Join Room (ID: {room_id})",
                        "room_id": room_id,
                        "player_list": player_name_list,
                    },
                    to=sid,
                )

    def game_start(
        self,
        sid="",
        data={},
    ):
        """
        ゲームを開始する
        {
            "room_id": "string",
            "game_num": 10 // 最初のときは全てのゲーム数を入れる。それ以降は今現在のゲーム数を入れる
        }
        """

        room_id = data["room_id"]
        game_num = data["game_num"]

        room = self.rooms.get_room(room_id)
        # room.deck.reset()
        if room:
            # game_numを設定(is_firstがFalseのとき)
            if not room.is_started:
                room.set_game_num(game_num)
                room.is_started = True
            room.add_current_game_index()

            current_game_index = room.current_game_index
            total_game_num = room.game_num
            room.round_num = 0
            # プレイヤーのステータスをリセット
            # すべてのプレイヤーをアクティブに
            room.active_players = [
                (sid, player_data["name"], player_data["is_ai"])
                for sid, player_data in room.players.items()
            ]
            room.start_players = [sid for sid, name, is_ai in room.active_players]
            print(f"room.active_players: {room.active_players}")
            for player_id in room.players:
                room.players[player_id]["life"] = room.initial_life  # 初期ライフに戻す
            post_data = {
                "header": "game_start",
                "msg": f"Game {current_game_index}/{total_game_num} has started in room {room_id}",
                "active_players_list": room.active_players,
                "room_id": room_id,
            }
            self.sio.emit("room_message", post_data, room=room_id)
            self.ob_emit("room_message", post_data, room_id)
            print(
                f"Game {current_game_index}/{total_game_num} has started in room {room_id}"
            )
            # room.deck.reset() #新たなゲーム開始時山札をリセットする
            print("deck reset")
            # round_start
            self.round_start({"room_id": room_id, "round_num": 0})

    def round_start(self, data):
        """
        ラウンドを開始する
        {
            "room_id": "string",
            "round_num": 1
        }
        """
        room_id = data["room_id"]
        round_num = data["round_num"]
        room = self.rooms.get_room(room_id)

        # TODO; coyote実装班から、カードの処理をするようにする
        if room:
            if len(room.active_players) <= 1:
                print(f"Only one player remains in room {room_id}. Ending the game.")
                self.result({"room_id": room_id, "winner": room.active_players})
            post_data = {
                "header": "round_start",
                "msg": "Round start!",
                "round_num": room.round_num,
                "room_id": room_id,
            }
            self.sio.emit("room_message", post_data, room=room_id)
            self.ob_emit("room_message", post_data, room_id)
            print(f"{room_id} round {room.round_num} start!")
            # ライブラリ化してない
            # ゲーム開始時もしくは
            # if round_num == 0 or len(room.deck.cards) < len(room.active_players):
            #     #room.deck.reset()
            #     print("deck reset")
            # 引いたカードをプレイヤーに送る
            for sid, player_name, is_ai in room.active_players:
                room.players[sid]["card"] = room.draw_card()  # カードを引く
                print(f"draw {room.players[sid]['card']}")
                self.sio.emit("draw_card", {"card": room.players[sid]["card"]}, to=sid)
            for sid, player_name, is_ai in room.active_players:
                room.players[sid]["others_card_sum"] = room.convert_card(
                    [
                        player["card"]
                        for other_sid, player in room.players.items()
                        if other_sid != sid
                    ],
                    True,
                )  # 自分以外のすべてのカードの合計値を計算する
                # print(f"sumsum is {room.players[sid]["others_card_sum"]}")

            # ここでplayer["card"]の中身を書き換える(特殊カードの処理)
            room.card_sum = room.convert_card(
                [player["card"] for player in room.players.values()], False
            )  # すべてのカードの合計値を計算する
            room.start_new_round(round_num)
            self.turn_start({"room_id": room_id})

    def turn_start(self, data):
        def get_others_info(player_sid):
            next_sid = (
                room.active_players.index(
                    (
                        player_sid,
                        room.players[player_sid]["name"],
                        room.players[player_sid]["is_ai"],
                    )
                )
                + 1
            ) % len(room.active_players)
            prev_sid = (
                room.active_players.index(
                    (
                        player_sid,
                        room.players[player_sid]["name"],
                        room.players[player_sid]["is_ai"],
                    )
                )
                - 1
            ) % len(room.active_players)
            return [
                {
                    "sid": sid,
                    "card_info": None if sid == player_sid else player_data["card"],
                    "life": player_data["life"],
                    "is_next": sid == next_sid,
                    "is_prev": sid == prev_sid,
                }
                for sid, player_data in room.players.items()
            ]

        """
        ターンを開始する
        {
            "room_id": "string"
        }
        """
        room_id = data["room_id"]
        room = self.rooms.get_room(room_id)

        if room:
            # アクティブプレイヤーが1人以下ならゲーム終了
            if len(room.active_players) <= 1:
                print(f"Only one player remains in room {room_id}. Ending the game.")
                self.result({"room_id": room_id, "winner": room.active_players})
                return

            next_turn = room.get_next_turn_player()
            turn_player_sid, is_turn_player_ai = next_turn
            if turn_player_sid:
                # TODO: cardinfo, lifeなどをcoyote実装班からもらう
                # cardinfo,lifeは仮

                others_info = get_others_info(turn_player_sid)
                min_action = room.last_action + 1
                # TODO legal_actionを正しいものに変更
                room.legal_action = [-1, min_action, 140]
                log_info = room.logs["round_info"][-1]["turn_info"]
                # ai だったら10秒待つ
                if is_turn_player_ai:
                    # time.sleep(1)
                    print(f"ai and time sleep")
                # print(room.convert_card(player["card"] for sid, player in room.players.items()))
                # wait_cards = [player["card"] for sid, player in room.players.items() if sid != turn_player_sid]#配列に入れられるのを待つ
                self.sio.emit(
                    "turn_handling",
                    {
                        "header": "turn",
                        "player_sid": turn_player_sid,
                        "others_info": others_info,
                        "sum": room.players[turn_player_sid][
                            "others_card_sum"
                        ],  # 自分以外のすべてのカードの合計値を計算する
                        "log": log_info,
                        "legal_action": room.legal_action,
                    },
                    to=turn_player_sid,
                )
                # print(f"sum is {room.players[turn_player_sid]["others_card_sum"]}")
                # print(f"{turn_player_sid} turn start!")
                # TODO 追加しました（ドキュメントの更新など）
                for to_player_sid, _ in room.players.items():
                    others_info = get_others_info(to_player_sid)
                    self.sio.emit(
                        "turn_start",
                        {
                            "header": "turn_info",
                            "player_sid": to_player_sid,
                            "turn_player_sid": turn_player_sid,
                            "others_info": others_info,
                            "sum": room.players[to_player_sid][
                                "others_card_sum"
                            ],  # 自分以外のすべてのカードの合計値を計算する
                            "log": log_info,
                            "for_client_info": {
                                "sid": turn_player_sid,
                                "name": room.players[turn_player_sid],
                                "number": room.last_action,
                            },
                        },
                        to=to_player_sid,
                    )
                for to_player_sid in room.observers:
                    others_info = get_others_info(to_player_sid)
                    self.sio.emit(
                        "turn_start",
                        {
                            "header": "turn_info",
                            "player_sid": to_player_sid,
                            "turn_player_sid": turn_player_sid,
                            "others_info": others_info,
                            "sum": room.players[turn_player_sid][
                                "others_card_sum"
                            ],  # 自分以外のすべてのカードの合計値を計算する
                            "log": log_info,
                            "for_client_info": {
                                "sid": turn_player_sid,
                                "name": room.players[turn_player_sid],
                                "number": room.last_action,
                            },
                        },
                        to=to_player_sid,
                    )

    def turn_handling(self, sid, data):
        """
        ターンの処理を行う
        {
            "room_id": "string",
            "action": 1
        }
        """
        room_id = data["room_id"]
        player_sid = sid
        action = data["action"]
        print(f"[TURN ACTION] Player SID: {player_sid}, Action: {action}")
        room = self.rooms.get_room(room_id)
        room.last_action = action
        if room:
            room.logs["round_info"][-1]["turn_info"].append(
                {
                    "turn_count": room.current_turn_index,
                    "turn_player_sid": player_sid,
                    "action": action,
                }
            )
            min_range = room.legal_action[1]
            max_range = room.legal_action[2]
            if action not in [room.legal_action[0], *range(min_range, max_range + 1)]:
                print(f"Invalid action: {action}")
                # ランダムにアクションを選択
                action = random.choice(room.legal_action)

            if action == -1:  # コヨーテ発動
                # TODO: コヨーテの処理,lifeの管理の処理,必要に応じてdeath_playerを呼び出す

                # 最新のターン情報のインデックスを取得
                current_turn_index = len(room.logs["round_info"][-1]["turn_info"]) - 1

                is_coyote_success = False
                # 前のプレイヤーのアクションを取得
                if current_turn_index > 0:
                    prev_player_action = room.logs["round_info"][-1]["turn_info"][
                        current_turn_index - 1
                    ]["action"]
                    print(f"全体の合計値は{room.card_sum}です")
                    if (
                        prev_player_action > room.card_sum
                    ):  # すべての場のカードの合計を計算する
                        is_coyote_success = True
                else:
                    prev_player_action = None

                if is_coyote_success:
                    # コヨーテ成功の場合、前のプレイヤーのライフを減らす
                    prev_turn_index = (room.current_turn_index - 1) % len(
                        room.active_players
                    )
                    prev_player_sid = room.active_players[prev_turn_index][0]
                    player_died = room.reduce_player_life(prev_player_sid)
                else:
                    # コヨーテ失敗の場合、現在のプレイヤーのライフを減らす
                    player_died = room.reduce_player_life(player_sid)

                print(f"{player_sid} is coyote!")
                self.round_end(
                    {"room_id": room_id, "is_coyote_success": is_coyote_success}
                )
            else:
                room.add_current_turn_index()  # 次のターンへ
                # print(f"{player_sid} turn end!")
                self.turn_start({"room_id": room_id})

    def round_end(self, data):
        """
        playerが1人になったらresultへ移行
        そうでなければround_startへ移行
        {
            "room_id": "string"
            "is_coyote_success": bool
        }
        """
        room_id = data["room_id"]
        is_coyote_success = data["is_coyote_success"]
        room = self.rooms.get_room(room_id)
        if room:
            # ラウンド終了処理
            # TODO: cardinfo, lifeなどをcoyote実装班からもらう
            # cardinfo,lifeは仮
            players_info = [
                {
                    "sid": sid,
                    "card_info": player_data["card"],
                    "life": player_data["life"],
                }
                for sid, player_data in room.players.items()
            ]
            post_data = {
                "header": "round_end",
                "is_coyote_success": is_coyote_success,
                "players_info": players_info,
            }
            self.sio.emit("room_message", post_data, room=room_id)
            self.ob_emit("room_message", post_data, room_id)
            print(f"{room_id} round {room.round_num} end!")

            for sid, player_data in room.players.items():
                if (sid == active_sid for active_sid in room.active_players):
                    room.deck.cashed_cards.append(
                        player_data["card"]
                    )  # 引いたカードを捨て札に追加

            # room.Is_firsttime = True
            if room.is_shuffle_card:
                room.deck.reset()
                room.cashed_cards = []  # 捨て札をリセット
                room.is_shuffle_card = False
            room.round_num += 1  # 表記和おかしかったので) print(f"{room_id} round {room.round_num} end!")の後ろに移動
            # ゲーム終了判定
            if len(room.active_players) <= 1:
                print(f"{room_id} game end!")
                self.result({"room_id": room_id, "winner": room.active_players})
            else:
                self.round_start({"room_id": room_id, "round_num": room.round_num})

    def result(self, data):
        """
        ゲームの結果を送信
        試合数が game_num に達してなければstart_gameに戻る, 達していればエンド
        {
            "room_id": "string"
        }
        """
        room_id = data["room_id"]
        winner = data["winner"]
        room = self.rooms.get_room(room_id)
        is_game_continue = False
        if room:
            if room.current_game_index < room.game_num:
                is_game_continue = True

            # TODO: death_ranking, total_win_numをcoyote実装班からもらう
            death_ranking = [
                {"player_name": player_data["name"]}
                for sid, player_data, is_ai in room.death_players
            ]
            room.players[room.active_players[0][0]]["win_num"] += 1

            total_win_num = [
                {
                    "player_name": player_data["name"],
                    "each_ranking": player_data["win_num"],
                }
                for sid, player_data in room.players.items()
            ]

            # ↓完了 ひつじ
            # TODO: logを出す
            # log→logsに修正
            post_data = {
                "header": "game_end",
                "death_ranking": death_ranking,
                "winner": winner,
                "game_num": room.game_num,
                "total_win_num": total_win_num,
                "is_game_continue": is_game_continue,
            }
            self.sio.emit("room_message", post_data, room=room_id)
            self.ob_emit("room_message", post_data, room_id)
            print(f"{room_id} game end!")
            # is_game_continueがTrueならゲームを続ける
            # Falseなら終了.client側でleave_roomを呼ぶ

            room.logs["death_orders"] = room.death_order.copy()
            room.death_order = defaultdict(lambda: defaultdict(int))
            print(Log.output(room.logs))
            if is_game_continue:
                room.log = {"round_info": []}  # room.logのinit
                self.game_start(
                    data={"room_id": room_id, "game_num": room.current_game_index}
                )

    def send_room_info(self, sid, data):
        """
        ルーム情報を送信
        {
            "room_id": "string"
        }
        """
        room_id = data["room_id"]
        print(f"send_room_info {room_id}")
        room = self.rooms.get_room(room_id)
        if room:
            room_info = {
                "room_id": room.room_id,
                "game_num": room.game_num,
                "round_num": room.round_num,
                "players": [
                    {"sid": sid, "name": player["name"], "life": player["life"]}
                    for sid, player in room.players.items()
                ],
                "active_players": [
                    {"sid": sid, "name": name, "is_ai": is_ai}
                    for sid, name, is_ai in room.active_players
                ],
            }
            self.sio.emit("room_info", room_info, to=sid)
        else:
            self.sio.emit("on_room_message", {"error": "Room not found"}, to=sid)

    def send_rooms(self, sid):
        """
        RoomManagerの情報を送信
        """
        rooms_info = []
        for room_id, room in self.rooms.rooms.items():
            rooms_info.append(
                {
                    "room_id": room_id,
                    "game_num": room.game_num,
                    "round_num": room.round_num,
                    "players": [
                        {"name": player["name"], "life": player["life"]}
                        for player in room.players.values()
                    ],
                    "active_players": [
                        {"name": name} for _, name in room.active_players
                    ],
                }
            )

        self.sio.emit("rooms_info", {"rooms": rooms_info}, to=sid)
        print("rooms info sent!")

    def log(self, sid, data):
        """
        ログを送信
        {
            "room_id": "string"
        }
        """
        room_id = data["room_id"]
        room = self.rooms.get_room(room_id)
        if room:
            self.sio.emit(
                "log_info",
                {"header": "log_info", "room_id": room_id, "logs": room.logs},
                to=sid,
            )
            print(f"{room_id} log info sent!")

    def start(self):
        print(f"Starting TestServer on port {self.port}")
        eventlet.wsgi.server(
            eventlet.listen(("", self.port)), self.app, log_output=False
        )  # type: ignore


if __name__ == "__main__":
    server = Server()
    server.start()