import random
import threading
import socketio
import time
import coyote
import asyncio

class Client:
    def __init__(self,port=5000, room_id="test-room", player_name="player1", is_ai=False):
        self.sio = socketio.Client(logger=True, engineio_logger=False)
        self.room_id = room_id
        self.player_name = player_name
        self.server_url = f"http://localhost:{port}"
        self.hold_card = 0
        self.is_started = False
        self.action_start = False  # プレイヤーが行動したかを判定
        self.is_ai = is_ai
        self.timer = None
        self.hold_card = 0 #悠長になる
        self.player_sid = None
        
        # サーバーからのイベントを受信
        self.sio.on("connect", self.on_connect)
        self.sio.on("disconnect", self.on_disconnect)
        self.sio.on("join_room", self.join_room)
        self.sio.on("room_message", self.on_room_message)
        self.sio.on("turn_start", self.on_turn_start)
        self.sio.on("turn_handling", self.handle_turn)
        self.sio.on("room_info", self.room_info)
        self.sio.on("rooms_info", self.rooms_info)
        self.sio.on("draw_card", self.draw_card) #カードを引く
        self.sio.on("close_room", self.on_close_room)
        self.sio.on("connected",self.on_connected)

    def on_connected(self, data):
        self.player_sid = data["sid"]
    
    def on_connect(self):
        print('Connected to server')
        
    def on_disconnect(self):
        self.leave_room()
        print('Disconnected from server')

    def on_room_message(self, data):
        header = data.get("header")
        msg = data.get("msg")
        if header == "game_start":
            print(msg)
            self.is_started = True
        elif header == "round_start":
            print(msg)
        elif header == "round_end" or header == "join_room":
            print(msg)
        elif header == "log":
            print("Game log:")
            for log in data["log"]:
                print(log)
        else:
            print('Message from server:', data)

    def join_room(self):
        """ ルームに参加 """
        self.sio.emit("join_room", {"room_id": self.room_id, "player_name": self.player_name, "is_ai": self.is_ai})

    def start_game(self):
        """ ゲーム開始リクエストを送信 """
        game_num = int(input("Enter number of games: ") or 1)
        self.sio.emit("game_start", {"room_id": self.room_id, "game_num": game_num})

    def draw_card(self, data):
        return coyote.game.client_draw_card(self, data) #カードを引く

    def on_turn_start(self, data):
        """
        {
            "header": "turn_info",
            "player_sid": to_player_sid,
            "turn_player_sid": turn_player_sid,
            "others_info": others_info,
            "sum": room.convert_card(player["card"] for sid, player in room.players.items() if sid != turn_player_sid), #自分以外のすべてのカードの合計値を計算する
            "log": log_info
        },
        """
        turn_player = data["turn_player_sid"]
        others_info = data["others_info"]
        now_sum = data["sum"]
        now_log = data["log"]
        print(f"Now Player: {turn_player}")
        print(f"Other Players Info: {others_info}")
        print(f"Now OthersSum: {now_sum}") #デバック用
        print(f"now log: {now_log}")


    def handle_turn(self, turn_data):
        """
        プレイヤーがターンで行動 
        {
                            "header": "turn",
                            "player_sid": turn_player_sid,
                            "others_info": others_info,
                            "sum": sum([player["card_info"] for player in others_info]),
                            "log": log_info,
                            "legal_action": legal_action
                        },
        """
        others_info = turn_data["others_info"]
        print(f"Other players: {others_info}")
        sum = turn_data["sum"]
        print(f"Sum of other players' cards: {sum}")
        log_info = turn_data["log"]
        print(f"Log: {log_info}")
        legal_actions = turn_data["legal_action"]
        print(f"Possible actions: {legal_actions}")

        min_range = legal_actions[1]
        max_range = legal_actions[2]
        if min_range > 140:
            actions = [-1]
        else:
            actions = [legal_actions[0], *range(min_range, max_range+1)]
        print(f"Legal actions: {legal_actions}")

        #プレイヤーに引数を渡す
        if self.is_ai: #AIの場合
            action = self.AI_player_action(others_info,sum,log_info, actions)
            if action not in actions: #アクションが不正な場合
                print(f"Invalid action: {action}")
                # ランダムにアクションを選択
                action = random.choice(actions)
                print(f"Invalid action: {action}")
            self.sio.emit("turn_handling", 
                {"room_id": self.room_id, "action": action})    

        else: #プレイヤーの場合
            asyncio.run(self.player_action(sum,log_info, actions))
                         

    def on_close_room(self, data):
        print(data["msg"])

    # playerに引数を渡し、actionという返り値を受け取る
    # ここではactionを仮に1000としてlegal_actionの範囲外の値を返す
    def AI_player_action(self,others_info,sum,log, actions):
        action = 1000
        return action
    
    #人間が行動する場合
    #inputを打ち込みenterを押すとルームから退室してしまう
    async def player_action(self,sum,log, actions):
        self.action_start = False  # プレイヤーが行動したかを判定
        print("You have 10 seconds to enter your action.")

        if self.timer:
         self.timer.cancel()
        try:
            self.timer = threading.Timer(10, lambda: self.auto_action(actions))
            self.timer.start()
            print("Timer started.")
            action = int(input(f"Enter action ({actions}): "))
            if action not in actions: #アクションが不正な場合
                    print(f"Invalid action: {action}")
                    # ランダムにアクションを選択
                    action = random.choice(actions)   
        except  ValueError:
            print("Invalid input. Selecting a random action.")
            action = random.choice(actions)

        self.timer.cancel()
        print("Timer canceld.")
                # ↓以下をコメントアウトすると、自動行動ができる
        self.sio.emit("turn_handling", 
                    {"room_id": self.room_id, "action": action})
        self.action_start = True

    # タイマーで 10 秒後に自動行動
    def auto_action(self,actions):
        if not self.action_start:  # すでに行動していればスキップ
            action = random.choice(actions)
            print(f"[Timeout] No action taken, selecting randomly: {action}")
            self.sio.emit("turn_handling", {
                "room_id": self.room_id, 
                "action": action
            })
            self.action_start = True

    def leave_room(self):
        """ ルームから退出 """
        self.sio.emit("leave_room", {
            "room_id": self.room_id, 
            "player_name": self.player_name,
            "player_sid": self.player_sid
        })
        
    def get_log(self):
        """ ゲームのログをリクエスト """
        self.sio.emit("log", {"room_id": self.room_id, "player_sid": self.player_name})

    def connect(self):
        """ サーバーに接続 """
        try:
            self.sio.connect(self.server_url)
        except socketio.exceptions.ConnectionError: # type: ignore
            print("Failed to connect. Check server status and IP/port.")

    def run(self):
        """ コマンド入力ループ """
        while self.is_started == False:
            command = input("\nEnter command (start, log, exit): ").strip().lower()
            if command == "start":
                self.start_game()
            elif command == "log":
                self.get_log()
            elif command == "exit":
                print("🔌 Disconnecting...")
                self.leave_room() # ルームから退出することをサーバーに通知
                self.sio.disconnect()
                break
    def get_rooms_info(self):
        """ サーバーにすべてのルーム情報を要求 """
        self.sio.emit("send_rooms")

    def rooms_info(self, data):
        """ すべてのルーム情報を受信 """
        print("All Rooms Info:")
        for room in data["rooms"]:
            print(f"\nRoom ID: {room['room_id']}\n")
            print(f"Game Number: {room['game_num']}\n")
            print(f"Round Number: {room['round_num']}\n")
            print("Players:")
            for player in room["players"]:
                print(f"  - {player['name']} (Life: {player['life']})\n")
            print("Active Players:")
            for player in room["active_players"]:
                print(f"  - {player['name']}\n")
                
    def get_room_info(self, room_id):
        """ サーバーにルーム情報を要求 """
        self.sio.emit("send_room_info", {"room_id": room_id})

    def room_info(self, data):
        """ ルーム情報を受信 """
        print("Room Info:")
        print(f"Room ID: {data['room_id']}")
        print(f"Game Number: {data['game_num']}")
        print(f"Round Number: {data['round_num']}")
        print("Players:")
        for player in data["players"]:
            print(f"  - {player['name']} (Life: {player['life']})")
        print("Active Players:")
        for player in data["active_players"]:
            print(f"  - {player['name']}")
    
    def observer(self):
        """ ゲームを観戦 """
        while True:
            command = input("\nEnter command (1: room_info, 2: rooms_info, exit): ").strip().lower()
            if command == "1":
                room_id = input("Enter room ID: ")
                self.get_room_info(room_id)
            elif command == "2":
                self.get_rooms_info()
            elif command == "exit":
                print("🔌 Disconnecting...")
                self.sio.disconnect()
                break
