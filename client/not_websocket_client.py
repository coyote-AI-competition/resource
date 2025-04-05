import random
import threading
import socketio
import time
import coyote
import asyncio
from tqdm import tqdm

class Client:
    def __init__(self, player_name="player1", is_ai=False):
        self.hold_card = 0
        self.is_started = False
        self.action_start = False  
        self.is_ai = is_ai
        self.timer = None
        self.hold_card = 0
        self.player_name = player_name

    def draw_card(self, data):
        return coyote.game.client_draw_card(self, data) #カードを引く

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
        # tqdm.write(f"Other players: {others_info}")
        sum = turn_data["sum"]
        tqdm.write(f"{self.player_name} :Sum of other players' cards: {sum}")
        log_info = turn_data["log"]
        # tqdm.write(f"Log: {log_info}")　// ログは長いのでコメントアウト
        legal_actions = turn_data["legal_action"]
        # tqdm.write(f"Possible actions: {legal_actions}")

        min_range = legal_actions[1]
        max_range = legal_actions[2]
        if min_range > 140:
            actions = [-1]
        else:
            actions = [legal_actions[0], *range(min_range, max_range+1)]
        # tqdm.write(f"actions: {actions}")
        tqdm.write(f"{self.player_name} :Possible actions: {actions}")
        #プレイヤーに引数を渡す
        if self.is_ai: #AIの場合
            action = self.AI_player_action(others_info,sum,log_info, actions)
            if action not in actions: #アクションが不正な場合
                return -1
            tqdm.write(f"{self.player_name} selected action: {action}")
            return action 

        else: #プレイヤーの場合
            action = self.player_action(sum,log_info, actions)
            return action

    # playerに引数を渡し、actionという返り値を受け取る
    # ここではactionを仮に1000としてlegal_actionの範囲外の値を返す
    def AI_player_action(self,others_info,sum,log, actions):
        action = random.choice(actions)
        return action
    
    #人間が行動する場合
    #inputを打ち込みenterを押すとルームから退室してしまう
    def player_action(self,sum,log, actions):
        try:
            action = int(input(f"Enter action ({actions}): "))
            if action not in actions: #アクションが不正な場合
                    tqdm.write(f"Invalid action: {action}")
                    # ランダムにアクションを選択
                    action = random.choice(actions)  
            return action 
        except  ValueError:
            tqdm.write("Invalid input. Selecting a random action.")
