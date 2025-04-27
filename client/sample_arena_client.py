from .not_websocket_client import Client
from client_ai.train_deepcfr_for_coyote import train_deepcfr_for_coyote
from client_ai.make_decision import make_decision
from client_ai.StrategyNetwork import StrategyNetwork
import random

class SampleClient(Client):
    def __init__(self, player_name, is_ai=True):
        super().__init__()
        # strategy_netsを初期化
        self.player_name = player_name
        self.is_ai = is_ai
        self.strategy_net = StrategyNetwork(304, 141)  # 必要な引数を指定
    def AI_player_action(self,others_info, sum, log, player_card, actions, round_num):
        # カスタムロジックを実装
        print(f"[SampleClient] AI deciding action based on sum: {sum}, log: {log}, actions: {actions},others_info: {others_info}, round_num: {round_num}" )
        # 例: ランダムにアクションを選択
        # game state
        #others_info, actions, round_num = get_current_state_from_rounds(game_data.get("round_info", []))
        state = {
            "others_info": others_info,
            "legal_action": actions,
            "log": log,  # 既存のlog情報を使用
            "sum": sum,
            "round_num": round_num,
            "player_card": player_card
        }

        select_action = make_decision(state, self.strategy_net)

        state = {
            "others_info": others_info,
            "legal_action": actions,
            "log": log,  # 既存のlog情報を使用
            "sum": sum,
            "round_num": round_num,
            "player_card": player_card,
            "selectaction": select_action,
            "reword": reword 
        }
        self.strategy_net = train_deepcfr_for_coyote(current_state = state)

        RED = '\033[31m'
        END = '\033[0m'
        print(RED  + str(select_action) +  END)
        return  select_action
      
