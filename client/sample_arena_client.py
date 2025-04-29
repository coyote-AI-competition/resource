from .not_websocket_client import Client
from client_ai.train_deepcfr_for_coyote import train_deepcfr_for_coyote
from client_ai.make_decision import make_decision
from client_ai.StrategyNetwork import StrategyNetwork
from client_ai.CFRTrainingEvaluator import evaluate_cfr_training
import random

class SampleClient(Client):
    def __init__(self, player_name, is_ai=True):
        super().__init__()
        # strategy_netsを初期化
        self.player_name = player_name
        self.is_ai = is_ai
        self.strategy_net = StrategyNetwork(302, 141)  # 必要な引数を指定
        self.previous_round_num = 0
        self.Is_coyoted = None
        self.trajectory_value = 0
        self.prev_others_life = []

    def AI_player_action(self,others_info, sum, log, player_card, actions, round_num):
        # カスタムロジックを実装
        print(f"[SampleClient] AI deciding action based on sum: {sum}, log: {log}, actions: {actions},others_info: {others_info}, round_num: {round_num}" )
        # 例: ランダムにアクションを選択
        # game state
        #others_info, actions, round_num = get_current_state_from_rounds(game_data.get("round_info", []))
        now_round_num = round_num
        if now_round_num != self.previous_round_num:
            self.previous_round_num = now_round_num
            
            self.trajectory_value = 0
            others_life = [info["life"] for info in others_info]
            if (len(self.prev_others_life) == 0):
                self.prev_others_life = others_life
            elif others_life == self.prev_others_life:    
                self.Is_coyoted = True
                self.prev_others_life = others_life
            else:
                self.Is_coyoted = False
                self.prev_others_life = others_life    

        state = {
            "others_info": others_info,
            "legal_action": actions,
            "log": log,  # 既存のlog情報を使用
            "sum": sum,
            "round_num": round_num,
            "player_card": player_card,
            "Is_coyoted": self.Is_coyoted
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
        }
        evaluator = evaluate_cfr_training(iterations=50)
        prediction_fig = visualize_model_prediction(evaluator.strategy_net, test_states)
        prediction_fig.savefig('model_predictions.png')
        self.strategy_net = train_deepcfr_for_coyote(self, current_state = state)

        RED = '\033[31m'
        END = '\033[0m'
        print(RED  + str(select_action) +  END)
        return  select_action
      
