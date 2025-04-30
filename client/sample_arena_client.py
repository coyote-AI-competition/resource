import os
import numpy as np
import threading
from pathlib import Path
from collections import deque
import tensorflow as tf
from .not_websocket_client import Client
from client_ai.train_deepcfr_for_coyote import train_deepcfr_for_coyote
from client_ai.make_decision import make_decision
from client_ai.StrategyNetwork import StrategyNetwork
from client_ai.create_advantage_network import create_advantage_network
from client_ai.CFRTrainingEvaluator import evaluate_cfr_training,visualize_model_prediction
from datetime import datetime
import random

class SampleClient(Client):
    def __init__(self, player_name, is_ai=True):
        super().__init__()
        # strategy_netsを初期化
        self.player_name = player_name
        self.is_ai = is_ai
        # Create directories if they don't exist
        Path("models").mkdir(exist_ok=True)
        Path("save_picture").mkdir(exist_ok=True)
        self.strategy_net = StrategyNetwork(302, 141)  # 必要な引数を指定
        self.advantage_net = create_advantage_network()
        # Try to load pre-trained model if it exists
        self._load_model()
        self.previous_round_num = 0
        self.Is_coyoted = None
        self.trajectory_value = 0
        self.prev_others_life = []
        # Experience buffer for training
        self.experience_buffer = deque(maxlen=1000)
        
        # Training controls
        self.train_counter = 0
        self.train_frequency = 10  # Train every 10 actions
        self.training_in_progress = False
        self.game_state = []

    def _load_model(self):
        """Try to load previously trained model"""
        model_path = "models/model.keras"
        if os.path.exists(model_path):
            try:
                self.strategy_net.model = tf.keras.models.load_model(model_path)
                print(f"Loaded existing model from {model_path}")
            except Exception as e:
                print(f"Error loading model: {e}")

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

        self.experience_buffer.append(state.copy())
        select_action = make_decision(state, self.strategy_net)

        state = {
            "others_info": others_info,
            "legal_action": actions,
            "log": log,  # 既存のlog情報を使用
            "sum": sum,
            "round_num": round_num,
            "player_card": player_card,
            "selectaction": select_action,
            "Is_coyoted": self.Is_coyoted,
        }
        self.game_state.append(state.copy())
        evaluator = evaluate_cfr_training(self.game_state)
        prediction_fig = visualize_model_prediction(evaluator.strategy_net, self.game_state)
       
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        # ファイル名に日付を追加
        file_name = f"save_picture/model_predictions_{current_time}.png"        # Save the figure
        prediction_fig.savefig(file_name)
        self.strategy_net = train_deepcfr_for_coyote(self, current_state = state)

        RED = '\033[31m'
        END = '\033[0m'
        print(RED  + str(select_action) +  END)
        return  select_action
      
