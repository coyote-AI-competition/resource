from .not_websocket_client import Client
import numpy as np
import random
import tensorflow as tf
import pyspiel
from .coyote_game import *
import random
# from client.coyote_game import *

class CFRClient(Client):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.life_dict = dict()  # Initialize life_ls to None
        self.last_other_info = None  # Initialize last_other_info to None
        self.used_card_ls = []  # Initialize used_card to None

        # debug mode
        self.game = pyspiel.load_game("python_coyote")
        dirname = os.path.dirname(__file__)
        self.ckpt_path = os.path.join(dirname, "checkpoints", "deep_cfr.ckpt")

        # セッションを開いて保持しておく
        self.session = tf.Session()

        # solver 構築（ここでは with を使わない）
        self.solver = deep_cfr.DeepCFRSolver(
            game=self.game,
            session=self.session,
            policy_network_layers=[16, 16],
            advantage_network_layers=[16, 16],
            num_iterations=1,
            num_traversals=10,
            learning_rate=1e-4,
            batch_size_advantage=32,
            batch_size_strategy=32,
            memory_capacity=10000
        )

        self.session.run(tf.global_variables_initializer())

        self.saver = tf.train.Saver()
        if os.path.exists(self.ckpt_path + ".index"):
            print("✅ モデルを読み込み中...")
            self.saver.restore(self.session, self.ckpt_path)
            print("✅ 復元完了！")

    def AI_player_action(self, others_info, sum, log, actions):

        print(f"[CFRClient] AI deciding action based on sum: {sum}, log: {log}, actions: {actions}, others_info: {others_info}")

        # ゲームの状態を作成
        state = CoyoteState(self.game)
                
        # Estimate the used card based on others_info
        used_card = self.used_card_estimate(others_info)
        others_card = [info["card_info"] for info in others_info]

        if len(actions) == 1:
            # 1つの行動しかない場合、その行動を選択
            return actions[0]

        state.set_initial_state(len(others_info)+1, actions[1]-1, used_card, others_card)


        print(f"[CFRClient] State: {state}")
        print(f"[CFRClient] Input tensor: {state.information_state_tensor()}")


        rnd = random.random()

        # 学習済みのポリシーから行動確率を取得
        average_policy = self.solver.action_probabilities(state)

        print("行動確率:", average_policy)

        # 行動確率に基づいて行動を選択
        if rnd < average_policy[0]:
            return actions[1]  # %の確率で最初の行動を選択
        else:
            return actions[0]  # %の確率で2番目の行動を選択
    
    def used_card_estimate(self, others_info):
        #Estimate the used card based on others_info, sum, log, actions

        # if self.life_ls exists:
        if len(self.life_dict) == 0:
            for info in others_info:
                print(f"[CFRClient] Life info: {info}")
                self.life_dict[info["name"]] = info["life"]
            self.last_other_info = others_info  # Store the initial state of others_info

        now_life_dict = {info["name"] :info["life"] for info in others_info}
        # Check if the life_dict has changed
        if self.life_dict != now_life_dict:
            print(f"[CFRClient] Life before: {self.life_dict}")
            self.life_dict = dict()  # Reset life_dict for the new game state
            for info in others_info:
                self.life_dict[info["name"]] = info["life"]
            
            for info in self.last_other_info:
                self.used_card_ls.append(info["card_info"])

            if 101 in self.used_card_ls:
                self.used_card_ls = []

            self.last_other_info = others_info  # Update last_other_info to the current state
            print(f"[CFRClient] Life after: {self.life_dict}")

        visible_cards = [info["card_info"] for info in others_info]
        print(f"[CFRClient] Used card: {self.used_card_ls + visible_cards}")

        return self.used_card_ls + visible_cards

# Statistical Class for evaluating the performance of the AI
class StatClient(Client):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.life_dict = dict()  # Initialize life_ls to None
        self.last_other_info = None  # Initialize last_other_info to None
        self.used_card_ls = []  # Initialize used_card to None
        self.all_card =  [-10, -5, -5, 0, 0, 0,
                      1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3,
                      4, 4, 4, 4, 5, 5, 5, 5, 
                      10, 10, 10, 15, 15, 20, 
                      100, 101, 102, 103]
        self.question = False
        self.double = False
        self.max = False
        self.remove_max = 0

    def AI_player_action(self, others_info, _, log, actions):

        if len(actions) == 1:
            # 1つの行動しかない場合、その行動を選択
            return actions[0]
        
        declare = actions[1]-1

        # Estimate the used card based on others_info
        used_card = self.used_card_estimate(others_info)

        for card in used_card:
            if card in self.all_card:
                self.all_card.remove(card)

        if 101 in self.all_card:
            self.all_card.remove(101)
            self.append(0)

        others_card = [info["card_info"] for info in others_info]
        
        if 100 in others_card:
            self.double = True
            others_card.remove(100)
        if 102 in others_card:
            self.max = True
            others_card.remove(102)
        if 103 in others_card:
            self.question = True
            others_card.remove(103)
        if 101 in others_card:
            others_card.remove(101)

        


        


        return random.choice(actions)
    
    def used_card_estimate(self, others_info):
        #Estimate the used card based on others_info, sum, log, actions

        # if self.life_ls exists:
        if len(self.life_dict) == 0:
            for info in others_info:
                print(f"[StatClient] Life info: {info}")
                self.life_dict[info["name"]] = info["life"]
            self.last_other_info = others_info  # Store the initial state of others_info

        now_life_dict = {info["name"] :info["life"] for info in others_info}
        # Check if the life_dict has changed
        if self.life_dict != now_life_dict:
            print(f"[StatClient] Life before: {self.life_dict}")
            self.life_dict = dict()  # Reset life_dict for the new game state
            for info in others_info:
                self.life_dict[info["name"]] = info["life"]
            
            for info in self.last_other_info:
                self.used_card_ls.append(info["card_info"])

            if 101 in self.used_card_ls:
                self.used_card_ls = []

            self.last_other_info = others_info  # Update last_other_info to the current state
            print(f"[StatClient] Life after: {self.life_dict}")

        visible_cards = [info["card_info"] for info in others_info]
        print(f"[StatClient] Used card: {self.used_card_ls + visible_cards}")

        return self.used_card_ls + visible_cards