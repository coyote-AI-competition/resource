from collections import defaultdict
from .encode_state import encode_state
import numpy as np 
import tensorflow as tf
def calculate_advantages(self, game_states, advantage_net):
    """
    それぞれのプレイヤーの反実仮想アドバンテージを計算する
    
    Args:
        game_states: シミュレーションから得られたゲーム状態のリスト
        advantage_nets: それぞれのアドバンテージネットワーク(モデル)
        num_players: プレイヤーの数
        
    Returns:
        dict: 辞書のキーはプレイヤーの識別子や情報セットを表し、
              値はその情報セットにおけるアドバンテージのリストまたは値
    """
    #キーがない場合そのキーを作成する
    advantages = defaultdict(list)
     
    #報酬を定義する
    def define_reward(state_info):      
        reward = 0.0

        # If player won (has more life than others)
        if state_info["Is_coyoted"] == True:
            self.Is_coyoted = None
            reward -= 100  # Lose
            print("コヨーテされた")
        elif state_info["Is_coyoted"] == False:
            self.Is_coyoted = None  
            reward += 0.001  # Win
        else:
            reward += 0.0             

        # Intermediate rewards based on card value and sum
        # Higher card value is generally better in most card games
        if "player_card" in state_info:
            card_value = state_info["player_card"]
            # Special handling for special cards (100+)
            if isinstance(card_value, int) and card_value >= 100:
                # Special cards might be valuable
                card_bonus = 0.05
            elif isinstance(card_value, int):
                # Regular cards: normalize between -0.05 and 0.05
                card_range = 30  # Assuming cards range from -10 to 20
                card_bonus = (card_value + 10) / card_range * 0.1 - 0.05
            else:
                card_bonus = 0
            reward += card_bonus  

        # Reward for making good declarations relative to the actual sum
        if "sum" in state_info and "selectaction" in state_info:
            game_sum = state_info["sum"]
            declared_value = state_info["selectaction"]
            
            # Penalize declarations that are too high (over the actual sum)
            if declared_value > game_sum:
                
                if declared_value > game_sum * 1.2:
                    # Large penalty for over-declarations
                    reward -= 100   
                    print("宣言値が実際の合計の120%を超えた")
                else:
                    reward -= max(0.5, (declared_value - game_sum)*2)   
                    print("宣言値が実際の合計の120%を超えなかった")
            else:
                # Small reward for close but under declarations
                closeness = 1 - (game_sum - declared_value) / game_sum if game_sum > 0 else 0
                reward += closeness * 0.5
        
        return reward              
     
    # Process game trajectory in reverse to calculate counterfactual values

    # 最後の状態が終了状態なら、その報酬を取得
    if game_states:
        self.trajectory_value = define_reward(game_states[-1])
        print(f"trajectory_value: {self.trajectory_value}")
    
    def predict_action_value(advantage_net, encoded_state):
        # テンソルに変換
        if isinstance(encoded_state, np.ndarray):
            encoded_state = tf.convert_to_tensor(encoded_state, dtype=tf.float32)
        
        # 形状を(None, 318)に調整
        if len(encoded_state.shape) == 1:
            encoded_state = tf.expand_dims(encoded_state, axis=0)  # (1, 318)の形状に
        elif len(encoded_state.shape) == 3:
            encoded_state = tf.reshape(encoded_state, (-1, self.input_size))  # (1, 1, 318) → (1, 318)
        
        output = advantage_net(encoded_state, training=False)
        print(f"Model output: {output.numpy()}")
        return output
    
    for state_info in reversed(game_states):
        state = {
            "others_info": state_info["others_info"],
            "legal_action": state_info["legal_action"],
            "log": state_info["log"],
            "sum": state_info["sum"],
            "round_num": state_info["round_num"],
            "player_card": state_info["player_card"]
        }
        action_taken = state_info["selectaction"]
        
        encoded_state = encode_state(state)
        
        # テンソルに変換（バッチ次元を1つだけ追加）
        if isinstance(encoded_state, np.ndarray):
            encoded_state = tf.convert_to_tensor(encoded_state, dtype=tf.float32)
        
        # 形状を(None, 318)に調整
        if len(encoded_state.shape) == 1:
            encoded_state = tf.expand_dims(encoded_state, axis=0)  # (1, 318)の形状に
        elif len(encoded_state.shape) == 3:
            encoded_state = tf.reshape(encoded_state, (-1, self.input_size))  # (32, 1, 318) → (32, 318)
        
        # バッチ処理時の形状を調整
        if len(encoded_state.shape) == 2 and encoded_state.shape[0] > 1:
            encoded_state = tf.reshape(encoded_state, (-1, self.input_size))  # (32, 318)の形状を維持
        
        action_values = predict_action_value(advantage_net, encoded_state).numpy()[0]
        print(f"Action values: {action_values}") #ここの値を評価する
        
        # Calculate advantages (counterfactual regret)
        legal_actions = state["legal_action"]
        
        # For each legal action, estimate its advantage
        advantage_vector = np.zeros(141)  # 例: 141
        for action in legal_actions:

            if action == action_taken:
                advantage = self.trajectory_value - action_values[action]# AIが選択した行動をdifine_rewardで計算した報酬から引く
            else:
                advantage = -action_values[action]

            advantage_vector[action] = advantage
        # Store advantage for this information set and action
        info_set_key = f"player_state{hash(str(encoded_state.numpy().tobytes()))}"
        advantages[info_set_key].append((encoded_state, advantage_vector))

    return advantages