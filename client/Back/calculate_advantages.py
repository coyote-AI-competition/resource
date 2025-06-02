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

        #もしコヨーテされたら報酬を減らす
        if state_info["Is_coyoted"] == True:
            self.Is_coyoted = None
            reward -= 10  
            print("コヨーテされた")
        elif state_info["Is_coyoted"] == False:
            self.Is_coyoted = None  
            reward += 0.001  
        else:
            reward += 0.0             

        #宣言値ごとに報酬を定義する
        if "sum" in state_info and "selectaction" in state_info:
            game_sum = state_info["sum"]
            declared_value = state_info["selectaction"]
            
            if declared_value > game_sum:
                
                if declared_value > game_sum * 1.2:
                    #場の合計より大幅に大きい場合
                    reward -= 100 
                    print("宣言値が実際の合計の120%を超えた")
                else:
                    #場の合計より大きい場合
                    reward -= max(0.5, (declared_value - game_sum))*10   
                    print("宣言値が実際の合計の120%を超えなかった")
            else:
                #宣言値が場の合計より小さい場合
                reward += 1
        
        return reward              
     

    # 最後の状態が終了状態なら、その報酬を取得
    if game_states:
        #即時報酬を定義する
        self.trajectory_value = define_reward(game_states[-1])
    
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
        # 例: NumPy配列からTensorFlowテンソルへの変換Tensor
        # NumPy配列: np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        # 変換後: <tf.Tensor: shape=(2, 3), dtype=float32, numpy=array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=float32)>
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
        
        output = advantage_net(encoded_state, training=False)
        action_values = output.numpy()[0]
        print(f"Action values: {action_values}") #ここの値を評価する
        
   
        legal_actions = state["legal_action"]
      
        advantage_vector = np.zeros(141)  # 例: 141
        for action in legal_actions:

            if action == action_taken:
                advantage = self.trajectory_value - action_values[action]# AIが選択した行動をdifine_rewardで計算した報酬から引く
            else:
                advantage = -action_values[action]

            advantage_vector[action] = advantage
        
        info_set_key = f"player_state{hash(str(encoded_state.numpy().tobytes()))}"
        advantages[info_set_key].append((encoded_state, advantage_vector))

    return advantages