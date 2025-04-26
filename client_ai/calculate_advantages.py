from collections import defaultdict
from .encode_state import encode_state
import numpy as np 
def calculate_advantages(game_states, advantage_net):
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
    
    # Process game trajectory in reverse to calculate counterfactual values
    trajectory_value = 0  # Final game value
    
    for state_info in reversed(game_states):
        state = {
            "others_info": state_info["others_info"],
            "legal_action": state_info["legal_action"],
            "log": state_info["log"],
            "sum": state_info["sum"],
            "round_num": state_info["round_num"],
            "player_card": state_info["player_card"]
        }
        action_taken = state_info["action"]
        
        # Encode state for network input
        encoded_state = encode_state(state)
        
        # Get current strategy's action values
        action_values = advantage_net(np.expand_dims(encoded_state, axis=0)).numpy()[0]
        
        # Calculate advantages (counterfactual regret)
        # In a real implementation, you would calculate the actual counterfactual values
        # by considering alternative actions' outcomes
        legal_actions = state["legal_action"]
        
        # For each legal action, estimate its advantage
        for action in legal_actions:
            # This is a simplified advantage calculation
            # In real CFR, you would compute the true counterfactual value
            if action == action_taken:
                advantage = trajectory_value - action_values[action]
            else:
                # Estimate counterfactual value for actions not taken
                # This is a simplified approach; real CFR would simulate alternative outcomes
                advantage = -action_values[action]
            
            # Store advantage for this information set and action
            info_set_key = f"player_state{hash(str(encoded_state.tobytes()))}"
            advantages[info_set_key].append((action, advantage, encoded_state))
    
    return advantages