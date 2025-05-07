from .encode_state import encode_state
from .sample_from_distribution import sample_from_distribution
import logging
def make_decision(state, strategy_net):
    """
    現在の状態から最適な宣言値を決定
    
    Args:
        state: ゲーム状態
        strategy_net: 戦略ネットワーク
    
    Returns:
        int: 選択された宣言値
    """
    # 状態を入力形式に変換
    input_state = encode_state(state)
    
    # 戦略ネットワークから確率分布を取得
    probabilities = strategy_net.predict(input_state, state["legal_action"])
    logging.info(f"probabilities: {probabilities}")
    
    # 確率に基づいて行動を選択
    chosen_action = sample_from_distribution(probabilities, state["legal_action"])
    
    return chosen_action
