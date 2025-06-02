from .encode_state import encode_state
from .sample_from_distribution import sample_from_distribution
import logging
import os
import numpy as np

# ロガーの設定
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ログファイルのパスを絶対パスで設定
log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "make_decision.log")

# ファイルハンドラの設定
file_handler = logging.FileHandler(log_file, encoding='utf-8')
file_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

logger.info("MakeDecision logger initialized")

def make_decision(state, strategy_net):
    
    """
    現在の状態から最適な宣言値を決定
    
    Args:
        state: ゲーム状態
        strategy_net: 戦略ネットワーク
    
    Returns:
        int: 選択された宣言値
    """
    try:
        # 状態を入力形式に変換
        input_state = encode_state(state)
        # logger.info(f"Encoded state shape: {input_state.shape}")
        # logger.info(f"state type: {type(input_state)}")
        # logger.info(f"legal_action: {state['legal_action']}")
        
        # input_stateがnumpy配列であることを確認
        if not isinstance(input_state, np.ndarray):
            input_state = np.array(input_state)
        
        # 戦略ネットワークから確率分布を取得
        probabilities = strategy_net.prediction(input_state, state['legal_action'])
        # logger.info(f"Probabilities type: {type(probabilities)}")
        # logger.info(f"Probabilities content: {probabilities}")
        # logger.info(f"Legal actions: {state['legal_action']}")

        # 確率に基づいて行動を選択
        chosen_action = sample_from_distribution(probabilities, state['legal_action'])
        # logger.info(f"Chosen action: {chosen_action}")
        return chosen_action

    except Exception as e:
        logger.error(f"Error in make_decision: {str(e)}")
        logger.error(f"State: {state}")
        logger.error(f"Input state shape: {input_state.shape if 'input_state' in locals() else 'not created'}")
        logger.error(f"Legal actions: {state['legal_action']}")
        # エラーを再度発生させる
        raise
   
