import random
import logging
import os

# ロガーの設定
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ログファイルのパスを絶対パスで設定
log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "sample_from_distribution.log")

# ファイルハンドラの設定
file_handler = logging.FileHandler(log_file, encoding='utf-8')
file_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
file_handler.setFormatter(formatter)
#logger.addHandler(file_handler)

#logger.info("SampleFromDistribution logger initialized")

def sample_from_distribution(probs, legal_actions):
    """
    確率分布から行動をサンプリングする
    
    Args:
        probs: 行動→確率の辞書
        legal_actions: 可能な行動のリスト
    
    Returns:
        int: サンプリングされた行動
    """
    try:
        # logger.info(f"Input probabilities: {probs}")
        # logger.info(f"Legal actions: {legal_actions}")
        
        # 有効な行動のみを抽出
        valid_actions = []
        valid_probs = []
        
        for action, prob in probs.items():
            if action in legal_actions:
                valid_actions.append(action)
                valid_probs.append(float(prob))  # 確率をfloat型に変換
        
        # logger.info(f"Valid actions: {valid_actions}")
        # logger.info(f"Valid probabilities: {valid_probs}")
        
        # 確率の合計が1になるように正規化
        if valid_probs:
            sum_probs = sum(valid_probs)#1になる
            #logger.info(f"Sum of valid probabilities: {sum_probs}")
            
            if sum_probs > 0:
                #念のためもう一度確率を正規化
                valid_probs = [p / sum_probs for p in valid_probs]
                #logger.info(f"Normalized probabilities: {valid_probs}")
            
            # 確率に基づいてアクションを一つ取得
            chosen_action = random.choices(valid_actions, weights=valid_probs, k=1)[0]
            #logger.info(f"Chosen action: {chosen_action}")
        else:
           
            logger.info("No valid probabilities, choosing smallest action")
            chosen_action = legal_actions[1]
        
        return chosen_action
        
    except Exception as e:
        logger.error(f"Error in sample_from_distribution: {str(e)}")
        logger.error(f"Probabilities: {probs}")
        logger.error(f"Legal actions: {legal_actions}")
        raise