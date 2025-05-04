import random
def sample_from_distribution(probs, legal_actions):
    """
    確率分布から行動をサンプリングする
    
    Args:
        probs: 行動→確率の辞書
        legal_actions: 可能な行動のリスト
    
    Returns:
        int: サンプリングされた行動
    """
    # 有効な行動のみを抽出
    valid_actions = []
    valid_probs = []
    
    for action, prob in probs.items():
        if action in legal_actions:
            valid_actions.append(action)
            valid_probs.append(prob)
    
    # 確率の合計が1になるように正規化
    if valid_probs:
        sum_probs = sum(valid_probs)
        if sum_probs > 0:
            valid_probs = [p / sum_probs for p in valid_probs]
        
        # 確率に基づいてアクションをサンプリング
        print("valid_probs",valid_probs)
        chosen_action = random.choices(valid_actions, weights=valid_probs, k=1)[0]
    else:
        # 有効な確率がない場合はランダムに選択
        print("valid_probsが空です。ランダムに選択します。")
        chosen_action = random.choice(legal_actions)
    
    return chosen_action