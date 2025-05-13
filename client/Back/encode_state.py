import numpy as np
def encode_state(state):
    """
    ゲーム状態をdeepCFRのニューラルネットワーク入力形式に変換
    
    Args:
        state: ゲーム状態（辞書形式）
            - others_info: 他プレイヤーの情報
            - sum: 他プレイヤーのカード合計
            - log: ゲームログ情報
            - legal_action: 可能な行動リスト
    
    Returns:
        np.array: エンコードされた状態ベクトル
    """
    others_info = state["others_info"]
    sum_val = state["sum"]
    log_info = state["log"]
    round_num = state["round_num"]
    legal_action = state["legal_action"]
    player_card = state["player_card"]
    
    # 1. 他プレイヤー情報のエンコード
    card_values = [-10, -5, 0, 1, 2, 3, 4, 5, 10, 15, 20, 100, 101, 102, 103]
    player_features = []
    for player in others_info:
        # 特殊カード番号（100-103）を含めたエンコーディング
        card_value = player["card_info"]
        
        # カード値のone-hotエンコーディング（-10から20までの通常カードと特殊カード）
        # 通常カードの値: -10, -5, 0, 1, 2, 3, 4, 5, 10, 15, 20
        # 特殊カード値: 100(×2), 101(max→0), 102(?), 103(黒背景0)
        card_encoding = [0] * len(card_values)
        
        if card_value in card_values:
            card_idx = card_values.index(card_value)
            card_encoding[card_idx] = 1
        
        # プレイヤーの位置関係（次のプレイヤーか前のプレイヤーか）
        position_encoding = [
            1 if player["is_next"] else 0,
            1 if player["is_prev"] else 0
        ]
        
        # ライフ情報（正規化）
        life_normalized = player["life"] / 3  # 一般的な最大ライフ
        
        # このプレイヤーの特徴をまとめる
        player_features.extend(card_encoding + position_encoding + [life_normalized])
    
    # 自分の持つカードをエンコード
    player_card_encoding = [0] * len(card_values)
    if player_card in card_values:
        player_card_idx = card_values.index(player_card)
        player_card_encoding[player_card_idx] = 1

    player_features.extend(player_card_encoding)

    # プレイヤー数が変わる可能性があるため、常に5人分の情報を確保（6人対戦で自分を除く）
    # 1人あたりの特徴量サイズ = カード値(15) + 位置(2) + ライフ(1) = 18
    # 自分のカード情報(15)を加えた後のプレイヤー特徴量サイズ
    max_players = 5
    features_per_player = len(card_values) + 2 + 1 
    expected_size = max_players * features_per_player + len(card_values)

    padding_size = expected_size - len(player_features)
    if padding_size > 0:
        player_features.extend([0] * padding_size)
    
    # 2. 合計値の正規化
    theoretical_max_sum = 139  # 適切な値に調整（ゲーム状況による）
    sum_normalized = sum_val / theoretical_max_sum
    
    # 3. ゲームログの処理
    # ラウンド数の正規化
    round_normalized = round_num / 18  # 想定最大ラウンド数で正規化
    
    # ターン情報のエンコード（最新の10ターン分）
    max_turns = 10
    turn_features = []
    
    # 安全にturn_infoにアクセス
    recent_turns = []
    if isinstance(log_info, dict) and "turn_info" in log_info and isinstance(log_info["turn_info"], list):
        recent_turns = log_info["turn_info"][-max_turns:]
    
    for turn in recent_turns:
        # プレイヤー名をIDに変換（簡易的な実装）
        player_names = []
        if isinstance(log_info, dict) and "player_info" in log_info:
            player_names = [p["name"] for p in log_info["player_info"] if isinstance(p, dict) and "name" in p]
        
        player_id = player_names.index(turn["turn_player"]) if turn["turn_player"] in player_names else 0
        
        # プレイヤーIDをone-hot
        player_one_hot = [0] * 6  # 6人プレイヤー
        if 0 <= player_id < 6:
            player_one_hot[player_id] = 1
        
        # 宣言値の正規化
        declared_value = turn["declared_value"] / 139  # 想定最大宣言値
        
        turn_features.extend(player_one_hot + [declared_value])
    
    # 履歴が足りない場合はパディング
    padding_size = max_turns * 7 - len(turn_features)  # 7 = 6(プレイヤーID) + 1(宣言値)
    if padding_size > 0:
        turn_features.extend([0] * padding_size)
    
    # 4. 可能なアクション（legal_action）のマスク
    max_action_value = 139  # 想定される最大宣言値
    action_mask = [0] * (max_action_value + 1)
    
    for action in legal_action:
        if 0 <= action <= max_action_value:
            action_mask[action+1] = 1
        elif action == -1:
            action_mask[0] = 1    
    
    # すべての特徴を連結
    features = player_features + [sum_normalized , round_normalized] + turn_features + action_mask
    
    # 形状を(None, 318)に調整
    features = np.array(features, dtype=np.float32)
    features = np.expand_dims(features, axis=0)  # (1, 318)の形状に
    return features