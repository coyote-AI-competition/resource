import numpy as np
import json
from typing import Dict, List, Union, Any

# 文字列型の引数を受け取り、JSONファイルを読み込んで辞書型で返す
def load_game_states_from_json(json_file_path: str) -> Dict[str, Any]:
    """
    JSONファイルからゲームのラウンド情報を読み込む
    
    Args:
        json_file_path: ゲーム情報を含むJSONファイルのパス
    
    Returns:
        Dict: ゲーム情報を含む辞書
    """
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            game_data = json.load(f)
        
        print(f"Successfully loaded game data from {json_file_path}")
        return game_data
    except Exception as e:
        print(f"Error loading JSON file: {e}")
        return {"round_info": []}

def get_current_state_from_rounds(round_info: List[Dict[str, Any]], round_idx: int = -1, player_name: str = None) -> Dict[str, Any]:
    """
    ラウンド情報から特定のラウンドの状態を抽出する
    
    Args:
        round_info: ラウンド情報のリスト
        round_idx: 抽出するラウンドのインデックス（デフォルトは最新ラウンド）
        player_name: 視点となるプレイヤー名（指定がない場合は最初のプレイヤー）
    
    Returns:
        Dict: ゲーム状態を表す辞書
    """
    if not round_info:
        return {}
    
    # 指定されたラウンド（デフォルトは最新）を取得
    if round_idx < 0:
        round_idx = len(round_info) + round_idx
    
    if round_idx < 0 or round_idx >= len(round_info):
        round_idx = len(round_info) - 1
    
    current_round = round_info[round_idx]
    
    # プレイヤー情報を取得
    players = current_round.get("player_info", [])
    if not players:
        return {}
    
    # 視点となるプレイヤーを決定
    current_player_idx = 0
    if player_name:
        # enumerateで要素とインデックスを取得
        for i, player in enumerate(players):
            if player.get("name") == player_name:
                current_player_idx = i
                break
    
    current_player = players[current_player_idx]
    
    # 他のプレイヤーの情報を構築
    others_info = []
    for i, player in enumerate(players):
        if i != current_player_idx:
            # 位置関係を計算
            player_count = len(players)
            is_next = (i == (current_player_idx + 1) % player_count)
            is_prev = (i == (current_player_idx - 1) % player_count)
            
            others_info.append({
                "card_info": player.get("card", 0),
                "is_next": is_next,
                "is_prev": is_prev,
                "life": player.get("life", 0)
            })
    
    # ターン情報からゲームログを構築
    turn_info = current_round.get("turn_info", [])
    
    # 合計値を計算
    sum_val = (player.get("card", 0) for player in players if isinstance(player.get("card", 0), (int, float)))
    
    # 可能なアクションを計算（ここでは簡易的に）
    # 実際のゲームルールに基づいて調整が必要
    last_action = -1
    if turn_info:
        last_action = turn_info[-1].get("action", -1)
    
    # 次の合法的なアクションを計算
    legal_action = []
    if last_action < 0:
        # 最初のアクション
        legal_action = list(range(1, 6))
    else:
        # 次のアクションは前のアクション以上でなければならない
        max_action = 140  # 適切な値に調整
        legal_action = list(range(last_action + 1, min(last_action + 6, max_action + 1)))
    
    # 過去のラウンド情報も含めてログを構築
    all_round_turn_info = []
    for r in round_info[:round_idx+1]:
        all_round_turn_info.extend(r.get("turn_info", []))
    
    game_state = {
        "others_info": others_info,
        "sum": sum_val,
        "log": {
            "round_count": current_round.get("round_count", 0),
            "turn_info": [
                {
                    "turn_player": turn.get("turn_player", ""),
                    "declared_value": turn.get("action", 0)
                } for turn in all_round_turn_info
            ],
            "player_info": [{"name": player.get("name", "")} for player in players]
        },
        "legal_action": legal_action,
        "current_player": current_player.get("name", "")
    }
    
    return game_state

def encode_state(state: Union[Dict[str, Any], str]) -> np.ndarray:
    """
    ゲーム状態をdeepCFRのニューラルネットワーク入力形式に変換
    
    Args:
        state: ゲーム状態（辞書形式またはJSONファイルパス）
    
    Returns:
        np.array: エンコードされた状態ベクトル
    """
    # 文字列の場合はJSONファイルとして読み込む
    if isinstance(state, str):
        game_data = load_game_states_from_json(state)
        if not game_data or "round_info" not in game_data:
            raise ValueError(f"Could not load game state from {state}")
        
        # ラウンド情報から現在の状態を構築
        state = get_current_state_from_rounds(game_data.get("round_info", []))
    
    # 辞書からキーを取得
    others_info = state.get("others_info", [])
    sum_val = state.get("sum", 0) # 合計値いらない
    log_info = state.get("log", {"round_count": 0, "turn_info": [], "player_info": []})
    legal_action = state.get("legal_action", [])
    
    # 1. 他プレイヤー情報のエンコード
    player_features = []
    for player in others_info:
        # 特殊カード番号（100-103）を含めたエンコーディング
        card_value = player.get("card_info", 0)
        
        # カード値のone-hotエンコーディング（-10から20までの通常カードと特殊カード）
        # 通常カードの値: -10, -5, 0, 1, 2, 3, 4, 5, 10, 15, 20
        # 特殊カード値: 100(×2), 101(max→0), 102(?), 103(黒背景0)
        card_values = [-10, -5, 0, 1, 2, 3, 4, 5, 10, 15, 20, 100, 101, 102, 103]
        card_encoding = [0] * len(card_values)
        
        if card_value in card_values:
            card_idx = card_values.index(card_value)
            card_encoding[card_idx] = 1
        
        # プレイヤーの位置関係（次のプレイヤーか前のプレイヤーか）
        position_encoding = [
            1 if player.get("is_next", False) else 0,
            1 if player.get("is_prev", False) else 0
        ]
        
        # ライフ情報（正規化）
        life_normalized = player.get("life", 0) / 3  # 一般的な最大ライフ
        
        # このプレイヤーの特徴をまとめる
        player_features.extend(card_encoding + position_encoding + [life_normalized])
    
    # プレイヤー数が変わる可能性があるため、常に5人分の情報を確保（6人対戦で自分を除く）
    # 足りない場合は0パディング
    max_players = 5
    padding_size = max_players * (len(card_values) + 2 + 1) - len(player_features)
    if padding_size > 0:
        player_features.extend([0] * padding_size)
    
    # 2. 合計値の正規化
    theoretical_max_sum = 140  # 適切な値に調整（ゲーム状況による）
    sum_normalized = sum_val / theoretical_max_sum
    
    # 3. ゲームログの処理
    # ラウンド数の正規化
    round_normalized = log_info.get("round_count", 0) / 10  # 想定最大ラウンド数で正規化
    
    # ターン情報のエンコード（最新の10ターン分）
    max_turns = 10
    turn_features = []
    
    recent_turns = log_info.get("turn_info", [])[-max_turns:] if "turn_info" in log_info else []
    for turn in recent_turns:
        # プレイヤー名をIDに変換（簡易的な実装）
        player_names = [p.get("name", "") for p in log_info.get("player_info", [])]
        player_id = player_names.index(turn.get("turn_player", "")) if turn.get("turn_player", "") in player_names else 0
        
        # プレイヤーIDをone-hot
        player_one_hot = [0] * 6  # 6人プレイヤー
        if 0 <= player_id < 6:
            player_one_hot[player_id] = 1
        
        # 宣言値の正規化
        declared_value = turn.get("declared_value", 0) / 140  # 想定最大宣言値
        
        turn_features.extend(player_one_hot + [declared_value])
    
    # 履歴が足りない場合はパディング
    padding_size = max_turns * 7 - len(turn_features)  # 7 = 6(プレイヤーID) + 1(宣言値)
    if padding_size > 0:
        turn_features.extend([0] * padding_size)
    
    # 4. 可能なアクション（legal_action）のマスク
    max_action_value = 140  # 想定される最大宣言値
    action_mask = [0] * (max_action_value + 1)
    
    for action in legal_action:
        if 0 <= action <= max_action_value:
            action_mask[action] = 1
    
    # すべての特徴を連結
    features = player_features + [sum_normalized, round_normalized] + turn_features + action_mask
    
    return np.array(features, dtype=np.float32)

def encode_batch_states(game_data: Dict[str, Any], player_name: str = None) -> np.ndarray:
    """
    ゲームデータから複数の状態を抽出してバッチエンコードする
    
    Args:
        game_data: ゲームデータを含む辞書
        player_name: 視点となるプレイヤー名
        
    Returns:
        np.ndarray: エンコードされた状態のバッチ
    """
    round_info = game_data.get("round_info", [])
    
    # 各ラウンドの状態をエンコード
    encoded_states = []
    for i in range(len(round_info)):
        state = get_current_state_from_rounds(round_info, i, player_name)
        encoded_state = encode_state(state)
        encoded_states.append(encoded_state)
    
    return np.array(encoded_states)

def save_game_states_to_json(game_data: Dict[str, Any], output_file: str) -> None:
    """
    ゲームデータをJSONファイルに保存
    
    Args:
        game_data: 保存するゲームデータ
        output_file: 出力JSONファイルのパス
    """
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(game_data, f, ensure_ascii=False, indent=2)
        print(f"Successfully saved game data to {output_file}")
    except Exception as e:
        print(f"Error saving game data to JSON: {e}")

# 使用例
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        json_file = sys.argv[1]
        print(f"Processing game data from {json_file}")
        
        try:
            # ゲームデータをロード
            game_data = load_game_states_from_json(json_file)
            
            # 最新ラウンドの状態を取得
            current_state = get_current_state_from_rounds(game_data.get("round_info", []))
            
            # 状態をエンコード
            encoded_state = encode_state(current_state)
            print(f"Encoded state shape: {encoded_state.shape}")
            print(f"First 10 values: {encoded_state[:10]}")
            
            # すべてのラウンドをバッチエンコード
            encoded_batch = encode_batch_states(game_data)
            print(f"Encoded batch shape: {encoded_batch.shape}")
            
        except Exception as e:
            print(f"Error processing game data: {e}")
    else:
        print("Usage: python encode_state.py <json_file_path>")
        
        # テスト用のサンプルゲームデータ
        sample_game_data = {
            "round_info": [
                {
                    "round_count": 1,
                    "player_info": [
                        {"name": "PreAI1", "life": 3, "card": 1},
                        {"name": "PreAI2", "life": 3, "card": -10},
                        {"name": "PreAI3", "life": 3, "card": 4},
                        {"name": "PreAI4", "life": 3, "card": 2},
                        {"name": "PreAI5", "life": 3, "card": 102}
                    ],
                    "turn_info": [
                        {"turn_count": 0, "turn_player": "PreAI1", "action": -1}
                    ]
                },
                {
                    "round_count": 2,
                    "player_info": [
                        {"name": "PreAI1", "life": 3, "card": 103},
                        {"name": "PreAI2", "life": 3, "card": -5},
                        {"name": "PreAI3", "life": 3, "card": 4},
                        {"name": "PreAI4", "life": 3, "card": 3},
                        {"name": "PreAI5", "life": 2, "card": 10}
                    ],
                    "turn_info": [
                        {"turn_count": 0, "turn_player": "PreAI1", "action": 1},
                        {"turn_count": 1, "turn_player": "PreAI2", "action": 2},
                        {"turn_count": 2, "turn_player": "PreAI3", "action": 6},
                        {"turn_count": 3, "turn_player": "PreAI4", "action": 7},
                        {"turn_count": 4, "turn_player": "PreAI5", "action": 14},
                        {"turn_count": 5, "turn_player": "PreAI1", "action": -1}
                    ]
                }
            ]
        }
        
        # サンプルデータから状態を抽出
        sample_state = get_current_state_from_rounds(sample_game_data.get("round_info", []))
        
        # サンプル状態をエンコード
        encoded = encode_state(sample_state)
        print(f"Sample state encoded shape: {encoded.shape}")
        
        # サンプルをJSONに保存
        save_game_states_to_json(sample_game_data, "sample_game_data.json")
        print("Saved sample game data to sample_game_data.json")