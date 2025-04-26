def sample_game_states():
    """
    ゲームシミュレーションを行い、学習用の状態データを収集する
    
    Returns:
        list: 収集したゲーム状態のリスト（各状態には実際の行動と結果も含む）
    """
    collected_states = []
    
    # 複数回のゲームシミュレーションを実行
    num_simulations = 50  # シミュレーション回数
    
    for _ in range(num_simulations):
        # 新しいゲームインスタンスを作成（自分+5つのCPU）
        game = CoyoteGame(num_players=6)
        game_finished = False
        
        # ゲームが終了するまでシミュレーション
        while not game_finished:
            # 現在のプレイヤー
            current_player = game.get_current_player()
            
            # 現在の状態を取得
            current_state = game.get_state_for_player(current_player.id)
            
            # 可能な行動のリスト
            legal_actions = current_state["legal_action"]
            
            # 探索のためランダムに行動を選択（または既存の戦略を使用）
            if random.random() < 0.8:  # 80%の確率でランダム探索
                action = random.choice(legal_actions)
            else:
                # 現在の戦略を使用（学習が進んだ後）
                action = make_decision(current_state, strategy_net)
            
            # 行動を実行し、結果を取得
            next_state, reward, game_finished = game.step(current_player.id, action)
            
            # CFR用の情報を記録
            trajectory_info = {
                "state": current_state,
                "action": action,
                "reward": reward,
                "next_state": next_state,
                "player_id": current_player.id,
                "is_terminal": game_finished
            }
            
            collected_states.append(trajectory_info)
    
    return collected_states