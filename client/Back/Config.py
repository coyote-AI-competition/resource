class Config():
    discount: float = 0.99  # 割引率
    lr: float = 0.00025     # 学習率

    window_length: int = 4  # 状態とするフレーム数
    batch_size: int = 32    # バッチサイズ
    target_model_update_interval: int = 10000  # Target Q-network の同期間隔
    enable_reward_clip: bool = True  # 報酬のclipping

    # Annealing e-greedy
    initial_epsilon: float = 1.0       # 初期ε
    final_epsilon: float = 0.1         # 最終ε
    exploration_steps: int = 1_000_000 # 最終εになるステップ数
    test_epsilon: float = 0  # テスト時のε

    # Experience Replay
    capacity: int = 1_000_000
    memory_warmup_size: int = 50_000  # 学習を始めるまでに貯める経験数