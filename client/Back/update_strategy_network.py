import numpy as np
import logging
#確率分布を学習する
def update_strategy_network(self,strategy_net, advantage_net, strategy_buffer, advantages, batch_size=32, epochs=10):
    """
    ストラテジーネットワークの更新
    
    Args:
        strategy_net: ストラテジーネットワーク
        advantage_net: アドバンテージネットワーク
        reservoir_buffer: ストラテジーネットワークの経験
        batch_size: バッチサイズ
        epochs: エポック数
    """
    # 各状態についてpolicyを計算し、strategy_bufferに追加
    for info_set, advantage_data in advantages.items():
        for encoded_state, advantage_vector in advantage_data:
     
            #全体をシフトして負の評価を考慮する
            epsilon = 1e-6
            min_adv = np.min(advantage_vector)
            shifted_advantages = advantage_vector - min_adv + epsilon
            sum_positive = np.sum(shifted_advantages)
            if sum_positive > 0:
                policy = shifted_advantages / sum_positive
            else:
                policy = np.ones_like(advantage_vector) / len(advantage_vector)
            logging.info(f"policy: {policy}")
            logging.info(f"sun_positive: {sum_positive}")
            logging.info(f"shifted_advantages: {shifted_advantages}")
            strategy_buffer.add((encoded_state, policy))

    if len(strategy_buffer.buffer) < batch_size:
        return  # Not enough data
    
    # Sample states from buffer
    samples = strategy_buffer.sample(batch_size)
    states = np.array([s[0] for s in samples])
    self.policy_targets = np.array([s[1] for s in samples])
    
    # 状態テンソルの形状を(None, 318)に調整
    if len(states.shape) == 3:
        states = states.reshape(-1, self.input_size)  # (32, 1, 318) → (32, 318)
    
    # Train strategy network to predict this policy
    strategy_net.model.fit(
        states, 
        self.policy_targets,
        epochs=epochs,
        verbose=0
    )