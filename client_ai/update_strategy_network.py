import numpy as np
import logging
#確率分布を学習する
def update_strategy_network(self,strategy_net, advantage_net, reservoir_buffer, advantages, batch_size=32, epochs=10):
    """
    Update strategy network based on advantage network
    
    Args:
        strategy_net: Strategy network to update
        advantage_net: Advantage network with learned advantages
        reservoir_buffer: Buffer of game states
        batch_size: Training batch size
        epochs: Number of training epoch
    """
    # 各状態についてpolicyを計算し、strategy_bufferに追加
    for info_set, advantage_data in advantages.items():
        for encoded_state, advantage_vector in advantage_data:
            # regret matching
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
            reservoir_buffer.add((encoded_state, policy))

    if len(reservoir_buffer.buffer) < batch_size:
        return  # Not enough data
    
    # Sample states from buffer
    samples = reservoir_buffer.sample(batch_size)
    states = np.array([s[0] for s in samples])
    policy_targets = np.array([s[1] for s in samples])
    
    # 状態テンソルの形状を(None, 318)に調整
    if len(states.shape) == 3:
        states = states.reshape(-1, self.input_size)  # (32, 1, 318) → (32, 318)
    
    # Get advantage values
    # advantage_values = advantage_net.predict(states)
    # print("Advantage values:", advantage_values)
    # print("Max advantage index:", np.argmax(advantage_values), "Value:", np.max(advantage_values))
    # print("Top 5 advantage indices:", np.argsort(advantage_values)[-5:])
    
    # Convert advantages to strategy (policy)
    # Using a softmax over advantages to get policy
    # In real CFR, you would compute normalized positive advantages
    # policy_targets = np.zeros_like(advantage_values)
    
    # for i in range(len(states)):
    #     # Get legal actions for this state (simplified - assuming all actions are legal)
    #     # In real implementation, extract legal actions from state
    #     legal_actions = list(range(advantage_values.shape[1]))
        
    #     # Apply regret matching to convert advantages to policy
    #     positive_advantages = np.maximum(advantage_values[i], 0)
    #     sum_positive = np.sum(positive_advantages)
        
    #     if sum_positive > 0:
    #         policy_targets[i] = positive_advantages / sum_positive
    #     else:
    #         # Uniform strategy over legal actions if no positive regret
    #         for action in legal_actions:
    #             policy_targets[i, action] = 1.0 / len(legal_actions)
    
    # Train strategy network to predict this policy
    strategy_net.model.fit(
        states, 
        policy_targets,
        epochs=epochs,
        verbose=0
    )