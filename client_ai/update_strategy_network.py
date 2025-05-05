import numpy as np
def update_strategy_network(self,strategy_net, advantage_net, reservoir_buffer, batch_size=32, epochs=10):
    """
    Update strategy network based on advantage network
    
    Args:
        strategy_net: Strategy network to update
        advantage_net: Advantage network with learned advantages
        reservoir_buffer: Buffer of game states
        batch_size: Training batch size
        epochs: Number of training epochs
    """
    if len(reservoir_buffer.buffer) < batch_size:
        return  # Not enough data
    
    # Sample states from buffer
    samples = reservoir_buffer.sample(batch_size)
    states = np.array([s[0] for s in samples])
    
    # 状態テンソルの形状を(None, 318)に調整
    if len(states.shape) == 3:
        states = states.reshape(-1, self.input_size)  # (32, 1, 318) → (32, 318)
    
    # Get advantage values
    advantage_values = advantage_net.predict(states)
    
    # Convert advantages to strategy (policy)
    # Using a softmax over advantages to get policy
    # In real CFR, you would compute normalized positive advantages
    policy_targets = np.zeros_like(advantage_values)
    
    for i in range(len(states)):
        # Get legal actions for this state (simplified - assuming all actions are legal)
        # In real implementation, extract legal actions from state
        legal_actions = list(range(advantage_values.shape[1]))
        
        # Apply regret matching to convert advantages to policy
        positive_advantages = np.maximum(advantage_values[i], 0)
        sum_positive = np.sum(positive_advantages)
        
        if sum_positive > 0:
            policy_targets[i] = positive_advantages / sum_positive
        else:
            # Uniform strategy over legal actions if no positive regret
            for action in legal_actions:
                policy_targets[i, action] = 1.0 / len(legal_actions)
    
    # Train strategy network to predict this policy
    strategy_net.model.fit(
        states, 
        policy_targets,
        epochs=epochs,
        verbose=0
    )