import numpy as np
def update_advantage_network(self,advantage_net, advantage, buffer, batch_size=32, epochs=10):
    """
    Update the advantage network for a player
    
    Args:
        advantage_net: Player's advantage network
        advantages: Dictionary of advantages for different info sets
        player: Player index
        buffer: Reservoir buffer for storing training data
        batch_size: Training batch size
        epochs: Number of training epochs
    """
    for info_set, advantage_data in advantage.items():
        for action, advantage_value, encoded_state in advantage_data:
          
            buffer.add((encoded_state, action, advantage_value))


    if len(buffer.buffer) < batch_size:
        return  
    
    #len(samples)はターン数に相当する
    samples = buffer.sample(batch_size)
    states = np.array([s[0] for s in samples])
    actions = np.array([s[1] for s in samples])
    advantages = np.array([s[2] for s in samples])
    
    # 状態テンソルの形状を(None, 318)に調整
    if len(states.shape) == 3:
        states = states.reshape(-1, self.input_size)  # (32, 1, 318) → (32, 318)
    
    #len(samples)×141の行列を作成し0に初期化
    targets = np.zeros((len(samples), advantage_net.output_shape[1]))
    for i, (action, advantage) in enumerate(zip(actions, advantages)):
        targets[i, action] = advantage
    
    # Train the advantage network
    advantage_net.fit(states, targets, epochs=epochs, verbose=0)