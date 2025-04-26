import numpy as np
def update_advantage_network(advantage_net, advantage, buffer, batch_size=32, epochs=10):
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
    # Extract training data for this player
    for info_set, advantage_data in advantage.items():
        for action, advantage_value, encoded_state in advantage_data:
            # Add to reservoir buffer
            buffer.add((encoded_state, action, advantage_value))

    # Sample from buffer and prepare training data
    if len(buffer.buffer) < batch_size:
        return  # Not enough data for training
    
    samples = buffer.sample(batch_size)
    states = np.array([s[0] for s in samples])
    actions = np.array([s[1] for s in samples])
    advantages = np.array([s[2] for s in samples])
    
    # Create target values (one-hot with advantages)
    targets = np.zeros((len(samples), advantage_net.output_shape[1]))
    for i, (action, advantage) in enumerate(zip(actions, advantages)):
        targets[i, action] = advantage
    
    # Train the advantage network
    advantage_net.fit(states, targets, epochs=epochs, verbose=0)