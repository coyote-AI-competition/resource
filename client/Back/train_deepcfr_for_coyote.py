from .create_advantage_network import create_advantage_network
from .StrategyNetwork import StrategyNetwork
from .reservoirbuffer import ReservoirBuffer
from .encode_state import encode_state
from .calculate_advantages import calculate_advantages
from .update_strategy_network import update_strategy_network
from .update_advantage_network import update_advantage_network
import numpy as np
import logging
def train_deepcfr_for_coyote(self,iterations=10,current_state=None):
    """
    Train Deep CFR for Coyote game

    Args:
        iterations: Number of training iterations
        num_players: Number of players in the game

    Returns:
        list: Trained strategy networks for each player
    """
    num_players = 6

    # Create reservoir buffers for each player
    advantage_buffer = ReservoirBuffer()
    strategy_buffer = ReservoirBuffer()
    try:

      # 状態をエンコード
      encoded_state = encode_state(current_state)
      logging.info(f"Encoded state shape: {encoded_state.shape}")
      logging.info(f"First 10 values: {encoded_state[:10]}")

    except Exception as e:
        print(f"Error processing game data: {e}")

    for i in range(iterations):
        print(f"Iteration {i+1}/{iterations}")

        game_state = [current_state] 
        advantages = calculate_advantages(self, game_state, self.advantage_net)

        update_advantage_network(
            self,
            self.advantage_net,
            advantages,
            advantage_buffer
        )


        # Periodically update strategy networks
        if i % 10 == 0:

            update_strategy_network(
                self,
                self.strategy_net,
                self.advantage_net,
                strategy_buffer,
                advantages
            )

    return self.strategy_net

