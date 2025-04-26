from .create_advantage_network import create_advantage_network
from .StrategyNetwork import StrategyNetwork
from .reservoirbuffer import ReservoirBuffer
from .encode_state import encode_state
from .calculate_advantages import calculate_advantages
from .update_strategy_network import update_strategy_network
from .update_advantage_network import update_advantage_network
def train_deepcfr_for_coyote(iterations=1000,current_state=None):
    """
    Train Deep CFR for Coyote game

    Args:
        iterations: Number of training iterations
        num_players: Number of players in the game

    Returns:
        list: Trained strategy networks for each player
    """
    num_players = 6
    # Create networks for each player
    advantage_net = create_advantage_network
    strategy_net = StrategyNetwork(303, 141)

    # Create reservoir buffers for each player
    advantage_buffer = ReservoirBuffer()
    strategy_buffer = ReservoirBuffer()
    try:
      # ゲームデータをロード
      #   json_file = json_file_path
      #   game_data = load_game_states_from_json(json_file)

      # 最新ラウンドの状態を取得
      # game state
      #current_state = get_current_state_from_rounds(game_data.get("round_info", []))

      # 状態をエンコード
      encoded_state = encode_state(current_state)
      print(f"Encoded state shape: {encoded_state.shape}")
      print(f"First 10 values: {encoded_state[:10]}")

      # すべてのラウンドをバッチエンコード
    #   encoded_batch = encode_batch_states(game_data)
    #   print(f"Encoded batch shape: {encoded_batch.shape}")

    except Exception as e:
        print(f"Error processing game data: {e}")

    for i in range(iterations):
        print(f"Iteration {i+1}/{iterations}")

        # Simulate games using current strategy
        game_states = [current_state] #simulate_coyote_game(strategy_nets, num_players)

        # Calculate advantages 
        advantages = calculate_advantages(game_states, advantage_net)

        update_advantage_network(
            advantage_net,
            advantages,
            advantage_buffer
        )

        # Periodically update strategy networks
        if i % 10 == 0:
            strategy_net.model.save(f"/content/drive/MyDrive/models/model_{player}.keras")

            update_strategy_network(
                strategy_net,
                advantage_net,
                advantage_buffer
            )

            # Save models
            for player in range(num_players):
                strategy_net.model.save(f"/content/drive/MyDrive/models/model_{player}.keras")

    return strategy_net

