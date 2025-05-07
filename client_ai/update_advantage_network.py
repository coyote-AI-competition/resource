import numpy as np
import logging
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
    #advantageをバッファに詰め込んでる
    for info_set, advantage_data in advantage.items():
        for encoded_state, advantage_vector in advantage_data:
            logging.info("Advantage vector: %s", advantage_vector)
            logging.info("Max advantage index: %s, Value: %s", np.argmax(advantage_vector), np.max(advantage_vector))
            buffer.add((encoded_state, advantage_vector))

    #バッファに詰め込んだデータがbatch_sizeより少ない場合は訓練しない
    if len(buffer.buffer) < batch_size:
        logging.info(f"Not enough data to train. Buffer size: {len(buffer.buffer)}")
        return  
    
    #len(samples)はターン数に相当する
    logging.info(f"十分なbufferがある: {len(buffer.buffer)}")
    samples = buffer.sample(batch_size)
    states = np.array([s[0] for s in samples])
    targets = np.array([s[1] for s in samples])
    #advantages = np.array([s[2] for s in samples])

    
    # 状態テンソルの形状を(None, 318)に調整
    if len(states.shape) == 3:
        # 3次元の状態テンソル(batch_size, 1, input_size)を2次元(batch_size, input_size)に変換
        # 例: (32, 1, 318) → (32, 318)の形状に変換する
        # これは advantage_net が2次元入力を想定しているため必要な変換
        states = states.reshape(-1, self.input_size)  # (32, 1, 318) → (32, 318)
    
    #サンプル数×141の行列を作成し0に初期化
    #宣言値とその価値を保管する
    # info_set_key = f"player_state{hash(str(encoded_state.numpy().tobytes()))}"
    # advantages[info_set_key].append((action, advantage, encoded_state))
    # targets = np.zeros((len(samples), advantage_net.output_shape[1]))
    # for i, (action, advantage) in enumerate(zip(actions, advantages)):
    #     targets[i, action] = advantage
    
    
    # 状態(states)と目標値(targets)を使ってAdvantageネットワークを訓練
    # states: プレイヤーの状態情報 (batch_size, input_size)の形状
    # targets: 各行動の優位性(advantage)値 (batch_size, num_actions)の形状
    # epochs: 訓練エポック数
    # verbose=0: 訓練の進捗を表示しない
    advantage_net.fit(states, targets, epochs=epochs, verbose=0)