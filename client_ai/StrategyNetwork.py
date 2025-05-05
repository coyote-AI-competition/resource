import tensorflow as tf
import numpy as np

class StrategyNetwork:
    def __init__(self, total_sum, input_size, output_size=141):  # 入力サイズを318に固定
        """
        deepCFRの戦略ネットワークを初期化
        
        Args:
            input_size: 入力特徴の次元数（デフォルト: 317）
            output_size: 出力（行動）の次元数（デフォルト: 141）
        """
        self.input_size = input_size
        self.output_size = output_size
        self.total_sum = total_sum
        self.model = self._build_model()
        
    def _build_model(self):
        """ニューラルネットワークモデルを構築"""
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(self.input_size,)),
            tf.keras.layers.Dense(256),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(256),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(self.output_size)  # 出力層はロジット
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True), #交差エントロピー損失関数
            metrics=['accuracy'] #正解率
        )
        return model
    
    def predict(self, state, legal_actions=None):
        """
        戦略ネットワークを使用して、各行動の確率分布を予測する
        
        Args:
            state: エンコードされたゲーム状態np.array
            legal_actions: 可能な行動のリスト（指定がない場合はすべての行動から選択）
        
        Returns:
            dict: 行動をキー、選択確率を値とする辞書
        """
        # テンソルに変換
        if isinstance(state, np.ndarray):
            state = tf.convert_to_tensor(state, dtype=tf.float32)
        
        # 形状を(None, input_size)に調整
        if len(state.shape) == 1:
            state = tf.expand_dims(state, axis=0)  # (1, input_size)の形状に
        elif len(state.shape) == 3:
            state = tf.reshape(state, (-1, self.input_size))  # (1, 1, input_size) → (1, input_size)
        
        # 入力サイズの確認と調整
        if state.shape[-1] != self.input_size:
            # 不足している次元を0で埋める
            padding = tf.zeros((state.shape[0], self.input_size - state.shape[-1]))
            state = tf.concat([state, padding], axis=1)
        
        # ニューラルネットワークで予測
        logits = self.model(state, training=False).numpy()[0] # 結果を取得
            # コヨーテの選択確率を調整
        if legal_actions is not None and -1 in legal_actions:
            # 前のプレイヤーの宣言値と場の合計を考慮
            previous_declaration = legal_actions[1] - 1  # 前のプレイヤーの宣言値
            current_sum = self.total_sum  # 場の合計
            
            if previous_declaration > current_sum:
                # 前のプレイヤーの宣言が実際の合計より大きい場合
                # コヨーテの確率を上げる
                logits[0] += 2.0  # コヨーテのインデックスは0
                            # 他の行動の確率を下げる
                for i in range(1, len(logits)):
                    logits[i] -= 0.5
        
        # 可能な行動のみに制限
        #-1から140まで
        if legal_actions is not None:
            mask = np.ones_like(logits) * -1e9
            for action in legal_actions:
                mask[action+1] = 0 # 1を足すのは、-1から始まるインデックスを考慮するため
            logits = logits + mask # 許可されていない行動のスコアは -1e9 に近い極端に小さい値に設定
        
        # ソフトマックスで確率分布に変換
        probabilities = np.exp(logits) / np.sum(np.exp(logits))

        for action in range(len(probabilities)):
            if action > self.total_sum * 0.8:  # 宣言値が既知の合計の80%を超える場合
                probabilities[action] *= 0.7  # 確率を減らす
        
        # 確率の正規化（小さい確率を0に）
        threshold = 1e-5  # 閾値
        probabilities[probabilities < threshold] = 0
        probabilities = probabilities / np.sum(probabilities)  # 再正規化
        
        
        # 結果を辞書形式で返す
        #-1から140までaction
        #0〜141までindex
        action_probs = {}
        for i in range(len(probabilities)):  # インデックスを-1から140の範囲に変換
  
            if probabilities[i] > 0:
                action_probs[i] = float(probabilities[i])
        
        return action_probs
    '''
    例
    action_probs = {
    0: 0.1,  # 行動0の確率
    1: 0.3,  # 行動1の確率
    2: 0.6   # 行動2の確率
    }
    '''
