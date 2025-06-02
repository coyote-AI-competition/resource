import tensorflow as tf
import numpy as np
import logging

class StrategyNetwork:
    def __init__(self, total_sum, input_size, output_size=141):  # 入力サイズを318に固定
        """
        deepCFRの戦略ネットワークを初期化
        
        Args:
            input_size: 入力特徴の次元数（デフォルト: 317)
            output_size: 出力（行動）の次元数（デフォルト: 141)
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
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),#学習率
            loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True), #交差エントロピー損失関数
            metrics=['accuracy'] #正解率
        )
        return model
    
    def prediction(self, state, legal_actions):
        """
        戦略ネットワークを使用して、各行動の確率分布を予測する
        
        Args:
            state: エンコードされたゲーム状態np.array
            legal_actions: 可能な行動のリスト
        
        Returns:
            dict: 行動をキー、選択確率を値とする辞書
        """
        
        # テンソルに変換
        if isinstance(state, np.ndarray):
            state = tf.convert_to_tensor(state, dtype=tf.float32)
        
        # 形状を(1, input_size)に調整
        if len(state.shape) == 1:
            state = tf.expand_dims(state, axis=0)  # (input_size,) → (1, input_size)
        elif len(state.shape) == 2 and state.shape[0] != 1:
            state = tf.expand_dims(state[0], axis=0)  # (batch_size, input_size) → (1, input_size)
        elif len(state.shape) == 3:
            state = tf.reshape(state[0], (1, self.input_size))  # (batch_size, 1, input_size) → (1, input_size)

        # モデルの予測（直接呼び出し）
        logits = self.model(state)
        logits = logits.numpy()[0]  # バッチ次元を削除

        
        # マスクの作成と適用
        mask = np.ones_like(logits) * -1e9
        for action in legal_actions:
            if action > self.total_sum:
                continue
            if action == -1:  # コヨーテアクションの場合
                mask[0] = 0
            elif 0 <= action + 1 < len(mask):  # 通常アクションの場合
                mask[action + 1] = 0
        logits = logits + mask
        
        # ソフトマックスで確率に変換
        exp_logits = np.exp(logits - np.max(logits))  # 数値安定性のため最大値を引く
        probabilities = exp_logits / np.sum(exp_logits)
        
        # 合法手の確率を調整
        for action in legal_actions:
            if action > self.total_sum:
                if action == -1:
                    probabilities[0] *= 0.7
                elif 0 <= action + 1 < len(probabilities):
                    probabilities[action + 1] *= 0.7
        
        # 確率の正規化（小さい確率を0に）
        threshold = 1e-9
        probabilities[probabilities < threshold] = 0
        if np.sum(probabilities) > 0:
            probabilities = probabilities / np.sum(probabilities)
        else:
            # すべての確率が0の場合、合法手に均等に分配
            for action in legal_actions:
                if action <= self.total_sum:
                    if action == -1:
                        probabilities[0] = 1 / len(legal_actions)
                    elif 0 <= action + 1 < len(probabilities):
                        probabilities[action + 1] = 1 / len(legal_actions)
        
        # 結果を辞書形式で返す
        action_probs = {}
        for i in range(len(probabilities)):
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

logits: [ 1.22457094e-01 -4.30556387e-01 -3.16377319e-02  2.45254785e-01
  1.58598237e-02 -2.09477365e-01 -3.96683693e-01 -4.70872641e-01
 -1.87415212e-01 -1.60400406e-01 -1.97609752e-01  3.33674043e-01
  2.34142780e-01 -8.32803309e-01  1.99890092e-01  2.38919735e-01
 -5.67488894e-02  2.55110979e-01 -2.17046499e-01 -3.99057269e-01
  6.47019625e-01 -1.38353826e-02 -6.82597160e-02  4.16436940e-01
  7.66445696e-02 -9.88784209e-02  7.14716464e-02  4.66795325e-01
  1.65193871e-01  1.21084332e-01  4.94306028e-01 -2.80021846e-01
 -1.69805199e-01  2.51998067e-01 -3.72004926e-01 -1.98936477e-01
 -1.43850967e-01 -5.30333817e-03 -1.03223994e-01 -1.61558956e-01
 -2.69638300e-01  3.88702571e-01 -3.90698850e-01 -3.73131543e-01
  1.92969278e-01  5.24535477e-01 -5.49451113e-01 -2.75856525e-01
  6.06835485e-01  1.43728152e-01 -1.43570349e-01 -3.32691893e-02
 -7.65157461e-01 -8.72291103e-02  5.25319874e-02 -2.74462134e-01
 -3.94941360e-01 -3.28433573e-01  2.57894456e-01 -2.22618133e-02
 -4.04618889e-01 -3.93895656e-02 -3.71805243e-02 -1.48253813e-01
  3.83762091e-01 -1.20932534e-01 -1.87520459e-01  1.94739223e-01
 -2.03549713e-01  1.14069857e-01  2.53289878e-01  4.27465349e-01
  2.98282474e-01  4.22549486e-01  3.17404538e-01 -2.24723235e-01
 -3.18817735e-01 -5.61019257e-02  3.82125705e-01 -3.47228982e-02
 -5.18853664e-01 -4.65586007e-01 -3.61785144e-01 -5.40898740e-02
  2.53823161e-01 -4.82387215e-01 -1.56316206e-01 -5.18646650e-02
 -2.65948236e-01 -1.64450616e-01 -1.39067277e-01  1.94532778e-02
  2.47659013e-01 -9.29117426e-02  1.69479057e-01  1.44720048e-01
 -3.18615913e-01 -3.09020758e-01 -3.08915675e-01 -4.73797053e-01
 -1.38401449e-01  5.25766611e-01  2.56872475e-01  1.65006258e-02
  3.59890133e-01  4.88821596e-01 -4.66996849e-01  1.22111499e-01
  4.50904548e-01  2.18829781e-01 -1.35147005e-01  4.00356613e-02
  5.16049922e-01 -4.14972752e-02 -6.01311564e-01  5.19873083e-01
 -7.47524574e-02 -3.53336692e-01  4.05257195e-01 -2.26636425e-01
 -2.83076227e-01  5.28706703e-04 -1.08752161e-01  1.50572628e-01
  1.30512506e-01  5.25951743e-01  1.53017595e-01  3.57872903e-01
 -1.50436938e-01  1.19548663e-01  2.36338265e-02 -1.43781498e-01
  2.40084738e-01  2.08492026e-01  1.15423230e-02 -3.80248874e-01
  3.49057317e-01  4.90611702e-01  2.29714796e-01  2.36446962e-01
  1.57950968e-01]
  '''
