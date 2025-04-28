import tensorflow as tf

class StrategyNetwork:
    def __init__(self, input_size, output_size=141):  # 0〜140までの141通りの宣言値
        """
        deepCFRの戦略ネットワークを初期化
        
        Args:
            input_size: 入力特徴の次元数
            output_size: 出力（行動）の次元数
        """
        self.input_size = input_size
        self.output_size = output_size
        self.model = self._build_model()
        
    def _build_model(self):
        """ニューラルネットワークモデルを構築"""
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(self.input_size,)),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(self.output_size)  # 出力層はロジット（未正規化の確率）
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
            state: エンコードされたゲーム状態（np.array）
            legal_actions: 可能な行動のリスト（指定がない場合はすべての行動から選択）
        
        Returns:
            dict: 行動をキー、選択確率を値とする辞書
        """
        # バッチ次元を追加
        state_batch = np.expand_dims(state, axis=0)#stateを配列の中に内包する[0]→[[0]]
                                                   #単一の状態でバッチとして扱うために、次元を追加
        
        # ニューラルネットワークで予測
        logits = self.model(state_batch, training=False).numpy()[0] # 結果を取得
        
        # 可能な行動のみに制限
        if legal_actions is not None:
            mask = np.ones_like(logits) * -1e9 #配列と同じ形状の配列を作成し、そのすべての要素を -1e9に設定
            for action in legal_actions:
                if 0 <= action < len(mask):
                    mask[action] = 0
            logits = logits + mask # 許可されていない行動のスコアは -1e9 に近い極端に小さい値に設定
        
        # ソフトマックスで確率分布に変換
        # 選択肢の確率が同じになる？
        probabilities = np.exp(logits) / np.sum(np.exp(logits))
        
        # 結果を辞書形式で返す
        action_probs = {}
        for i in range(len(probabilities)):
            if legal_actions is None or i in legal_actions:
                if probabilities[i] > 0:
                    action_probs[i] = float(probabilities[i])
        
        return action_probs
