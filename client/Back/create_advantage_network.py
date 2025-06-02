import tensorflow as tf
def create_advantage_network(self):
    """アドバンテージネットワークを作成"""
    # 入力サイズはゲーム状態に合わせて調整
    input_size = self.input_size  # 適切なサイズに調整
    output_size = 141  # -1〜139の宣言値
    
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_size,)),
        tf.keras.layers.Dense(256),  #層
        tf.keras.layers.BatchNormalization(),  # バッチ正規化を追加
        tf.keras.layers.Activation('relu'),# 活性化関数はReLUを使用
        tf.keras.layers.Dropout(0.2),# ドロップアウトを追加
        tf.keras.layers.Dense(256),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(output_size)
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), # 学習率を0.001に設定
        loss='mse', # 損失関数は平均二乗誤差を使用
        metrics=['mae'] # 評価指標として平均絶対誤差を使用
    )
    
    return model