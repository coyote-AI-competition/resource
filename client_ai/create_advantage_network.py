import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

def create_advantage_network():
    """アドバンテージネットワークを作成"""
    # 入力サイズはゲーム状態に合わせて調整
    input_size = 304  # 適切なサイズに調整
    output_size = 141  # 0〜140の宣言値
    
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_size,)), # 長さ1000のベクトルを入力
        tf.keras.layers.Dense(256, activation='relu'), # 重み付きの線形変換を行い、正規化線形関数ReLU(入力が正の値の場合はそのまま出力し、負の値の場合は0を出力)勾配消失に有効
                                                       # 重みは二変数の係数、重回帰分析でバイアスを求めやすくするために１を加える
                                                       # 非線形変換を行う関数を活性化関数という。
        tf.keras.layers.Dropout(0.2),  # 過学習を防ぐために、一定の確率でノードを無効化するドロップアウト層
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(output_size) # 出力層、アドバンテージ値を出力するためのノード数は141
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), # 学習率を0.001に設定
        loss='mse', # 損失関数は平均二乗誤差を使用
        metrics=['mae'] # 評価指標として平均絶対誤差を使用
    )
    
    return model