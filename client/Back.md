# コヨーテAIを作ってみた話 〜不完全情報ゲームにDeepCFR風DQNを適用してみる〜
こんにちは、今回はカードゲーム「コヨーテ（Coyote）」のAIを作った話をZenn記事風にまとめてみます。不完全情報ゲームの中でも特殊なこのゲームに対し、ルールベースAIとDQN風戦略ネットワークを組み合わせて構築しました。

## コヨーテってどんなゲーム？
コヨーテは「自分のカードだけ見えない」というユニークなルールのゲームです。以下のような特徴があります。

・全員がカードを1枚ずつ持っており、自分のカードは見えない

・順番に場の合計値を推測して「30！」などと宣言する

・宣言が回ってきた時に「それは超えてる！」と思ったら「コヨーテ！」でストップ

・実際の合計値を見て超えていれば宣言者がライフを失い、そうでなければコヨーテをかけた側がライフを失う

・特殊カードが存在し、さらに推測を難しくしている

## ゲームルールと特殊カード
AI実装の前提として、コヨーテにおける特殊カードの意味を簡単に整理しておきます。

| カード     | 枚数 | 効果            |
| ------- | -- | ------------- |
| -10     | 1  | 負の影響を与える      |
| -5      | 2  | 同上            |
| 0       | 3  | 合計に影響なし       |
| 1〜5     | 4  | 基本的な得点カード     |
| 10      | 3  | 中程度の得点        |
| 15      | 2  | 高得点カード        |
| 20      | 1  | 最大級のカード       |
| `0(黒)`  | 1  | ラウンド後に山札をリセット |
| `max→0` | 1  | 最大値を0として扱う    |
| `?`     | 1  | 終了時に山札から追加ドローして場の合計に加算する |
| `×2`    | 1  | 合計を2倍にする      |

## 状態入力：AIが受け取る情報
AIは以下のような状態情報を辞書形式で受け取ります。
```python
state = {
    "others_info": others_info,  # 他プレイヤーのカードとライフ
    "legal_action": actions,     # 自分が宣言できる値（例：26〜34）
    "log": log_info,             # 宣言履歴（誰が何を言ったか）
    "sum": sum,                  # 他人のカード合計（学習時のみ）
    "round_num": round_num,      # 現在のラウンド数
    "player_card": player_card,  # 自分のカード（学習時のみ）
    "Is_coyoted": Is_coyoted     # 前ラウンドでコヨーテされたか
}
```
本番では自分のカードと場の合計（player_card, sum）は見えませんが、**学習時はゲームシステムを改造**してAIにこれらの値を渡しています。

## ルールベースAIの戦略
まず、AIの基礎行動はルールベースAIによって補助されます。これにより、初期段階の意思決定が安定します。

主な戦略：
1 山札の追跡

  ・ゲーム内で使用済みカードを記録し、山札から削除。

  ・山札が切れたらシャッフルして再構成。

2 自分のカードの推定

　・他プレイヤーのカード情報、発言内容、ライフ状況から逆推定。

3 期待値の計算

　・6人分のカード期待値を確率的に評価し、自分のカード候補を調整。

4 場の合計を計算

　・予測した山札と自部の手札とほかのプレイヤーの手札から場の合計を推測する。

## 推論フェーズの流れ
AIが実際にどのように行動を決定するのか、その流れを追ってみましょう。

1. 状態のエンコード（encode_state()）
　・全状態はOne-hotベクトルでエンコード。

　・others_info は最大6人分でパディングされ、固定長（例：317）に変換。

2. 戦略ネットワーク（StrategyNet）による推論
```python
   model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(317,)),
    tf.keras.layers.Dense(256),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(256),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(141)  # 宣言値に対応するロジット
  ])
```
　・非合法な宣言値（合計以下の数など）は -9 をかけて無視。

　・出力ロジットに softmax を適用し、確率を計算。

　・全体の確率に 0.7 を掛けて安定性を向上。

3. 宣言値のサンプリング（sample_from_distribution()）
　・確率分布に基づいて行動（宣言値）を1つ選択して返す。

## 学習アーキテクチャ（DeepCFR風）
このAIは強化学習の一種である**DQNとDeepCFR（Counterfactual Regret Minimization）**の発想を取り入れています。
DQN：Q関数の近似で意思決定
DQNは、**「今この状態でこの行動を取ったら、将来どれくらい得するか**を予測するQ関数をニューラルネットで近似する強化学習の基本手法。

学習の流れ
ゲーム内での状態（プレイヤー情報、ログ、カード構成など）をベクトル化

状態を入力として、各合法な宣言値のQ値を出力

実際の行動結果（勝敗）から報酬を与え、予測誤差をもとにQ関数を更新

ε-greedy法でランダム性と最適行動をバランスよく選択
DeepCFR：不完全情報ゲーム（ポーカーやコヨーテなど）で最適戦略を見つけるために、深層学習を用いて反実的後悔最小化（CFR）を実行するアルゴリズム
今回はおもにDQNの手法を用い、DeepCFRについては下記の発想を採用した。
| ネットワーク名           | 役割                           |
| ----------------- | ---------------------------- |
| **アドバンテージネットワーク** | 各状態における行動ごとの「後悔値（Regret）」を推定 |
| **戦略ネットワーク**      | 後悔の少なさに基づき、確率的な宣言戦略を生成       |


使用した関数の主な役割：
calculate_advantages：
　・プレイヤーが取った行動とその結果から**後悔値（advantage）**を計算。

　・とくに悪い行動には 強い負の報酬 を与える設計。

update_advantage_network：
　・プレイログを使ってアドバンテージネットワークを更新。

　・プレイヤーの失敗パターンを学習。

update_strategy_network：
　・アドバンテージの結果を活用して戦略ネットワークを更新。

　・より勝率の高い宣言値を選ぶようになる。

## 苦労した点とその解決

1. 推論時に発生する型エラー

問題：strategy_netの予測時に TypeError が発生。Kerasモデルが辞書を受け取れない。

原因：モデルのロード処理で、Kerasモデルをstrategy_netのクラスに直接代入してしまっていた。そのため、predict関数を呼ぶと意図せずKeras内部のpredict()が使われてしまい、引数不一致でクラッシュ。

解決法：strategy_netにはpredict()関数を持つラッパークラスを定義し、そのインスタンスにKerasモデルを代入すべきだった。原因特定には地道なprintデバッグが最も有効だった。

2. AIが不自然に大きい値を宣言する

問題：AIがリスクの高い「大きすぎる宣言」を繰り返してライフを失う傾向があった。

解決法：報酬関数の工夫。宣言値と実際の合計との差に応じて段階的なペナルティを設定した。
```python
def define_reward(state_info):
    reward = 0.0
    if state_info["Is_coyoted"] == True:
        reward -= 10
    elif state_info["Is_coyoted"] == False:
        reward += 0.001
    if "sum" in state_info and "selectaction" in state_info:
        game_sum = state_info["sum"]
        declared_value = state_info["selectaction"]
        if declared_value > game_sum:
            if declared_value > game_sum * 1.2:
                reward -= 100
            else:
                reward -= max(0.5, (declared_value - game_sum))*10
        else:
            reward += 1
    return reward
```
AIがリスクを取りすぎる傾向を減らし、実際の合計に近い値を宣言するように調整した。

3. 負の報酬を戦略ネットワークにどう学習させるか

問題：アドバンテージには負の値があるが、戦略ネットワークでは確率として使う必要があるため、そのままでは使えない。

解決法：アドバンテージを最小値で平行移動し、全体を正にシフトして正規化。
```python 
epsilon = 1e-6
min_adv = np.min(advantage_vector)
shifted_advantages = advantage_vector - min_adv + epsilon
sum_positive = np.sum(shifted_advantages)
if sum_positive > 0:
    policy = shifted_advantages / sum_positive
else:
    policy = np.ones_like(advantage_vector) / len(advantage_vector)
```
すべてのアクションが負でも確率として扱えるようにし、安定した学習が可能になった。

学習中のデータ
損失関数が小さくなっているか
```python
# 損失関数を計算（二乗誤差）
loss = (action - actual_sum) ** 2
self.history['loss_values'].append(loss)
```
![](https://storage.googleapis.com/zenn-user-upload/90ab2d496ef3-20250516.png)
場の合計よりギリギリ小さな値を出すほうが自分のターンが回ってくる前にほかのプレイヤー同士でライフを減らす可能性が上がるため損失関数は０に近づくことは好ましいが、場の合計より小さな数字でも問題はない
場の合計をどれだけ超えているか
```python
# 宣言値が合計値を超えた割合を計算
over_ratio = (action - actual_sum) / actual_sum if actual_sum > 0 else 0
self.history['over_declaration_ratio'].append(over_ratio)
```
![](https://storage.googleapis.com/zenn-user-upload/1c51059f32c3-20250516.png)
宣言した数の遷移
![](https://storage.googleapis.com/zenn-user-upload/c2404b8a6c06-20250516.png)
推量中のデータ
勝率
![](https://storage.googleapis.com/zenn-user-upload/5590271e8468-20250516.png)
PreAI1が機会学習によるAI、ほかがルールベースによるAI

## 感想
コヨーテが不完全情報ゲームという性質上正直ルールベースのAIのほうが強いんじゃないかと思っていましたが思いのほか勝てるようになったので良かったです。
まだ３時間ほどしか学習させてないのでもっと学習させるとどうなるかも記事にできたらなと思います。
粗はたくさんあると思うので気軽にご指摘ください。




