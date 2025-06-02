# チーム：はまおにメロメロ（はまお・めろチーム）
## 動かし方
arena.pyで以下のようにしたら動きます
```
    from client.hamao_ni_melomelo import HamaoNiMeloMelo as HamaoNiMeloMelo
    
    predefs = [
        [HamaoNiMeloMelo(player_name="HamaoNiMeloMelo", is_ai=True), "HamaoNiMeloMelo"],
    ]
```

arena.pyの例
```
from server.arena import Arena
if __name__ == "__main__":
    from client.sample_arena_client import SampleClient as SampleClient
    from client.hamao_ni_melomelo import HamaoNiMeloMelo as HamaoNiMeloMelo
    
    predefs = [
        [HamaoNiMeloMelo(player_name="HamaoNiMeloMelo", is_ai=True), "HamaoNiMeloMelo"],
        [SampleClient(player_name="PreAI1", is_ai=True), "PreAI1"],
        [SampleClient(player_name="PreAI2", is_ai=True), "PreAI2"],
        [SampleClient(player_name="PreAI3", is_ai=True), "PreAI3"],
        [SampleClient(player_name="PreAI3", is_ai=True), "PreAI4"],
        [SampleClient(player_name="PreAI3", is_ai=True), "PreAI5"]
    ]
    
    arena = Arena(total_matches=5, predefined_clients=predefs)
    arena.run()
    #
    
    # 事前クライアントを指定せずに起動
    
    # arena = Arena()
    # arena.run()
```
## 追加で必要なライブラリなど
ありません

# アイデア概要

本大会で実装したのは、ルールベースのAIです。以下の3つの観点に注力しました。

1. **宣言値の決定**
2. **未使用カードの推定**
3. **コヨーテ宣言の基準**

各項目について詳しく説明します。

---

## 1. 宣言値の決定

* **目標**：次のプレイヤーの成功確率を変えずに、自分以外のプレイヤーがコヨーテを仕掛ける確率を高める。
* **方針**：現在の宣言値+1ではなく、場であり得る値のうち「宣言済み値より大きい最小値」を選択。
* **注意点**：自分視点の計算だけでは不十分。次のプレイヤーの視点で成功確率が変わらない組み合わせを、全探索して把握する必要があります。

---

## 2. 未使用カードの推定

* **前提**：山札は「全て尽きる」か「黒0カードが出る」までリセットされず、それ以外は利用済みカードは再度出ません。
* **課題**：前ターンの自分のカードが見えないため、単純なカウント手法が使えない。
* **解決策**：

  1. カード出現ログに対して「黒0が出た」「出ていない」の二択をビット全探索でシミュレーション
  2. 各シナリオで残りのカードとその確率を計算
  3. 正規化して、場であり得る合計値ごとの成功確率を算出

---

## 3. コヨーテ宣言の基準

* **理想モデル**：成功率100%のときのみ宣言するAIは安定するが、実践では控えめな方が強い。
* **検証結果**：成功率50%〜100%までのモデルを対戦させたところ、成功率80%のAIが最も勝率が高かった。
* **結論**：本AIでは、自身のコヨーテ成功確率が**80%以上**のときにのみ宣言する。

---

# 工夫したポイント

## 空白ターンの取り扱い（404埋め）

* 自分に一度もターンが回らずライフ変化だけ起きた場合、カードログが失われる。
* 総山札枚数の整合性を保つため、空白ターンに相当するダミーカードを「404」としてログへ追加。
* ライフ情報の変化から空白ターン数とプレイヤー数を計算し、404を埋める枚数を動的に調整。

## ランダムAIへの対策

* ランダムAIは宣言値が大きくばらつきやすい特性を利用。
* 「隣接プレイヤーの宣言差が10以上」が全宣言回数の40%以上に達したらランダムAIと判定。
* ランダムAIが次ターンなら、成功確率を低めに扱い、自滅を待つ戦略を優先。

---

# まとめ

大会終盤や終了後に多くの質問をお送りしましたが、運営の皆様のご対応に感謝いたします。
コヨーテ戦略の考察は非常に刺激的で、学びの多い大会でした。改めて御礼申し上げます。
