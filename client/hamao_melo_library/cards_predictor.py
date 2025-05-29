TOTAL_CARDS = 35  # ?を除いた総数
ALL_CARD = {
    -10: 1,
    -5: 2,
    0: 3,
    1: 4,
    2: 4,
    3: 4,
    4: 4,
    5: 4,
    10: 3,
    15: 2,
    20: 1,
    100: 1,  # ×2カード
    101: 1,  # 黒0カード（シャッフル）max→0カード
    102: 1,  # max→0カード
    103: 1,  # ?カード（コヨーテ対応）
}


class CardsPredictor:
    """
    103(?カード)は別カードに置き換わるため別処理を行う
    """

    def __init__(self):
        """
        各カードが出る確率を予測するクラス
        """
        self.log_list = []
        self.pre_lifes = []
        self.unseen_probs = None

    def initialize(self):
        """
        対戦が終わったまたは黒0が出たタイミングで初期化
        判断基準はライフが変わったかどうか
        """
        self.log_list = []

    def _calc_empty_turns(self, lifes_list):
        """
        空白のターンが存在する場合
        1. lifeesの減りがどれくらいか
        2. 空白のターンを404で埋める（人数が減っている場合は、ログのそれぞれのカード枚数が多くなるようにする）
        """
        pre_life_sum = sum(self.pre_lifes)
        now_life_sum = sum(lifes_list)
        empty_turns = pre_life_sum - now_life_sum - 1

        pre_player_num = len(self.pre_lifes)
        now_player_num = len(lifes_list)
        difference_player_num = pre_player_num - now_player_num

        if difference_player_num > empty_turns:
            for i in range(empty_turns):
                self.log_list.append([404] * (pre_player_num - 1 - i))
        else:
            for _ in range(empty_turns - difference_player_num):
                self.log_list.append([404] * (pre_player_num))
            for i in range(difference_player_num):
                self.log_list.append([404] * (pre_player_num - i))

    def get_others_info(self, others_info):
        """
        1. 他プレイヤーのライフ情報取得
        2. ライフに変化があればログを更新
            2(a). 新しい対戦に切り替わったかの確認
            2(b). 各カードが出る確率を更新する

        """
        lifes_list = sorted(info["life"] for info in others_info)
        if self.pre_lifes != lifes_list:
            # 新しい対戦開始
            if sum(lifes_list) > sum(self.pre_lifes):
                self.initialize()
            # 空白のターンが存在する場合
            elif sum(lifes_list) + 1 < sum(self.pre_lifes):
                self._calc_empty_turns(lifes_list)
            now_cards = [info["card_info"] for info in others_info]
            self.log_list.append(now_cards)
            self.update_probs()
            self.pre_lifes = lifes_list

    def _remove_invalid_logs(self):
        """
        黒0, 総枚数超過, カード別枚数超過のログを削除(空白ターンを意味する404に注意!!)
        1. 総枚数が超過するか（最新ログは1枚だとして、途中で35枚を超えたログがあったら終わり）
        2. 最新のログを除いた状態で黒0とカード別枚数超過を調べる
        (a) 黒0出現, (b) 総枚数超過, (c) カード別枚数超過
        """
        counts = {card: 0 for card in ALL_CARD}
        total_seen = 0
        remove_idx = None

        # 最新から古い順に走査
        for idx in range(len(self.log_list) - 1, -1, -1):
            entry = self.log_list[idx]
            total_seen += len(entry) + 1
            for c in entry:
                if c != 404:
                    counts[c] = counts.get(c, 0) + 1

            # (a) 黒0出現, (b) 総枚数超過, (c) カード別枚数超過
            if (
                ((101 in entry) and idx + 1 < len(self.log_list))
                or (total_seen > TOTAL_CARDS)
                or any((c != 404 and counts[c] > ALL_CARD.get(c, 0)) for c in entry)
            ):
                remove_idx = idx
                break

        # 条件に当てはまったログとそれ以前を削除
        if remove_idx is not None:
            remove_idx = min(len(self.log_list) - 2, remove_idx)
            self.log_list = self.log_list[remove_idx + 1 :]

    def update_probs(self):
        """
        各カードが出る確率を計算する
        """
        # 不要なログを削除
        self._remove_invalid_logs()

        # すべてのログに対して、黒0が出たかどうかでbit全探索をする
        self.unseen_probs = {
            card: (0, 0.0) for card in ALL_CARD
        }  # card : (最大枚数, 確率)
        n = len(self.log_list) - 1
        for bit in range(1 << n):
            prob = 1.0
            now_cards_num = TOTAL_CARDS
            # 確率は古いやつから計算する（listで先頭の方が古いことに注意する）
            for idx in range(n):
                if 103 in self.log_list[idx]:
                    # ?カードが出た場合
                    now_cards_num -= len(self.log_list[idx]) - 1

                    if (bit >> idx) & 1:
                        # 黒0が出たとする場合
                        prob *= 2.0 / now_cards_num
                    else:
                        # 黒0が出なかったとする場合
                        prob *= (now_cards_num - 2) / now_cards_num
                    now_cards_num -= 2
                elif 404 in self.log_list[idx]:
                    # 空白のターンについて
                    if (bit >> idx) & 1:
                        # 黒0が出たとする場合
                        prob *= len(self.log_list[idx]) / now_cards_num
                    else:
                        amari_card = now_cards_num - len(self.log_list[idx])
                        prob *= amari_card / now_cards_num
                    now_cards_num -= len(self.log_list[idx]) + 1
                else:
                    # ?カードが出なかった場合
                    now_cards_num -= len(self.log_list[idx])

                    if (bit >> idx) & 1:
                        # 黒0が出たとする場合
                        prob *= 1.0 / now_cards_num
                    else:
                        prob *= (now_cards_num - 1) / now_cards_num
                    now_cards_num -= 1

            # まだ出ていないカードは最新のログから計算する
            each_card_num = {card: 0 for card in ALL_CARD}
            for c in self.log_list[-1]:
                each_card_num[c] = each_card_num.get(c, 0) + 1

            for idx in range(n - 1, -1, -1):
                # 黒0が出たら終わり
                if (bit >> idx) & 1:
                    break
                entry = self.log_list[idx]
                for c in entry:
                    if c != 404:
                        each_card_num[c] = each_card_num.get(c, 0) + 1

            # 残っているカードの枚数を記憶
            unseen_cards_sum = TOTAL_CARDS - sum(each_card_num.values())

            # それぞれのカードに対する確率を計算
            for card, max_cnt in ALL_CARD.items():
                seen = each_card_num.get(card, 0)
                remain = max_cnt - seen
                if remain > 0 and card != 103:
                    pre_cnt, pre_prob = self.unseen_probs[card]
                    self.unseen_probs[card] = (
                        max(pre_cnt, remain),
                        pre_prob + (prob * remain / max_cnt) / unseen_cards_sum,
                    )

    def get_unseen_probs(self):
        """
        まだ出ていないカードを返す
        """
        return self.unseen_probs
