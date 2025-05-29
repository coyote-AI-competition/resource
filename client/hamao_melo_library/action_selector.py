from itertools import combinations
from bisect import bisect_left
from collections import defaultdict


class ActionSelector:
    """
    自分が何を宣言するかを決めるためのクラス
    """

    def __init__(self):
        pass

    def calc_score(self, cards):
        """
        スコアを計算する関数
        """
        normal = []
        doubled = False
        max_to_zero = False

        for c in cards:
            if c == 100:
                doubled = True
            elif c == 102:
                max_to_zero = True
            elif c == 101:
                normal.append(0)
            else:
                assert c < 100, "Invalid card value"
                normal.append(c)

        # max→0 の効果
        if max_to_zero and normal:
            m = max(normal)
            normal.remove(m)

        total = sum(normal)
        if doubled:
            total *= 2
        return total

    def calc_max_score(self, others_info):
        """あり得る最大値を計算する関数"""
        # 1) 他人のカード一覧
        cards = [info["card_info"] for info in others_info]

        # 2) full deck と unseen を作成
        unseen = [20, 15, 15, 10, 10, 10, 100]
        hatena_flag = False
        for c in cards:
            if c == 103:
                hatena_flag = True
            if 10 <= c <= 20 or c == 100:
                unseen.remove(c)

        if hatena_flag:
            # ?がある場合
            cards.remove(103)
            max_score = -200
            # 大きい方から2つ使う場合
            now_cards = cards.copy() + [unseen[0], unseen[1]]
            now_score = self.calc_score(now_cards)
            if now_score > max_score:
                max_score = now_score
            if unseen[-1] == 100:
                # ×2と最大値
                now_cards = cards.copy() + [unseen[0], unseen[-1]]
                now_score = self.calc_score(now_cards)
                if now_score > max_score:
                    max_score = now_score
            return max_score
        else:
            # ?がない場合
            max_score = -200
            # 一番大きい数を使う場合
            now_cards = cards.copy() + [unseen[0]]
            now_score = self.calc_score(now_cards)
            if now_score > max_score:
                max_score = now_score
            if unseen[-1] == 100:
                now_cards = cards.copy() + [unseen[-1]]
                now_score = self.calc_score(now_cards)
                if now_score > max_score:
                    max_score = now_score
            return max_score

    def calc_next_min(self, others_info, declared_sum):
        """
        自分のカードと次のプレイヤーのカードを全通り試して、あり得る中での最小値
        """
        seen_cards = [info["card_info"] for info in others_info if not info["is_next"]]
        q_flag = 103 in seen_cards
        unknown_count = 2 + (1 if q_flag else 0)
        base_cards = [c for c in seen_cards if c != 103]

        all_cards = (
            [-10] * 1
            + [-5] * 2
            + [0] * 3
            + sum(([i] * 4 for i in range(1, 6)), [])
            + [10] * 3
            + [15] * 2
            + [20] * 1
            + [100, 101, 102]
        )

        for card in base_cards:
            all_cards.remove(card)

        possible_scores = set()
        for draw in combinations(all_cards, unknown_count):
            trial_cards = base_cards + list(draw)
            score = self.calc_score(trial_cards)
            possible_scores.add(score)

        possible_scores = sorted(possible_scores)
        possible_scores.append(possible_scores[-1] + 1)

        # あり得るスコアの中で、宣言値+1以上の中で最小の値を探す
        next_min_score = bisect_left(possible_scores, declared_sum + 1)
        return possible_scores[next_min_score]

    def choice_action(self, others_info, unseen_probs, declared_sum, next_min_score, gap_player):
        """
        others_info: List[{"card_info": int, "is_next": bool}, ...]
        → 自分以外の人が場に出しているカード情報（次のプレイヤーは is_next=True）
        戻り値: あり得るスコアを重複なく昇順ソートした List[int]
        """
        # 自分のカードと?のカードをunseen_probsで、場の合計と起きる確率を計算する
        seen_cards = [info["card_info"] for info in others_info]
        q_flag = 103 in seen_cards
        base_cards = [c for c in seen_cards if c != 103]

        unknown_count = 1 + (1 if q_flag else 0)
        possible_scores = defaultdict(float)

        unseen_cards = []
        for key, value in unseen_probs.items():
            cnt, prob = value
            if prob > 0.0:
                unseen_cards.extend([key] * cnt)

        for draw in combinations(unseen_cards, unknown_count):
            trial_cards = base_cards + list(draw)
            score = self.calc_score(trial_cards)
            probs = 1.0
            for c in draw:
                probs *= unseen_probs[c][1]
            possible_scores[score] += probs

        # possible_scoresのscoreで昇順ソートする
        possible_scores = sorted(possible_scores.items())

        # 正規化をする
        total_prob = sum(prob for _, prob in possible_scores)
        if total_prob > 0:
            possible_scores = [
                (score, prob / total_prob) for score, prob in possible_scores
            ]
        else:
            return -1

        # declared_sum-1までの確率を足し合わせる
        success_probs = 0.0
        for score, prob in possible_scores:
            if score < declared_sum:
                success_probs += prob
            else:
                break

        if gap_player and success_probs >= 0.95:
            return -1
        elif not gap_player and success_probs >= 0.8:
            return -1
        else:
            return next_min_score
