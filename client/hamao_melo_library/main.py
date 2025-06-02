from .action_selector import ActionSelector
from .cards_predictor import CardsPredictor


class MyLibrary:
    def __init__(self):
        self.action_selector = ActionSelector()
        self.cards_predictor = CardsPredictor()

        # playerと宣言する数との開きを計算する
        self.player_declaration_gap = {}
        self.my_name = None

    def _gap_count(self, others_info, log):
        n = len(log) - 1
        for i in range(n, 0, -1):
            player_name = log[i].get("turn_player", None)
            if player_name is not None:
                # 自分の名前かの確認をする
                if self.my_name is None:
                    others_name_list = [info["name"] for info in others_info]
                    if player_name not in others_name_list:
                        self.my_name = player_name
                        break

                # 自分の名前ならpassする
                if player_name == self.my_name:
                    break

                # 他のプレイヤーの名前なら、宣言数と1個前の宣言数との差を計算する
                pre_declared = log[i - 1].get("action", None)
                now_declared = log[i].get("action", None)
                if pre_declared is not None and now_declared is not None:
                    gap = now_declared - pre_declared
                    if player_name in self.player_declaration_gap:
                        gap_cnt, total_cnt = self.player_declaration_gap[player_name]
                        if gap >= 10:
                            gap_cnt += 1
                        self.player_declaration_gap[player_name] = (
                            gap_cnt,
                            total_cnt + 1,
                        )
                    else:
                        gap_cnt = 1 if gap >= 10 else 0
                        self.player_declaration_gap[player_name] = (gap_cnt, 1)


    def action(self, others_info, log, actions):
        self._gap_count(others_info, log)
        self.cards_predictor.get_others_info(others_info)
        unseen_probs = self.cards_predictor.get_unseen_probs()

        """
        例外処理
        """
        # コヨーテしかできない場合
        if len(actions) == 1:
            return -1
        # 最初の番であり、コヨーテが出来ない場合は0以上であり得る数の最小
        if actions[0] != -1:
            next_min = self.action_selector.calc_next_min(others_info, 0)
            return next_min

        """
        通常の処理
        """
        # あり得る最大値を計算
        max_score = self.action_selector.calc_max_score(others_info)
        declared_sum = actions[1] - 1
        if declared_sum > max_score:
            return -1

        # 自分と次のプレイヤーのカードを全通り試して、あり得る中での最小値
        next_min_score = self.action_selector.calc_next_min(others_info, declared_sum)

        next_player_name = None
        for info in others_info:
            if info["is_next"]:
                next_player_name = info["name"]
                break

        gap_player = False
        if next_player_name in self.player_declaration_gap:
            gap_cnt, total_cnt = self.player_declaration_gap[next_player_name]
            if gap_cnt / total_cnt >= 0.4:
                gap_player = True

        # 自分と次のプレイヤーのカードを全通り試して、あり得る中での最小値
        action = self.action_selector.choice_action(
            others_info, unseen_probs, declared_sum, next_min_score, gap_player
        )
        return action
