from .action_selector import ActionSelector
from .cards_predictor import CardsPredictor


class MyLibrary:
    def __init__(self):
        self.action_selector = ActionSelector()
        self.cards_predictor = CardsPredictor()

    def action(self, others_info, log, actions):
        self.cards_predictor.get_others_info(others_info)
        unseen_probs = self.cards_predictor.get_unseen_probs()

        """
        例外処理
        """
        # コヨーテしかできない場合
        if len(actions) == 1:
            return -1
        # 最初の番であり、コヨーテが出来ない場合は0以上であり得る数の最小
        if actions[1] == 1:
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

        # 自分と次のプレイヤーのカードを全通り試して、あり得る中での最小値
        action = self.action_selector.choice_action(
            others_info, unseen_probs, declared_sum, next_min_score
        )
        return action
