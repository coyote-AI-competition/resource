from client.not_websocket_client import Client

class SampleClient(Client):
    def AI_player_action(self,others_info, see_sum, log, actions, round_num):
      # カスタムロジックを実装
        print(f"[SampleClient] AI deciding action based on sum: {see_sum}, log: {log}, actions: {actions},others_info: {others_info}, round_num: {round_num}")

        all_cards = [-10, -5, -5 , 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 10, 10, 10, 15, 15, 20, 100, 101, 102, 103]
        my_possible_cards = all_cards

        opponent_cards = []
        for other_info in others_info:
          my_possible_cards.remove(other_info["card_info"])
          opponent_cards.append(other_info["card_info"])

        # 事前分布の準備
        possible_sums = []
        for card in my_possible_cards:
          if not card in [100, 101, 102, 103]:
            possible_sums.append(see_sum + int(card))
          elif card == 100:
            possible_sums.append(see_sum * 2)
          elif card == 101:
            tmp = see_sum - max(opponent_cards)
            possible_sums.append(tmp)
          elif card == 102:
            possible_sums.append(see_sum + 1)
          elif card == 103:
            for card in my_possible_cards:
              if not card in [100, 101, 102, 103]:
                possible_sums.append(see_sum + int(card))
              elif card == 100:
                possible_sums.append(see_sum * 2)
              elif card == 101:
                tmp = see_sum - max(opponent_cards)
                possible_sums.append(tmp)
              elif card == 102:
                possible_sums.append(see_sum + 1)
              elif card == 103:
                pass

        sum_range = list(range(-20, 140 + 1))
        p_list = []
        for i in sum_range:
          p = possible_sums.count(i) / len(possible_sums)
          p_list.append(p)

        # 事後分布の計算
        for l in log:
            act = int(l['action']) 

            likelihoods_for_sum = []
            for s_idx, s_val in enumerate(sum_range): 
                true_sum_candidate = s_val
                if true_sum_candidate < act:
                    likelihood = 0.1
                else:
                    likelihood = 0.9 
                likelihoods_for_sum.append(likelihood)

            new_p_list = [p_prior * likelihood for p_prior, likelihood in zip(p_list, likelihoods_for_sum)]

            # 正規化
            sum_new_p = sum(new_p_list)
            if sum_new_p > 0:
                p_list = [p / sum_new_p for p in new_p_list]
            else:
                p_list = [1.0 / len(p_list)] * len(p_list)

        try:
          tmp = act
        except UnboundLocalError:
          act = -20

        if sum(p_list[act + 20 :]) < 0.5 and act != -20:
          action = -1
        elif act == -20:
          action = 1
        else:
          action = p_list[act + 20 :].index(max(p_list[act + 20 :])) + act + 1

        print(f"[SampleClient] AI selected action: {action}")
        return action
