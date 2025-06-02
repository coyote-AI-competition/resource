import random 
import matplotlib.pyplot as plt
import numpy as np
# deckがわかっている場合のimport


def calculate_sum_without_random(players_card_list):
    """カードの合計値を計算する関数

    Args:
        players_card_list (list): プレーヤーのカードリスト
        1. 100以上のカードは特別なカードとして扱う
        2. 100 -> x2, 101 -> max0, 102 -> 0 ,103 -> ?
        3. 100以上のカードは、0にする
        4. 102は0を追加する
        5. 101は最大値を0にする

    Returns:
        tuple: 合計値とシャッフルの有無
    """
    
    calculate_sum = 0 
    number_cards: list[int] = [card for card in players_card_list if card < 100]
    special_cards: list[int] = [card for card in players_card_list if card >= 100]
    is_double = False
    is_suffle = False 
    for cards in special_cards:
        if cards == 102:
            number_cards.append(0)
        elif cards == 101:
            if number_cards:  # 空でなければ
                max_card_index = number_cards.index(max(number_cards))
                number_cards[max_card_index] = 0
                is_suffle = True
        elif cards == 100:
            number_cards.append(0)
            is_double = True
    if is_double:
        calculate_sum = sum(number_cards) * 2
    else:
        calculate_sum = sum(number_cards)
    return calculate_sum,is_suffle


def calculate_sum_coyote(players_card_list:list[int],now_dock:list[int]):
    caluclate_sum = 0 
    special_cards:list[int] = [card for card in players_card_list if card >= 100]
    if 103 in special_cards :
        card_expect = []
        # 103のカードを除外する
        special_cards.remove(103)
        for card in now_dock:
            # cardをplayers_card_listに追加する
            random_players_card_list = players_card_list + [card]
            # その合計値を計算する
            caluclate_sum,is_suffle = calculate_sum_without_random(random_players_card_list)
            card_expect.append(caluclate_sum)
        return card_expect
            
    else:
        caluclate_sum,is_suffle =  calculate_sum_without_random(players_card_list)
        return [caluclate_sum]
    
def calculate_sum_with_deck_state(observation_card:list[int],decks:list[int]):
    calculate_sum_list = []
    for i in decks:
        all_cards = observation_card+[i]
        calculate_sum = calculate_sum_coyote(all_cards,decks)
        calculate_sum_list.extend(calculate_sum)
    return sorted(calculate_sum_list)

def calculate_sum_state_with_card(observation_card:list[int],mycard:int):
    coyote_card = [100, 101, 102, 103, -10, -5, -5, 0, 0, 0, 
        1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4,
        5, 5, 5, 5, 10, 10, 10, 15, 15, 20]
    all_cards = observation_card + [mycard]
    # ここで、coyote_cardの中からmycardとobservation_cardを除外する　ただし一づつ
    # そのために、observation_cardの中からmycardを除外する
    for card in observation_card:
        if card in coyote_card:
            coyote_card.remove(card)
    caluclate_sum = calculate_sum_coyote(all_cards,coyote_card)
    return caluclate_sum

def calculate_sum_expect_list(observation_card:list[int]):
    coyote_card = [100, 101, 102, 103, -10, -5, -5, 0, 0, 0, 
        1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4,
        5, 5, 5, 5, 10, 10, 10, 15, 15, 20]
    # coyote_cardの中からmycardとobservation_cardを除外する　ただし一づつ
    # observation_cardの中からmycardを除外する
    for card in observation_card:
        if card in coyote_card:
            coyote_card.remove(card)
    
    caluclate_sum_list = []
    for i in coyote_card:
        all_cards = observation_card 
        caluclate_sum = calculate_sum_state_with_card(all_cards,i)
        caluclate_sum_list.extend(caluclate_sum)    
    return sorted(caluclate_sum_list)


def expect_value(observation:list[int]):
    return sum(observation) / len(observation)


# サンプリングした後に自分が勝てる期待値を計算する。
def current_win_ratio(sampling_actions_list:list[int],current_declare):
    min_action = min(sampling_actions_list)
    max_action = max(sampling_actions_list)
    # もしすでに宣言されているcurrent_declareよりも大きい場合には、勝てる確率は0
    if max_action < current_declare:
        return 0
    # もしすでに宣言されているcurrent_declareよりも小さい場合には、勝てる確率は1
    elif min_action > current_declare:
        return 1
    # それ以外の場合には、勝てる確率は、とりうる状態数が残りどれだけあるかで計算する
    else:
        # current_declareよりも大きい sampling_actions_listの数を数える
        count = 0
        for action in sampling_actions_list:
            if action > current_declare:
                count += 1
        # sampling_actions_listの数を数える
        total_count = len(sampling_actions_list)
        # 勝てる確率を計算する
        win_ratio = count / total_count
        return win_ratio




if __name__ == '__main__':
    coyote_card = [100, 101, 102, 103, -10, -5, -5, 0, 0, 0, 
        1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4,
        5, 5, 5, 5, 10, 10, 10, 15, 15, 20]
    for i in range(5):
        observation_card = random.sample(coyote_card, 5)
        print(observation_card)
        game_actions = calculate_sum_expect_list(
            observation_card
        )
        print(game_actions)
        # 一番出現回数の多いものを取得する
        E = expect_value(game_actions)
        N = 5
        # 勝率の計算 Eの場合
        win_ratio = current_win_ratio(game_actions,E-N)
        print(win_ratio)
        
        #===============
        # グラフの描画
        #===============
        y_max = max(game_actions.count(i) for i in game_actions)
        plt.hist(game_actions,bins=100)
        plt.vlines(
            E,
            color='red',
            label='expect_value',
            ymax=y_max,
            ymin=0,
        )
        plt.show()
        
    
    