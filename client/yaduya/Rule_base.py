from ..not_websocket_client import Client
import random


class ConstClient(Client):
    def __init__(self, player_name="RandomAI", is_ai=True):
        super().__init__(player_name=player_name, is_ai=is_ai)
        self.round_number = 0
        self.cards =[100, 101, 102, 103, -10, -5, -5, 0, 0, 0, 
        1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4,
        5, 5, 5, 5, 10, 10, 10, 15, 15, 20]
        # 100 -> x2, 101 -> max0, 102 -> 0 ,103 -> ?
        self.prev_round = 0
        self.shuffle = False
        
        
    @property 
    def hold_card_random(self):
        # ランダムなカードをもつ
        card = random.choice([card for card in self.cards if card < 100])
        return card
    
    def players_card_list(self, others_info):
        """PLAYERのカードリストを取得する
        1. others_infoからカード情報を取得
        2. players_card_listに格納する

        Args:
            others_info (list[dict]): 他のプレイヤーの情報を格納したリスト

        Returns:
            list[int]: 他のプレイヤーのカード情報を格納したリスト
        """
        N = len(others_info)
        players_card_list = []
        for i in range(N):
            players_card_list.append(others_info[i]["card_info"])
        return players_card_list
    
    # estimateしたものをさらにestimateする。自分のカードが取得することができないので、自分のestimateしたカードは除外されるようにした。
    def estimate_now_deck(self,players_card_list:list[int]):
        # すでに出たカードを除外した新しいデッキを作成する
        if self.shuffle:
            self.estimate_now_decks_card= [card for card in self.cards if card not in players_card_list]
            self.shuffle = False
        else:
            self.estimate_now_decks_card = [card for card in self.estimate_now_decks_card if card not in players_card_list]
            if len(self.estimate_now_decks_card) == 0:
                self.estimate_now_decks_card = [card for card in self.cards if card not in players_card_list]
                
        self.estimate_now_decks_card = [card for card in self.estimate_now_decks_card if card < 100]
        return self.estimate_now_decks_card
    def calclate_sum(self,players_card_list:list[int]):
        caluclate_sum = 0
        number_cards: list[int] = [card for card in players_card_list if card < 100]
        special_cards: list[int] = [card for card in players_card_list if card >= 100]
        is_double = False
        for cards in special_cards:
            if cards ==103:
                choice_card = random.choice(self.estimate_now_deck(players_card_list))
                number_cards.append(choice_card)
            elif cards == 102:
                number_cards.append(0)
                
            elif cards == 101:
                max_card_index = number_cards.index(max(number_cards))
                number_cards[max_card_index] = 0
                self.shuffle = True
            elif cards == 100:
                number_cards.append(0)
                is_double = True
        
        if is_double:
            caluclate_sum = sum(number_cards) * 2
        else:
            caluclate_sum = sum(number_cards)
        return caluclate_sum
    
    
    

    def AI_player_action(self,others_info, sum, log, actions, round_num):
        if self.round_number != round_num:
            self.round_number = round_num
            self.shuffle = True
            
        # カスタムロジックを実装
        players_card_list = self.players_card_list(others_info)
        # 自分のカードを推定する。
        my_card = self.hold_card_random
        players_card_list.append(my_card)
        caluclate_sum = self.calclate_sum(players_card_list)
        
        # アクションの方法 上げ幅を5に制限する。
        if len(actions) == 1:
            action = actions[0]
        else:
            prev_now_declare = actions[1] -1 
            print('estimate sum', caluclate_sum)
            if caluclate_sum >= prev_now_declare:
                action_space = [action for action in actions if action <= caluclate_sum]
                random_index = random.randint(0, len(action_space)-1)
                random_five_index = random.randint(0,5)
                if random_index < random_five_index:
                    action = action_space[random_index]
                else:
                    action = action_space[random_five_index]
                if action is None:
                    action = actions[1]
            else:
                Epsilon = random.uniform(0, 1)
                if Epsilon < 0.8:
                    if len(actions) > 1 :
                        action = actions[1]
                    else:
                        action = actions[0]
                else:
                    action = -1
        if len(log)==0:
            action = actions[1]
        print(actions)
        return action
    
class PlusOneClient(ConstClient):
    def AI_player_action(self,others_info, sum, log, actions, round_num):
        if self.round_number != round_num:
            self.round_number = round_num
            self.shuffle = True
            
        # カスタムロジックを実装
        players_card_list = self.players_card_list(others_info)
        # 自分のカードを推定する。
        my_card = self.hold_card_random
        players_card_list.append(my_card)
        caluclate_sum = self.calclate_sum(players_card_list)
        
        if len(actions) == 1:
            action = actions[0]
        else:
            prev_now_declare = actions[1] -1 
            print('estimate sum', caluclate_sum)
            if caluclate_sum >= prev_now_declare:
                # 基本的に+1を選択する関数
                action = actions[1]
            else:
                action = -1
        if len(log)==0:
            action = actions[1]
        return action
