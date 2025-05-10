from .not_websocket_client import Client
import random
import logging

class Deck:
    def __init__(self):
        #初期条件
        #?→ max→0→×2:103,102,101,100に対応
        self.cards = [-10, -5, -5, 0, 0, 0,
                      1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3,
                      4, 4, 4, 4, 5, 5, 5, 5,
                      10, 10, 10, 15, 15, 20,
                      100, 101, 102, 103]

        #self.cashed_cards = [] #山札に戻すカードを格納するリスト

    def shuffle(self):
        print ("Deck shuffled.")
        random.shuffle(self.cards)

    def draw(self):
        if len(self.cards) > 0:
            return self.cards.pop()
        else:
            print("No card left in the deck.")
            #random.shuffle(self.cashed_cards) #山札に戻すカードをシャッフルする
            #山札が空になったら、捨て札を山札に追加する
           # self.cards = self.cashed_cards.copy()
           # self.cashed_cards = []
            self.reset()
            return self.cards.pop()

    def top_show_card(self):
        if len(self.cards) > 0:
            return self.cards[-1]
        return None

    def reset(self):
        self.cards = [-10, -5, -5, 0, 0, 0,
                      1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3,
                      4, 4, 4, 4, 5, 5, 5, 5,
                      10, 10, 10, 15, 15, 20,
                      100, 101, 102, 103]
        self.shuffle()

class SampleClient(Client):
           
    def AI_player_action(self,others_info, sum, log, player_card, actions, round_num):
        print(f"[SampleClient] AI deciding action based on sum: {sum}, log: {log}, actions: {actions},others_info: {others_info}, round_num: {round_num}" )
       

        RED = '\033[31m'
        END = '\033[0m'
        logging.info("select_action: %s" ,select_action)
        return  select_action
      
