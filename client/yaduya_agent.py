from typing import Literal
from .not_websocket_client import Client
import random
from .yaduya.logic import *  # または必要な関数/クラス名を指定
# N = Player 
# 初めの場合には、Estimate-N を行い、そのほかの場合には、+1をする戦法
class PlayerN(Client):
    def __init__(self, player_name="player1", is_ai=False):
        super().__init__(player_name, is_ai)
        self.all_cards =[100, 101, 102, 103, -10, -5, -5, 0, 0, 0, 
        1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4,
        5, 5, 5, 5, 10, 10, 10, 15, 15, 20]
        # 100 -> x2, 101 -> max0, 102 -> 0 ,103 -> ?
        self.shuffle = False
        self.estimate_now_decks_card = self.all_cards
        
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
            self.estimate_now_decks_card= [card for card in self.all_cards if card not in players_card_list]
            self.shuffle = False
        else:
            self.estimate_now_decks_card = [card for card in self.estimate_now_decks_card if card not in players_card_list]
            if len(self.estimate_now_decks_card) == 0:
                self.estimate_now_decks_card = [card for card in self.all_cards if card not in players_card_list]
        return self.estimate_now_decks_card
    
    def action_estimate(self,others_info,actions):
        """行動可能なアクションの推定を行う
        1. 自分のカードを取得する
        2. 行動可能なアクションを全て取得する。
        3. 現在の勝率を計算する
        """
        # メンバーの数の取得
        N = len(others_info) + 1
        # 認識できるカード情報を取得する
        observation_cards = self.players_card_list(others_info)
        estimate_now_deck = self.estimate_now_deck(observation_cards)
        # このカードの中で動けるカードを推定する
        action_space=calculate_sum_with_deck_state(observation_cards,estimate_now_deck)
        current_declare = self.get_current_declare(actions)
        # 現在の勝率を計算する
        win_ratio = current_win_ratio(action_space,current_declare)
        # 自分が次の行動によりwinratioが変化するかを求める
        action = 1
        next_win_ratio: float | Literal[0] | Literal[1] = current_win_ratio(action_space,current_declare+action)
        
        if win_ratio > 0 and win_ratio < 0.1:
            # もしもかなり勝率が低い場合にはコヨーテをする
            return -1
        elif next_win_ratio == win_ratio:
            # この場合には、+1戦略をとる
            return current_declare + 1
        elif next_win_ratio > win_ratio:
            # この場合には、+1戦略をとる
            print('==============明かなバグです====================')
            return current_declare + 1
        elif next_win_ratio < win_ratio:
            # この場合には、戦略を+N戦法として次の自分のターンの時の状況を考える
            next_trun_win_ratio = current_win_ratio(action_space,current_declare+N)
            if next_trun_win_ratio == next_win_ratio:
                # この場合には、+1戦略をとる
                return current_declare + 1
            elif next_trun_win_ratio > next_win_ratio:
                print('==============明かなバグです====================')
                return current_declare + 1
            else :
                if next_trun_win_ratio < 0.05:
                    # もしもかなり勝率が低い場合にはコヨーテをする
                    return -1
                    
                else:
                    # それ以外の場合には、+1戦略をとる
                    return current_declare + 1
        else:
            # それ以外の場合には、+1戦略をとる
            print('==============明かなバグです====================')
            return current_declare + 1
        
    def get_current_declare(self,actions):
        """現在の宣言を取得する

        Args:
            actions (list): 行動可能なアクションのリスト
        """
        if len(actions) == 1:
            return 139
        elif len(actions) >= 2:
            return actions[1]
        else:
            return 0
    
    def AI_player_action(self,others_info, sum, log, actions, round_num):
        # カスタムロジックを実装
        action = self.action_estimate(others_info,actions)
        return action








