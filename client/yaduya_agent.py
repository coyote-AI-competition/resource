from typing import Literal
from .not_websocket_client import Client
import random
from .yaduya.logic import *  # または必要な関数/クラス名を指定
from .yaduya.replay import ReplayBuffer
from .yaduya.Reinforce import Agent
import logging

# ログの設定
logging.basicConfig(
  filename='example.log',   # ログファイルの名前
  level=logging.DEBUG,      # ログレベル（DEBUG以上のレベルが記録される）
)
# N = Player 
# 初めの場合には、Estimate-N を行い、そのほかの場合には、+1をする戦法
class PlayerReinforce(Client):
    def __init__(self, player_name="player1", is_ai=False):
        super().__init__(player_name, is_ai)
        self.all_cards =[100, 101, 102, 103, -10, -5, -5, 0, 0, 0, 
        1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4,
        5, 5, 5, 5, 10, 10, 10, 15, 15, 20]
        # 100 -> x2, 101 -> max0, 102 -> 0 ,103 -> ?
        self.shuffle = False
        self.estimate_now_decks_card = self.all_cards
        self.agent = Agent(state_size=4, action_size=2)
        self.previous_state = None
        self.count = 0 
        self.coyote = False
        self.previous_round = 0
        self.done = False
        self.previous_action = 1
        
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
        4. 勝てなければ-1をして、勝てる場合には+1をする。
        """
        # 認識できるカード情報を取得する
        observation_cards = self.players_card_list(others_info)
        estimate_now_deck = self.estimate_now_deck(observation_cards)
        # このカードの中で動けるカードを推定する
        action_space=calculate_sum_with_deck_state(observation_cards,estimate_now_deck)
        current_declare = self.get_current_declare(actions)
        # 現在の勝率を計算する
        win_ratio = current_win_ratio(action_space,current_declare)
        E = int(expect_value(action_space))
        
        # 自分が次の行動によりwinratioが変化するかを求める
        action = 1
        next_win_ratio: float | Literal[0] | Literal[1] = current_win_ratio(action_space,current_declare+action)
        if   0 == win_ratio :
            # もしもかなり勝率が低い場合にはコヨーテをする
            return -1
        else:
            # それ以外の時には、+1戦略を行う
            return current_declare + 1
    
    
    
    def learning_step(self,others_info,actions,round,log):
        """
        学習するためのものを出力するためのメソッド
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
        E = int(expect_value(action_space))
        
        state = [
            win_ratio,
            current_declare,
            E,
            N
        ]
        
        
        next_win_ratio = current_win_ratio(action_space,current_declare+N)
        next_state = [
            next_win_ratio,
            current_declare+N,
            E,
            N
        ]
        
        self.done = self.round_estimate(round)
        action = self.agent.get_action(state)
        
        if self.done:
            if len(log) == 0:
                if self.previous_state == None:
                    reward = -1
                self.agent.add_experience(
                        state,
                        self.previous_action,
                        reward,
                        next_state,
                        self.done,
                        is_next=True
                    )
                else:
                    reward = -1
                    self.agent.add_experience(
                        self.previous_state,
                        action,
                        reward,
                        next_state,
                        self.done,
                        is_next=True
                    )
                
                logging.debug(f"log is 0 count: {self.count}, epsilon: {self.agent.epsilon}, state: {state}, action: {action}, reward: {reward}, next_state: {next_state}, done: {self.done}")
                logging.debug(f"action: {action}")
                self.previous_state = None
                self.done = False
            else:
                if self.previous_state == None:
                    reward = 1
                    self.agent.add_experience(
                        state,
                        action,
                        reward,
                        next_state,
                        self.done,
                        is_next=True
                    )
                else:
                    reward = 1
                    self.agent.add_experience(
                        self.previous_state,
                        action,
                        reward,
                        next_state,
                        self.done,
                        is_next=True
                    )
                logging.debug(f"coyo-te sucess!!: {self.count}, epsilon: {self.agent.epsilon}, state: {state}, action: {action}, reward: {reward}, next_state: {next_state}, done: {self.done}")
                logging.debug(f"action: {action}")
                self.previous_state = None
                self.done = False
                
        else:
            if self.previous_state == None:
                self.previous_state = state
                reward = 1
                self.agent.add_experience(
                    state,
                    action,
                    reward,
                    next_state,
                    self.done,
                    is_next=False
                )
                self.previous_state = state
            else:
            # もしもすでに経験がある場合には、次の状態を更新する。
                if self.previous_state == state:
                    pass 
                else:
                    reward = 1
                    self.agent.add_experience(
                        self.previous_state,
                        action,
                        reward,
                        next_state,
                        self.done,
                        is_next=False
                    )
                logging.debug(f"count: {self.count}, epsilon: {self.agent.epsilon}, state: {state}, action: {action}, reward: {reward}, next_state: {next_state}, done: {self.done}")
                logging.debug(f"action: {action}")
            self.previous_state = state
        self.previous_action = action
        
        self.agent.update()
        if self.agent.epsilon < self.agent.epsilon_end:
            pass
        else:
            self.agent.set_epsilon()
        self.count += 1
        print('count', self.count)
        if self.count % 1000 == 0:
            self.agent.sync_net()
            print('sync!')
            self.agent.save_model(self.player_name)
        
        if action == 0 :
            return -1 
        else:
            return current_declare + action
    
    
    def get_current_declare(self,actions):
        """現在の宣言を取得する

        Args:
            actions (list): 行動可能なアクションのリスト
        """
        if len(actions) == 1:
            return 139
        elif len(actions) >= 2:
            return actions[1]-1
        else:
            return 0
        
    def round_estimate(self,round_num):
        if round_num == 1:
            self.shuffle = True
            self.privious_round = round_num
            coyote = False
        else:
            if self.previous_round != round_num:
                self.previous_round= round_num
                coyote = True
            else:
                coyote = False
        return coyote
    
    def AI_player_action(self,others_info, sum, log, actions, round_num):
        action = self.learning_step(
            others_info,
            actions,
            round_num,
            log
        )
        print("round",round_num)
        return action



class PlayerLogi(PlayerReinforce):
    def __init__(self, player_name="player1", is_ai=False):
        super().__init__(player_name, is_ai)
    def AI_player_action(self,others_info, sum, log, actions, round_num):
        action = self.action_estimate(
            others_info,
            actions
        )
        return action




