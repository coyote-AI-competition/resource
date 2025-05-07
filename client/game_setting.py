import pyspiel
import numpy as np
from open_spiel.python.algorithms import external_sampling_mccfr
from open_spiel.python.algorithms import get_all_states
from open_spiel.python.algorithms import deep_cfr
import random
import pickle
import time
import tensorflow
import os
import tqdm
import matplotlib.pyplot as plt
import numpy as np
from pyspiel import PlayerId

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()  # TF1互換モードに切り替え


_NUM_PLAYERS = 6  # プレイヤー数
_NUM_DECLARATIONS = 121  # 宣言の最大値

_GAME_TYPE = pyspiel.GameType(
    short_name="python_coyote",
    long_name="Python Coyote",
    dynamics=pyspiel.GameType.Dynamics.SEQUENTIAL,
    chance_mode=pyspiel.GameType.ChanceMode.EXPLICIT_STOCHASTIC,
    information=pyspiel.GameType.Information.IMPERFECT_INFORMATION,
    utility=pyspiel.GameType.Utility.GENERAL_SUM,
    reward_model=pyspiel.GameType.RewardModel.TERMINAL,
    max_num_players=6,
    min_num_players=2,
    provides_information_state_string=True,
    provides_information_state_tensor=True,
    provides_observation_string=True,
    provides_observation_tensor=False,
    parameter_specification={},  # オプションだが明示
    default_loadable=False,      # 自作PythonゲームはFalseにしておくと安心
    provides_factored_observation_string=False
)

_GAME_INFO = pyspiel.GameInfo(
    num_distinct_actions=2,  # 宣言 + チャレンジ
    max_chance_outcomes=15,  # 山札のカードの種類数
    num_players=_NUM_PLAYERS,
    min_utility=-1.0,
    max_utility=1.0,
    utility_sum=0.0,
    max_game_length=100,
)

class Deck:
    def __init__(self):
        #初期条件
        #?→ max→0→×2:103,102,101,100に対応
        self.cards = [-10, -5, -5, 0, 0, 0,
                      1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3,
                      4, 4, 4, 4, 5, 5, 5, 5, 
                      10, 10, 10, 15, 15, 20, 
                      100, 101, 102, 103]
        
        self.cashed_cards = [] #山札に戻すカードを格納するリスト
    
    def shuffle(self):
        # print ("Deck shuffled.")
        random.shuffle(self.cards)
    
    def draw(self):
        if len(self.cards) > 0:
            return random.choice(self.cards) #ランダムにカードを引く
        else:
            # print("No card left in the deck.")
            random.shuffle(self.cashed_cards) #山札に戻すカードをシャッフルする
            #山札が空になったら、捨て札を山札に追加する
            self.reset()
            return random.choice(self.cards) #ランダムにカードを引く
    
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

class CoyoteState(pyspiel.State):
    def __init__(self, game):
        super().__init__(game)
        self._cur_player = 0
        self._is_terminal = False
        self._player_lives = [1] * _NUM_PLAYERS
        self._player_active = [True] * _NUM_PLAYERS
        self.current_declaration = 0
        self.last_declarer = _NUM_PLAYERS - 1
        self._calc_continue = False
        self.question_index = None
        rnd = random.Random(time.time_ns())
        self.deck = Deck() # 使用デッキ
        self.deck.shuffle() # デッキをシャッフル
        self._cards = [None] * _NUM_PLAYERS  # プレイヤーの手札
        self.is_double_card = False
        self.is_shuffle_card = False
        self._expecting_draw = True  # チャンスノードに移行するかどうか
        self._all_cards = [-10, -5, -5, 0, 0, 0,
                      1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3,
                      4, 4, 4, 4, 5, 5, 5, 5, 
                      10, 10, 10, 15, 15, 20, 
                      100, 101, 102, 103]
        # randomにカードを捨てる
        discard_cards_num = rnd.randint(0, len(self.deck.cards) - 1)
        for _ in range(discard_cards_num):
            discard_card = self.deck.draw()
            self.deck.cashed_cards.append(discard_card)
            self.deck.cards.remove(discard_card)
        self._used_cards = []  # 使用済みカード
    
        # print("Game started!")


    def current_player(self):
        if self._is_terminal:
            return PlayerId.TERMINAL
        if self._expecting_draw:
            return PlayerId.CHANCE   # ← チャンスノードに移行
        return self._cur_player       # ← プレイヤーノード
                    
    def convert_card(self, cards, Is_othersum, deck):
            # print (f"cards: {cards}")
            true_cards = sorted(cards, reverse = True) 
            index = 0 
            # print(f"Initial true_cards: {true_cards}")
            while index < len(true_cards):
                card = true_cards[index]
                # print(f"Card drawn: {card}")

                #?を引いたら次のカードを引き、出た番号のカードと交換する
                #全体の数の計算はラウンドにつき一回

                #maxを引いたら、最も大きいカードを0にする      
                if(card == 102):
                    normal_cards = [c for c in true_cards if c < 100] #通常カードを取得
                    if len(normal_cards) != 0:
                        max_card = max(c for c in true_cards if c < 100) #最大値を取得
                        max_index = true_cards.index(max_card) #最大値のインデックスを取得
                        true_cards[max_index] = 0 #最大値を0にする
                    true_cards[true_cards.index(102)] = 0    
                    
                #0(黒背景)を引いたら、ラウンド終了後山札をリセットする        
                elif(card == 101):
                    true_cards[index] = 0
                    #true_cards = sorted(( card for card in true_cards),reverse=True)
                    self.is_shuffle_card = True
                elif(card == 100):
                    true_cards[index] = 0
                    #true_cards = sorted(( card for card in true_cards),reverse=True)
                    self.is_double_card = True
                
                index += 1      
        
            return self.calc_card_sum(true_cards)   #関数の外に合計値を返す               

    def set_initial_state(self, player_num, current_declaration, used_cards, other_cards):
        self._player_lives = [0] * _NUM_PLAYERS
        self._player_active = [False] * _NUM_PLAYERS
        for i in range(player_num):
            self._player_lives[i] = 1
            self._player_active[i] = True
        self.current_declaration = current_declaration
        self.last_declarer = player_num - 1
        self._used_cards = used_cards
        print(other_cards)
        for i in range(_NUM_PLAYERS):
            if i == 0:
                self._cards[i] = "error"
            elif i < player_num:
                self._cards[i] = other_cards[i-1]
            else:
                self._cards[i] = 0
        self._expecting_draw = False  # チャンスノードに移行するかどうか

    def calc_card_sum(self, cards):
        card_sum = 0 #初期化
        for card in cards:
            card_sum += card
        if(self.is_double_card):
            card_sum *= 2 
            self.is_double_card = False
        # print(f"gamesum is {card_sum}")    
        return card_sum    

    def _legal_actions(self, player):
        # if not self._player_active[player]:
        #     return []
        # if self.last_declarer is None: 
        #     return [0]  # 最初のプレイヤーは宣言のみ
        # if self.current_declaration >= _NUM_DECLARATIONS - 1:
        #     return [1]
        actions = [0, 1] # 0: 宣言, 1: チャレンジ
        return actions
    
    def chance_outcomes(self):
        # 山札に残ったカードのカウント
        counts = {}
        for card in self.deck.cards:
            counts[card] = counts.get(card, 0) + 1
        total = len(self.deck.cards)
        return [(card, cnt/total) for card, cnt in counts.items()]

    def _apply_action(self, action):
        # Chanceノードに移行する場合
        if self.current_player() == PlayerId.CHANCE:
            # question カードを引いた場合
            if self._calc_continue:
                drawn_card = action  # action がカード値
                self._cards[self.question_index] = drawn_card
                self.deck.cards.remove(drawn_card)
                self._used_cards.append(drawn_card)
                self._calc_continue = False
                self._expecting_draw = False
                if self.deck.cards == []:
                    self.deck.reset()
                    self._used_cards = []
                return
            # ドロー処理
            drawn_card = action  # action がカード値
            self._cards[self._cur_player] = drawn_card
            self.deck.cards.remove(drawn_card)  # 山札からカードを削除
            if self.deck.cards == []:
                self.deck.reset()
                self._used_cards = []
            self._cur_player = (self._cur_player + 1) % _NUM_PLAYERS
            while not self._player_active[self._cur_player]:
                self._cards[self._cur_player] = 0
                self._cur_player = (self._cur_player + 1) % _NUM_PLAYERS
            # フラグクリア＆通常プレイヤーに戻す
            if not None in self._cards:
                self._expecting_draw = False
            else:
                self._expecting_draw = True
            return

        if (action == 1 or self.current_declaration >= _NUM_DECLARATIONS - 1) or self._calc_continue:  # チャレンジ
            # print(f"Current Cards: {self._cards}")
            # print(f"Player {self._cur_player} challenges with declaration {self.current_declaration}.")
            if 103 in self._cards:
                question_index = self._cards.index(103)
                self._cards[question_index] = None
                self.question_index = question_index
                self._calc_continue = True
                self._expecting_draw = True
                return

            true_total = self.convert_card(self._cards, False, self.deck)

            # used_cardsにカードを追加
            for i in range(_NUM_PLAYERS):
                if self._player_active[i] and i != self._cur_player:
                    # プレイヤーがアクティブな場合はカードを追加自分のカードは見えない
                    if self._cards[i] in self._all_cards and i != self.question_index:
                        self._used_cards.append(self._cards[i])

            if self.current_declaration > true_total:
                loser = self.last_declarer
            else:
                loser = self._cur_player
            self._player_lives[loser] -= 1
            if self._player_lives[loser] <= 0:
                self._player_active[loser] = False
                self._is_terminal = True
            self.current_declaration = 0

            if self.is_shuffle_card:
                # SHUFFLEカードを引いた => 山札をリセット
                self.deck.reset()
                self._used_cards = []
                self.is_shuffle_card = False
            self._cards = [None] * _NUM_PLAYERS  # プレイヤーの手札をNoneにする
            self._expecting_draw = True
            self.question_index = None
            # used_cardsにカードを追加
            # print(f"Used cards: {self._used_cards}")
            # print(f"Player {loser} loses a life!")
            # print(f"Player lives: {self._player_lives}")
        else:
            # print(f"Player {self._cur_player} declares {self.current_declaration+1}.")
            self.current_declaration = self.current_declaration + 1
            self.last_declarer = self._cur_player

        self._cur_player = (self._cur_player + 1) % _NUM_PLAYERS
        while not self._player_active[self._cur_player]:
            self._cur_player = (self._cur_player + 1) % _NUM_PLAYERS

    def is_terminal(self):
        # if self._is_terminal:
        #     print(f"Game over! Player {self._cur_player} wins!")
        return self._is_terminal

    def returns(self):
        reward_ls = np.array([1 if self._player_active[i] else -1 for i in range(_NUM_PLAYERS)])
        return reward_ls
    
    def information_state_string(self, player=None):
        if player is None:
            player = self._cur_player

        # プレイヤーの情報状態を文字列に変換
        # 宣言値 1 + プレイヤーのライフ _NUM_PLAYERS + プレイヤーの手札 (_NUM_PLAYERS - 1) * 4 + remaining_cards 15
        # 宣言値を文字列に変換
        declaration_str = str(self.current_declaration)
        # プレイヤーのライフを文字列に変換
        lives_str = ",".join([str(self._player_lives[(i+player) % _NUM_PLAYERS]) for i in range(_NUM_PLAYERS)])
        # プレイヤーの手札を文字列に変換
        hand_str = ",".join([str(self._cards[(i+player+1) % _NUM_PLAYERS]) for i in range(_NUM_PLAYERS-1)])
        # 残りのカードを文字列に変換
        cards_kind = [-10, -5, 0, 1, 2, 3, 4, 5, 10, 15, 20, 100, 101, 102, 103]
        visible_cards = self._all_cards.copy()
        for card in self._used_cards:
            if card in visible_cards:
                visible_cards.remove(card)
        for card in [self._cards[i] for i in range(len(self._cards)) if i != player and self._player_active[i]]:
            if card in visible_cards:
                visible_cards.remove(card)
        remaining_cards_str = ",".join([str(np.sum([1 for card in visible_cards if card == cards_kind[i]])) for i in range(15)])
        # プレイヤーの情報状態を文字列に結合
        info_state_str = f"{declaration_str},{lives_str},{hand_str},{remaining_cards_str}"
        return info_state_str

    def information_state_tensor(self, player=None):
        if player is None:
            player = self._cur_player
        # プレイヤーの情報状態をテンソルに変換
        # 宣言値 1 + プレイヤーの手札 (_NUM_PLAYERS - 1) * 4 + remaining_cards 15 + current_estimate + the difference between the current estimate and the current declaration
        tensor = np.zeros(1 + _NUM_PLAYERS + (_NUM_PLAYERS - 1)*4 + 15 + 1 + 1, dtype=np.float32)

        # 現在の宣言値を入れる
        tensor[0] = self.current_declaration

        # プレイヤーの手札を入れる
        for i in range(_NUM_PLAYERS-1):
            # (value,is_max,is_double,is_question)の形で各プレイヤー入れていく
            if self._cards[(i+player+1) % _NUM_PLAYERS] == 102:
                tensor[1 + i*4] = 0
                tensor[1 + i*4 + 1] = 1
                tensor[1 + i*4 + 2] = 0
                tensor[1 + i*4 + 3] = 0
            elif self._cards[(i+player+1) % _NUM_PLAYERS] == 100:
                tensor[1 + i*4] = 0
                tensor[1 + i*4 + 1] = 0
                tensor[1 + i*4 + 2] = 1
                tensor[1 + i*4 + 3] = 0
            elif self._cards[(i+player+1) % _NUM_PLAYERS] == 103:
                tensor[1 + i*4] = 0
                tensor[1 + i*4 + 1] = 0
                tensor[1 + i*4 + 2] = 0
                tensor[1 + i*4 + 3] = 1
            elif self._cards[(i+player+1) % _NUM_PLAYERS] == 101:
                tensor[1 + i*4] = 0
                tensor[1 + i*4 + 1] = 0
                tensor[1 + i*4 + 2] = 0
                tensor[1 + i*4 + 3] = 0
            else:
                tensor[1 + i*4] = self._cards[(i+player+1) % _NUM_PLAYERS]
                tensor[1 + i*4 + 1] = 0
                tensor[1 + i*4 + 2] = 0
                tensor[1 + i*4 + 3] = 0

        cards_kind = [-10, -5, 0, 1, 2, 3, 4, 5, 10, 15, 20, 100, 101, 102, 103]

        # 残りのカードを入れる
        visible_cards = self._all_cards.copy()
        for card in self._used_cards:
            if card in visible_cards:
                visible_cards.remove(card)
        for card in [self._cards[i] for i in range(len(self._cards)) if i != player and self._player_active[i]]:
            if card in visible_cards:
                visible_cards.remove(card)

        # 残りのカードを入れる
        for i in range(15):
            tensor[1 + (_NUM_PLAYERS - 1)*4 + i] = np.sum([1 for card in visible_cards if card == cards_kind[i]])

        # 現在の相手のカードの合計値を入れる
        current_estimate = 0
        max_card = -100
        max_flag = False
        double_flag = False
        for i in range(_NUM_PLAYERS):
            if self._cards[i] is None:
                continue
            elif (self._player_active[i]) and (i != player) and (self._cards[i] < 100):
                current_estimate += self._cards[i]
                if self._cards[i] > max_card:
                    max_card = self._cards[i]
            elif self._player_active[i] and (i != player) and (self._cards[i] == 100):
                current_estimate += 0
                double_flag = True
            elif self._player_active[i] and (i != player) and (self._cards[i] == 102):
                current_estimate += 0
                max_flag = True
        
        if max_flag:
            current_estimate -= max_card

        if double_flag:
            current_estimate *= 2

        tensor[1 + (_NUM_PLAYERS - 1)*4 + 15] = current_estimate
        # 現在の宣言値との差を入れる
        tensor[1 + (_NUM_PLAYERS - 1)*4 + 16] = current_estimate - self.current_declaration

        return tensor

    def __str__(self):
        return f"Cards: {self._cards}\nCurrent declaration: {self.current_declaration}\nLives: {self._player_lives}\nActive players: {self._player_active}"

class CoyoteObserver:
    def __init__(self, params):
        self.params = params or {}

    def set_from(self, state: CoyoteState, player: int):
        visible = [str(card) if i != player else "?" for i, card in enumerate(state._cards)]
        self.obs_str = f"Visible cards: {visible}\nDeclaration: {state.current_declaration}"

    def string_from(self, state: CoyoteState, player: int):
        self.set_from(state, player)
        return self.obs_str


class CoyoteGame(pyspiel.Game):
    def __init__(self, params=None):
        super().__init__(_GAME_TYPE, _GAME_INFO, params or {})
    
    def new_initial_state(self):
        return CoyoteState(self)
    
    def make_py_observer(self, iig_obs_type=None, params=None):
        return CoyoteObserver(params)


# 登録
pyspiel.register_game(_GAME_TYPE, CoyoteGame)

