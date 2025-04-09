import pyspiel
import numpy as np
import coyote
from open_spiel.python.algorithms import external_sampling_mccfr
from open_spiel.python.algorithms import exploitability
import random
import time

_NUM_PLAYERS = 2
_NUM_DECLARATIONS = 121  # 宣言の最大値

_GAME_TYPE = pyspiel.GameType(
    short_name="python_coyote",
    long_name="Python Coyote",
    dynamics=pyspiel.GameType.Dynamics.SEQUENTIAL,
    chance_mode=pyspiel.GameType.ChanceMode.EXPLICIT_STOCHASTIC,
    information=pyspiel.GameType.Information.IMPERFECT_INFORMATION,
    utility=pyspiel.GameType.Utility.ZERO_SUM,
    reward_model=pyspiel.GameType.RewardModel.TERMINAL,
    max_num_players=2,
    min_num_players=2,
    provides_information_state_string=True,
    provides_information_state_tensor=False,
    provides_observation_string=True,
    provides_observation_tensor=False,
    parameter_specification={},  # オプションだが明示
    default_loadable=False,      # 自作PythonゲームはFalseにしておくと安心
    provides_factored_observation_string=False
)


_GAME_INFO = pyspiel.GameInfo(
    num_distinct_actions=_NUM_DECLARATIONS + 1,  # 宣言 + チャレンジ
    max_chance_outcomes=0,
    num_players=_NUM_PLAYERS,
    min_utility=-1.0,
    max_utility=1.0,
    utility_sum=0.0,
    max_game_length=100,
)


class CoyoteState(pyspiel.State):
    def __init__(self, game):
        super().__init__(game)
        self._cur_player = 0
        self._is_terminal = False
        self._player_lives = [3] * _NUM_PLAYERS
        self._player_active = [True] * _NUM_PLAYERS
        self.current_declaration = 0
        self.last_declarer = None
        rnd = random.Random(time.time_ns())
        self.deck = coyote.game.Deck() # 使用デッキ
        self.deck.shuffle() # デッキをシャッフル
        self._cards = [self.deck.draw() for _ in range(_NUM_PLAYERS)]  # 各プレイヤーに1枚ずつカードを配る
        self.is_double_card = False
        self.is_shuffle_card = False
        print("Game started!")

    def calc_card_sum(self, cards):
        return coyote.game.calc_card_sum(self, cards)

    def current_player(self):
        return pyspiel.PlayerId.TERMINAL if self._is_terminal else self._cur_player

    def _legal_actions(self, player):
        if not self._player_active[player]:
            return []
        if self.last_declarer is None: 
            return [self.current_declaration+1]
        actions = [self.current_declaration+1, _NUM_DECLARATIONS]
        return actions
    
    def _apply_action(self, action):
        if action == _NUM_DECLARATIONS:  # チャレンジ
            print(f"Player {self._cur_player} challenges with declaration {self.current_declaration}.")
            true_total = coyote.game.convert_card(self, self._cards, False, self.deck)
            if self.current_declaration > true_total:
                loser = self.last_declarer
            else:
                loser = self._cur_player
            self._player_lives[loser] -= 1
            if self._player_lives[loser] <= 0:
                self._player_active[loser] = False
            self._is_terminal = sum(self._player_active) <= 1
            self.current_declaration = 0
            if self.is_shuffle_card:
                # SHUFFLEカードを引いた => 山札をリセット
                self.deck.reset()
                self.is_shuffle_card = False
            self._cards = [self.deck.draw() if self._player_active[i] else 0 for i in range(_NUM_PLAYERS)]  # 新しいカードを配る
            print(f"New cards: {self._cards}")
        else:
            print(f"Player {self._cur_player} declares {action}.")
            self.current_declaration = action
            self.last_declarer = self._cur_player
        self._cur_player = (self._cur_player + 1) % _NUM_PLAYERS

    def is_terminal(self):
        return self._is_terminal

    def returns(self):
        if not self._is_terminal:
            return [0.0] * _NUM_PLAYERS
        winner = self._player_active.index(True)
        return [_NUM_PLAYERS - 1 if i == winner else -1.0 for i in range(_NUM_PLAYERS)]
    
    def information_state_string(self, player):
        # 自分のカードは「?」、他人のカードは見える
        visible_cards = [str(c) if i != player else "?" for i, c in enumerate(self._cards)]
        lives = ",".join(map(str, self._player_lives))
        return f"cards:{','.join(visible_cards)}|decl:{self.current_declaration}|lives:{lives}"

    def __str__(self):
        return f"Cards: {self._cards}\nLives: {self._player_lives}\nCurrent declaration: {self.current_declaration}"


class CoyoteObserver:
    def __init__(self, params):
        self.params = params or {}

    def set_from(self, state: CoyoteState, player: int):
        visible = [str(card) if i != player else "?" for i, card in enumerate(state._cards)]
        self.obs_str = f"Visible cards: {visible}\nDeclaration: {state.current_declaration}\nLives: {state._player_lives}"

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


if __name__ == "__main__":

    # cfrで数理モデルを作成

    game = CoyoteGame() # ゲームのインスタンスを作成

    solver = external_sampling_mccfr.ExternalSamplingSolver(game)

    NUM_ITERATIONS = 1

    for i in range(NUM_ITERATIONS):
        solver.iteration()
        if i % 1000 == 0:
            avg_policy = solver.average_policy()
            nash_conv = exploitability.nash_conv(game, avg_policy)
            print(f"Iteration {i}, NashConv = {nash_conv}")

    average_policy = solver.average_policy()

    for state in game.new_initial_state().legal_actions():
        print(f"Action {state}: {average_policy.action_probabilities(state)}")
