import pyspiel
import numpy as np
import coyote
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
    utility=pyspiel.GameType.Utility.ZERO_SUM,
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
        self._player_lives = [1] * _NUM_PLAYERS
        self._player_active = [True] * _NUM_PLAYERS
        self.current_declaration = 0
        self.last_declarer = None
        rnd = random.Random(time.time_ns())
        self.deck = coyote.game.Deck() # 使用デッキ
        self.deck.shuffle() # デッキをシャッフル
        self._cards = [self.deck.draw() for _ in range(_NUM_PLAYERS)]  # 各プレイヤーに1枚ずつカードを配る
        self.is_double_card = False
        self.is_shuffle_card = False
        self._used_cards = []  # 使用済みカード
        self._all_cards = [-10, -5, -5, 0, 0, 0,
                      1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3,
                      4, 4, 4, 4, 5, 5, 5, 5, 
                      10, 10, 10, 15, 15, 20, 
                      100, 101, 102, 103]
        # print("Game started!")


    def calc_card_sum(self, cards):
        return coyote.game.calc_card_sum(self, cards)

    def current_player(self):
        return pyspiel.PlayerId.TERMINAL if self._is_terminal else self._cur_player

    def _legal_actions(self, player):
        # if not self._player_active[player]:
        #     return []
        # if self.last_declarer is None: 
        #     return [0]  # 最初のプレイヤーは宣言のみ
        # if self.current_declaration >= _NUM_DECLARATIONS - 1:
        #     return [1]
        actions = [0, 1] # 0: 宣言, 1: チャレンジ
        return actions
    
    def _apply_action(self, action):
        if (action == 1 or self.current_declaration >= _NUM_DECLARATIONS - 1) and self.last_declarer is not None:  # チャレンジ
            # print(f"Current Cards: {self._cards}")
            # print(f"Player {self._cur_player} challenges with declaration {self.current_declaration}.")
            true_total = coyote.game.convert_card(self, self._cards, False, self.deck)

            # used_cardsにカードを追加
            for i in range(_NUM_PLAYERS):
                if self._player_active[i]:
                    # プレイヤーがアクティブな場合はカードを追加
                    if self._cards[i] in self._all_cards:
                        self._used_cards.append(self._cards[i])

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
                self._used_cards = []
                self.is_shuffle_card = False
            for i in range(_NUM_PLAYERS):
                if self._player_active[i]:
                    if len(self.deck.cards) == 0:
                        self.deck.reset()
                        self._used_cards = []
                    # プレイヤーがアクティブな場合はカードを引く
                    card = self.deck.draw()
                    self._cards[i] = card
                else:
                    # プレイヤーがアクティブでない場合はカードを0にする
                    self._cards[i] = 0
            # used_cardsにカードを追加
            # print(f"New cards: {self._cards}")
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
        return self._is_terminal

    def returns(self):
        survive = sum(self._player_active)
        points = _NUM_PLAYERS - survive
        return [points/survive if self._player_active[i] else -1 for i in range(_NUM_PLAYERS)]
    
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
        # 宣言値 1 + プレイヤーのライフ _NUM_PLAYERS + プレイヤーの手札 (_NUM_PLAYERS - 1) * 4 + remaining_cards 15
        tensor = np.zeros(1 + _NUM_PLAYERS + (_NUM_PLAYERS - 1)*4 + 15, dtype=np.float32)

        # 現在の宣言値を入れる
        tensor[0] = self.current_declaration

        # プレイヤーのライフを入れる
        for i in range(_NUM_PLAYERS):
            tensor[i + 1] = self._player_lives[(i+player) % _NUM_PLAYERS]

        # プレイヤーの手札を入れる
        for i in range(_NUM_PLAYERS-1):
            # (value,is_max,is_double,is_question)の形で各プレイヤー入れていく
            if self._cards[(i+player+1) % _NUM_PLAYERS] == 102:
                tensor[1 + _NUM_PLAYERS + i*4] = 0
                tensor[1 + _NUM_PLAYERS + i*4 + 1] = 1
                tensor[1 + _NUM_PLAYERS + i*4 + 2] = 0
                tensor[1 + _NUM_PLAYERS + i*4 + 3] = 0
            elif self._cards[(i+player+1) % _NUM_PLAYERS] == 100:
                tensor[1 + _NUM_PLAYERS + i*4] = 0
                tensor[1 + _NUM_PLAYERS + i*4 + 1] = 0
                tensor[1 + _NUM_PLAYERS + i*4 + 2] = 1
                tensor[1 + _NUM_PLAYERS + i*4 + 3] = 0
            elif self._cards[(i+player+1) % _NUM_PLAYERS] == 103:
                tensor[1 + _NUM_PLAYERS + i*4] = 0
                tensor[1 + _NUM_PLAYERS + i*4 + 1] = 0
                tensor[1 + _NUM_PLAYERS + i*4 + 2] = 0
                tensor[1 + _NUM_PLAYERS + i*4 + 3] = 1
            elif self._cards[(i+player+1) % _NUM_PLAYERS] == 101:
                tensor[1 + _NUM_PLAYERS + i*4] = 0
                tensor[1 + _NUM_PLAYERS + i*4 + 1] = 0
                tensor[1 + _NUM_PLAYERS + i*4 + 2] = 0
                tensor[1 + _NUM_PLAYERS + i*4 + 3] = 0
            else:
                tensor[1 + _NUM_PLAYERS + i*4] = self._cards[(i+player+1) % _NUM_PLAYERS]
                tensor[1 + _NUM_PLAYERS + i*4 + 1] = 0
                tensor[1 + _NUM_PLAYERS + i*4 + 2] = 0
                tensor[1 + _NUM_PLAYERS + i*4 + 3] = 0

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
            tensor[1 + _NUM_PLAYERS + (_NUM_PLAYERS - 1)*4 + i] = np.sum([1 for card in visible_cards if card == cards_kind[i]])
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


if __name__ == "__main__":

    # cfrで数理モデルを作成
    # Check for TensorFlow GPU access

    game = CoyoteGame() # ゲームのインスタンスを作成

    dirname = os.path.dirname(__file__)

    # while True:
    # 保存先パス
    ckpt_path = os.path.join(dirname, "checkpoints", "deep_cfr.ckpt")
    # for _ in tqdm.tqdm(range(100000)):
    #     # セッション開始
    with tf.Session() as sess:

        # DeepCFRSolver 構築
        solver = deep_cfr.DeepCFRSolver(
            game=game,
            session=sess,
            policy_network_layers=[64, 32, 16, 8],
            advantage_network_layers=[64, 32, 16, 8],
            num_iterations=1,
            num_traversals=5,
            learning_rate=0.01,
            batch_size_advantage=32,
            batch_size_strategy=32,
            memory_capacity=10000
        )

        # 変数初期化
        sess.run(tf.global_variables_initializer())

        # saver 準備（solver後じゃないと変数が構築されてない）
        saver = tf.train.Saver()

        # チェックポイントが存在すれば復元
        if os.path.exists(ckpt_path + ".index"):
            print("✅ Checkpoint found. Restoring model...")
            saver.restore(sess, ckpt_path)
            print("✅ Model restored!")
            # epochの復元
            with open(os.path.join(dirname, "epoch.txt"), "r") as f:
                epoch = int(f.read())
            print("Epoch:", epoch)

            # Advantage lossesの復元
            with open(os.path.join(dirname, "advantage_losses.npy"), "rb") as f:
                advantage_losses_history = np.load(f)
            # print("Advantage losses:", advantage_losses_history[-1])

            # Policy lossの復元
            with open(os.path.join(dirname, "policy_loss.npy"), "rb") as f:
                policy_loss_history = np.load(f)
            print("Policy loss:", policy_loss_history[-1])
        
        else:

            # チェックポイントが存在しない場合は新規作成
            policy_loss_history = np.array([])
            advantage_losses_history = None


        # 学習（追加訓練）
        print("🚀 Starting DeepCFR training...")
        # 進捗表示
        for i in tqdm.tqdm(range(100000)):
            # 1イテレーションの学習
            _, advantage_losses, policy_loss = solver.solve()

            # 保存
            saver.save(sess, ckpt_path)
            print("💾 Model saved to:", ckpt_path)

            with open(os.path.join(dirname, "epoch.txt"), "w") as f:
                f.write(str(i + 1))

            advantage_losses_ls = np.array([advantage_losses[i] for i in range(_NUM_PLAYERS)]).reshape(-1, _NUM_PLAYERS)

            if advantage_losses_history is None:
                advantage_losses_history = advantage_losses_ls
            else:
                advantage_losses_history = np.vstack((advantage_losses_history, advantage_losses_ls))

            # historyに追加
            # advantage_losses_history = np.append(advantage_losses_history, advantage_losses)
            policy_loss_history = np.append(policy_loss_history, policy_loss)

            # 保存
            np.save(os.path.join(dirname, "advantage_losses.npy"), advantage_losses_history)
            np.save(os.path.join(dirname, "policy_loss.npy"), policy_loss_history)

            # plot loss
            for i in range(_NUM_PLAYERS):
                plt.plot(advantage_losses_history[:,i], label=f"Player_{i}_Advantage_Loss")
            plt.xlabel("Iteration")
            plt.ylabel("Loss")
            plt.yscale('log')
            plt.title("DeepCFR Advantage Loss")
            plt.legend()
            plt.savefig(os.path.join(dirname, "advantage_loss.png"))
            plt.close()
            
            # plot policy loss
            plt.plot(policy_loss_history, label="Policy Loss")
            plt.xlabel("Iteration")
            plt.yscale('log')
            plt.ylabel("Loss")
            plt.title("DeepCFR Loss")
            plt.legend()
            plt.savefig(os.path.join(dirname, "policy_loss.png"))
            plt.close()

        print("✅ Training complete!")