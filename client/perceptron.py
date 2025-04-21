import os
import pyspiel
import numpy as np
import coyote
import tensorflow.compat.v1 as tf
from collections import deque
import random
from tqdm import trange
import matplotlib.pyplot as plt

# Disable TF2 behavior
tf.disable_v2_behavior()

# === Coyote game implementation ===
_NUM_PLAYERS = 6
_NUM_DECLARATIONS = 121

_GAME_TYPE = pyspiel.GameType(
    short_name="python_coyote",
    long_name="Python Coyote",
    dynamics=pyspiel.GameType.Dynamics.SEQUENTIAL,
    chance_mode=pyspiel.GameType.ChanceMode.EXPLICIT_STOCHASTIC,
    information=pyspiel.GameType.Information.IMPERFECT_INFORMATION,
    utility=pyspiel.GameType.Utility.ZERO_SUM,
    reward_model=pyspiel.GameType.RewardModel.TERMINAL,
    max_num_players=_NUM_PLAYERS,
    min_num_players=2,
    provides_information_state_string=True,
    provides_information_state_tensor=True,
    provides_observation_string=True,
    provides_observation_tensor=False,
    parameter_specification={},
    default_loadable=False,
    provides_factored_observation_string=False
)

_GAME_INFO = pyspiel.GameInfo(
    num_distinct_actions=2,
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
        self.last_declarer = _NUM_PLAYERS - 1
        self.deck = coyote.game.Deck()
        self.deck.shuffle()
        self._cards = [self.deck.draw() for _ in range(_NUM_PLAYERS)]
        self._used_cards = []
        self._all_cards = [-10, -5, 0, 1, 2, 3, 4, 5, 10, 15, 20, 100, 101, 102, 103]
        # Special card flags
        self.is_double_card = False
        self.is_shuffle_card = False

    def calc_card_sum(self, cards):
        return coyote.game.calc_card_sum(self, cards)

    def current_player(self):
        return pyspiel.PlayerId.TERMINAL if self._is_terminal else self._cur_player

    def _legal_actions(self, player):
        return [0, 1]

    def _apply_action(self, action):
        if action == 1 or self.current_declaration >= _NUM_DECLARATIONS - 1:
            # Challenge logic
            true_total = coyote.game.convert_card(self, self._cards, False, self.deck)
            # Record used cards (excluding challenge initiator)
            for i in range(_NUM_PLAYERS):
                if self._player_active[i] and i != self._cur_player:
                    card = self._cards[i]
                    if card in self._all_cards:
                        self._used_cards.append(card)
            # Determine loser
            if self.current_declaration > true_total:
                loser = self.last_declarer
            else:
                loser = self._cur_player
            # Update lives
            self._player_lives[loser] -= 1
            if self._player_lives[loser] <= 0:
                self._player_active[loser] = False
            self._is_terminal = sum(self._player_active) <= 1
            self.current_declaration = 0
            # Draw new cards, reshuffle if needed
            if hasattr(self.deck, 'reset'):  # shuffle card logic if implemented
                if len(self.deck.cards) == 0:
                    self.deck.reset()
                    self._used_cards.clear()
            for i in range(_NUM_PLAYERS):
                if self._player_active[i]:
                    if len(self.deck.cards) == 0:
                        self.deck.reset()
                        self._used_cards.clear()
                    self._cards[i] = self.deck.draw()
                else:
                    self._cards[i] = 0
        else:
            # Declaration logic
            self.current_declaration += 1
            self.last_declarer = self._cur_player
        # Advance to next active player
        self._cur_player = (self._cur_player + 1) % _NUM_PLAYERS
        while not self._player_active[self._cur_player]:
            self._cur_player = (self._cur_player + 1) % _NUM_PLAYERS
        self._cur_player = (self._cur_player + 1) % _NUM_PLAYERS
        while not self._player_active[self._cur_player]:
            self._cur_player = (self._cur_player + 1) % _NUM_PLAYERS

    def is_terminal(self): return self._is_terminal

    def returns(self):
        initial = 3
        pf = 2.0
        rewards = []
        for i in range(_NUM_PLAYERS):
            own = self._player_lives[i]
            opp_lost = sum((initial - self._player_lives[j]) for j in range(_NUM_PLAYERS) if j != i)
            lost = initial - own
            reward = opp_lost - lost * pf
            if own == 0:
                reward = - initial * pf
            rewards.append(float(reward))
        return rewards

    def information_state_tensor(self, player=None):
        if player is None:
            player = self._cur_player
        # Initialize tensor: declaration + lives + hand info + remaining cards
        tensor = np.zeros(1 + _NUM_PLAYERS + (_NUM_PLAYERS - 1) * 4 + 15, dtype=np.float32)
        # Current declaration
        tensor[0] = self.current_declaration
        # Player lives
        for i in range(_NUM_PLAYERS):
            tensor[1 + i] = float(self._player_lives[(i + player) % _NUM_PLAYERS])
        off = 1 + _NUM_PLAYERS
        # Hand cards: value slot for <100, one-hot for special cards
        for i in range(_NUM_PLAYERS - 1):
            card = self._cards[(i + player + 1) % _NUM_PLAYERS]
            base = off + i * 4
            if card < 100:
                tensor[base] = float(card)
            elif card == 102:
                tensor[base + 1] = 1.0
            elif card == 100:
                tensor[base + 2] = 1.0
            elif card == 103:
                tensor[base + 3] = 1.0
        # Remaining cards count
        cards_kind = [-10, -5, 0, 1, 2, 3, 4, 5, 10, 15, 20, 100, 101, 102, 103]
        visible = list(self._all_cards)
        for c in self._used_cards:
            if c in visible:
                visible.remove(c)
        for idx, active in enumerate(self._player_active):
            if active and idx != player:
                c = self._cards[idx]
                if c in visible:
                    visible.remove(c)
        for j, kind in enumerate(cards_kind):
            tensor[off + (_NUM_PLAYERS - 1) * 4 + j] = float(visible.count(kind))
        return tensor

    def information_state_tensor(self, player=None):
        if player is None:
            player = self._cur_player
        # Initialize tensor: declaration + lives + hand info + remaining cards
        tensor = np.zeros(1 + _NUM_PLAYERS + (_NUM_PLAYERS - 1) * 4 + 15, dtype=np.float32)
        # Current declaration
        tensor[0] = self.current_declaration
        # Player lives
        for i in range(_NUM_PLAYERS):
            tensor[1 + i] = float(self._player_lives[(i + player) % _NUM_PLAYERS])
        off = 1 + _NUM_PLAYERS
        # Hand cards: value slot for <100, one-hot for special cards
        for i in range(_NUM_PLAYERS - 1):
            card = self._cards[(i + player + 1) % _NUM_PLAYERS]
            base = off + i * 4
            if card < 100:
                tensor[base] = float(card)
            elif card == 102:
                tensor[base + 1] = 1.0
            elif card == 100:
                tensor[base + 2] = 1.0
            elif card == 103:
                tensor[base + 3] = 1.0
            # 101: leave all zeros
        # Remaining cards count
        cards_kind = [-10, -5, 0, 1, 2, 3, 4, 5, 10, 15, 20, 100, 101, 102, 103]
        visible = list(self._all_cards)
        # remove used cards
        for c in self._used_cards:
            if c in visible:
                visible.remove(c)
        # remove other players' known cards
        for idx, active in enumerate(self._player_active):
            if active and idx != player:
                c = self._cards[idx]
                if c in visible:
                    visible.remove(c)
        for j, kind in enumerate(cards_kind):
            tensor[off + (_NUM_PLAYERS - 1) * 4 + j] = float(visible.count(kind))
        return tensor

class CoyoteObserver:
    def __init__(self, params=None): self.params = params or {}
    def set_from(self, state, player): pass
    def string_from(self, state, player): return ""

class CoyoteGame(pyspiel.Game):
    def __init__(self, params=None): super().__init__(_GAME_TYPE, _GAME_INFO, params or {})
    def new_initial_state(self): return CoyoteState(self)
    def make_py_observer(self, iig_obs_type=None, params=None): return CoyoteObserver(params)

pyspiel.register_game(_GAME_TYPE, CoyoteGame)

# === Supervised learning on declaration success/failure ===
# Remove Q-Network and perform MLP-based classification

# Hyperparameters for supervised training
data_episodes = 10000  # total transitions to collect
sim_rounds = 1000        # number of simulation-training cycles
train_epochs = 1000
batch_size = 64
learning_rate = 1e-3

# Prepare checkpoint directory and saver
checkpoint_dir = 'checkpoints_perceptron'
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

with tf.Session() as sess:
    # Build multilayer perceptron
    in_dim = 1 + _NUM_PLAYERS + (_NUM_PLAYERS - 1) * 4 + 15
    X = tf.placeholder(tf.float32, [None, in_dim], name="X")
    Y = tf.placeholder(tf.float32, [None], name="Y")
    hidden1 = tf.keras.layers.Dense(128, activation=tf.nn.relu)(X)
    hidden2 = tf.keras.layers.Dense(64, activation=tf.nn.relu)(hidden1)
    logits = tf.keras.layers.Dense(1)(hidden2)
    preds = tf.sigmoid(logits)
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.reshape(Y, [-1,1]), logits=logits))
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    saver = tf.train.Saver()

    sess.run(tf.global_variables_initializer())

    loss_history = []
    episodes_per_round = data_episodes // sim_rounds
    from tqdm import trange

    # Training rounds with tqdm visualization
    for rnd in trange(sim_rounds, desc='Simulation-Round'):
        # Collect training data for this round
        states = []
        labels_list = []
        for _ in range(episodes_per_round):
            state = CoyoteGame().new_initial_state()
            while not state.is_terminal():
                player = state.current_player()
                if player == 0:
                    s = state.information_state_tensor(0)
                    declared = state.current_declaration + 1
                    true_total = coyote.game.convert_card(state, state._cards, False, state.deck)
                    label = 1.0 if declared <= true_total else 0.0
                    states.append(s)
                    labels_list.append(label)
                    state._apply_action(0)
                else:
                    action = random.choice(state.legal_actions())
                    state._apply_action(action)
        # Prepare numpy arrays
        data_X = np.array(states, dtype=np.float32)
        data_Y = np.array(labels_list, dtype=np.float32)
        num_samples = data_X.shape[0]
        # Train on collected data
        for epoch in range(train_epochs):
            perm = np.random.permutation(num_samples)
            X_sh, Y_sh = data_X[perm], data_Y[perm]
            for i in range(0, num_samples, batch_size):
                batch_X = X_sh[i:i+batch_size]
                batch_Y = Y_sh[i:i+batch_size]
                sess.run(optimizer, feed_dict={X: batch_X, Y: batch_Y})
        # Compute loss for this round
        epoch_loss = sess.run(loss, feed_dict={X: data_X, Y: data_Y})
        loss_history.append(epoch_loss)
        print(f"Round {rnd+1}/{sim_rounds}, Loss: {epoch_loss:.4f}")
        # Plot loss history up to current round
        plt.figure()
        plt.plot(range(1, len(loss_history)+1), loss_history)
        plt.xlabel('Round')
        plt.ylabel('Loss')
        plt.title('Supervised Training Loss per Round')
        plt_path = os.path.join(checkpoint_dir, f'loss_round.png')
        plt.savefig(plt_path)
        plt.close()
        # Save checkpoint per round
        ckpt_path = os.path.join(checkpoint_dir, f'round.ckpt')
        saver.save(sess, ckpt_path)
    print("Supervised training complete!")
