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
            if own == 0: reward = - initial * pf
            rewards.append(float(reward))
        return rewards

    def information_state_tensor(self, player=None):
        if player is None:
            player = self._cur_player
        # Initialize tensor: declaration + lives + hand info + remaining cards
        tensor = np.zeros(1 + _NUM_PLAYERS + (_NUM_PLAYERS - 1) * 4 + 15 + 1, dtype=np.float32)
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

        # 現在の相手のカードの合計値を入れる
        current_estimate = 0
        max_card = -100
        max_flag = False
        double_flag = False
        for i in range(_NUM_PLAYERS):
            if self._player_active[i] and i != player and self._cards[i] < 100:
                current_estimate += self._cards[i]
                if self._cards[i] > max_card:
                    max_card = self._cards[i]
            elif self._player_active[i] and i != player and self._cards[i] == 100:
                current_estimate += 0
                double_flag = True
            elif self._player_active[i] and i != player and self._cards[i] == 102:
                current_estimate += 0
                max_flag = True
        
        if max_flag:
            current_estimate -= max_card

        if double_flag:
            current_estimate *= 2

        tensor[1 + _NUM_PLAYERS + (_NUM_PLAYERS - 1)*4 + 15] = current_estimate
    
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

# === DQN setup ===
LEARNING_RATE=1e-3; GAMMA=0.99
BATCH_SIZE=64; MEMORY_SIZE=10000
TARGET_UPDATE_FREQ=1000; EPISODES=50000
EPS_START=1.0; EPS_END=0.1; EPS_DECAY=30000

class ReplayBuffer:
    def __init__(self, cap): self.buf=deque(maxlen=cap)
    def add(self,t): self.buf.append(t)
    def sample(self,n): return random.sample(self.buf,n)
    def __len__(self): return len(self.buf)

class QNetwork:
    def __init__(self,scope,sess,in_dim,out_dim):
        self.sess=sess
        with tf.variable_scope(scope):
            s_pl=tf.placeholder(tf.float32,[None,in_dim],"s")
            a_pl=tf.placeholder(tf.int32,[None],"a")
            t_pl=tf.placeholder(tf.float32,[None],"t")
            h1=tf.keras.layers.Dense(8,activation=tf.nn.relu)(s_pl)
            h2=tf.keras.layers.Dense(4,activation=tf.nn.relu)(h1)
            h3=tf.keras.layers.Dense(8,activation=tf.nn.relu)(h2)
            qv=tf.keras.layers.Dense(out_dim)(h3)
            oh=tf.one_hot(a_pl,out_dim)
            q_taken=tf.reduce_sum(qv*oh,axis=1)
            loss=tf.losses.mean_squared_error(t_pl,q_taken)
            op=tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)
            self.states=s_pl; self.actions=a_pl; self.targets=t_pl
            self.q_values=qv; self.train_op=op; self.loss=loss
            self.params=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope)
    def predict(self,ss): return self.sess.run(self.q_values,{self.states:ss})
    def update(self,s,a,t):_,l=self.sess.run([self.train_op,self.loss],{self.states:s,self.actions:a,self.targets:t});return l
    def get_params(self): return self.params
    def set_params(self,ps): ops=[tf.assign(v,p) for v,p in zip(self.params,ps)];self.sess.run(ops)

# === Training ===
with tf.Session() as sess:
    game=CoyoteGame(); in_dim=1+_NUM_PLAYERS+(_NUM_PLAYERS-1)*4+15+1; out_dim=2
    q_net=QNetwork('q_net',sess,in_dim,out_dim)
    target=QNetwork('target_net',sess,in_dim,out_dim)
    sess.run(tf.global_variables_initializer()); target.set_params(q_net.get_params())

    cd='checkpoints_dqn'; os.makedirs(cd,exist_ok=True)
    saver=tf.train.Saver()
    buf=ReplayBuffer(MEMORY_SIZE)
    eps=EPS_START; decay=(EPS_START-EPS_END)/EPS_DECAY
    step=0; loss_hist=[]
    pbar=trange(EPISODES,desc='Episode')
    for ep in pbar:
        state=game.new_initial_state(); tot_l=0; cnt=0
        while not state.is_terminal():
            p=state.current_player(); s=state.information_state_tensor(p); prev=list(state._player_lives)
            a = random.choice(state.legal_actions()) if random.random()<eps else int(np.argmax(q_net.predict([s])[0]))
            state.apply_action(a)
            nl=state._player_lives; own_delta=nl[p]-prev[p]
            opp_delta=(sum(prev)-prev[p])-(sum(nl)-nl[p]); r=own_delta+opp_delta
            done=state.is_terminal(); s_next=None if done else state.information_state_tensor(p)
            buf.add((s,a,r,s_next,done))
            if len(buf)>=BATCH_SIZE:
                batch=buf.sample(BATCH_SIZE); S,A,R,Sn,D=zip(*batch)
                qn=[0 if d or nxt is None else np.max(target.predict([nxt])[0]) for nxt,d in zip(Sn,D)]
                tgt=np.array(R)+GAMMA*np.array(qn)
                l=q_net.update(list(S),list(A),tgt); tot_l+=l; cnt+=1
                if eps>EPS_END: eps-=decay
                if step%TARGET_UPDATE_FREQ==0: target.set_params(q_net.get_params())
                step+=1
        avg=tot_l/cnt if cnt>0 else 0.0; loss_hist.append(avg)
        saver.save(sess,os.path.join(cd,f"ep.ckpt"))
        # plot
        plt.figure(); plt.plot(loss_hist); plt.xlabel('Episode'); plt.ylabel('Avg Loss'); plt.title('DQN Loss'); plt.yscale('log')

        plt.savefig(os.path.join(cd,f"loss_ep.png")); plt.close()
        pbar.set_postfix({'loss':f'{avg:.4f}','eps':f'{eps:.3f}'})
    print("Training complete!")
