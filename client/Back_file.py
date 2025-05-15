from .not_websocket_client import Client
from .Back.make_decision import make_decision
from .Back.StrategyNetwork import StrategyNetwork
from .Back.reservoirbuffer import ReservoirBuffer
import random
import logging
import coyote
import numpy as np
import os
from pathlib import Path
import tensorflow as tf

# ログファイルのパスを絶対パスで設定
log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "client.log")

# ロギングの設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)
logger.info("Client logger initialized")

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
        logging.info ("Deck shuffled.")
        random.shuffle(self.cards)

    def draw(self):
        if len(self.cards) > 0:
            return self.cards.pop()
        else:
            logging.info("No card left in the deck.")
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
    def __init__(self, player_name="player1", is_ai=False):
        super().__init__()
        self.hold_card = 0
        self.is_started = False
        self.action_start = False  
        self.is_ai = is_ai
        self.hold_card = 0
        self.player_name = player_name
        self.mycard = 0
        self.others_expect_sum = 0
        self.expect_cards = []
        self.deck = Deck()
        self.players_life = {} #{"player_name": life}
        self.previous_round_num = -1
        self.expect_sum = 1
        self.is_double_card = False #二倍するかどうか
        self.is_shuffle_card = True
        self.is_coyoted = False
        self.previous_round_num = 0# 直前のラウンド番号
        self.prev_others_life = [] # 直前の他プレイヤーのライフを保存する変数
        
        # モデル関連の初期化
        self.input_size = 317
        # モデル保存用ディレクトリの作成
        Path("models").mkdir(exist_ok=True)
        Path("save_picture").mkdir(exist_ok=True)
        self.strategy_net = StrategyNetwork(self.expect_sum,self.input_size)  # 必要な引数を指定
        self.policy_targets = np.random.random((self.input_size, 141))  # ランダムな正の値を生成
        self.advantage_buffer = ReservoirBuffer()
        self.strategy_buffer = ReservoirBuffer()
        
        # モデルのロード
        try:
            strategy_model_path = "models/strategy_model_317_1.keras"
            if os.path.exists(strategy_model_path):
                self.strategy_net.model = tf.keras.models.load_model(strategy_model_path)
                logging.info(f"Loaded existing model from {strategy_model_path}")
            else:
                logging.info("No existing model found")
        except Exception as e:
            logging.info(f"Error loading model: {e}")
            self.strategy_net = None      

    #場のカードの合計値を計算する
    def calc_card_sum(self, true_cards):
        card_sum = 0 #初期化
        for card in true_cards:
            card_sum += card
        if(self.is_double_card):
            card_sum *= 2
            self.is_double_card = False
        logging.info(f"gamesum is {card_sum}")
        return card_sum
    
    def convert_card(self, cards, Is_othersum, deck):
        logging.info (f"cards: {cards}")
        true_cards = sorted(cards, reverse = True)
        index = 0
        logging.info(f"Initial true_cards: {true_cards}")
        while index < len(true_cards):
            card = true_cards[index]
            logging.info(f"Card drawn: {card}")

            #?を引いたら次のカードを引き、出た番号のカードと交換する
            #全体の数の計算はラウンドにつき一回

            if(card == 103):
                if Is_othersum:
                    new_card = 0  #他プレイヤーの合計値を計算する場合
                else :
                    new_card = deck.draw()
                    #deck.cashed_cards.append(103) #103を山札に戻す
                #logging.info(f"Drawn new card: {deck.cashed_cards}")
                #logging.info(f"Drawn new card: {new_card}")
                if new_card != None: #103を引いた時にNoneがcardsに含まれていたから
                   true_cards[index] = new_card
                   true_cards = sorted(true_cards,reverse=True)
                   #もし特殊カードを引いてしまったら処理をもう一度行う
                   continue
                else:
                    self.deck.reset()
                    logging.info("No card left in the deck.")
                    continue

            #maxを引いたら、最も大きいカードを0にする
            elif(card == 102):
                normal_cards = [c for c in true_cards if c < 100] #通常カードを取得
                if len(normal_cards) != 0:
                    max_card = max(c for c in true_cards if c < 100) #最大値を取得
                    max_index = true_cards.index(max_card) #最大値のインデックスを取得
                    true_cards[max_index] = 0 #最大値を0にする
                true_cards[true_cards.index(102)] = 0

            #0(黒背景)を引いたら、ラウンド終了後山札をリセットする
            elif(card == 101):
                true_cards[index] = 0
                self.is_shuffle_card = True
            elif(card == 100):
                true_cards[index] = 0
                self.is_double_card = True

            index += 1

        return self.calc_card_sum(true_cards)   #関数の外に合計値を返す
           
    def AI_player_action(self,others_info, sum, log, player_card, actions, round_num):
        logger.info(f"[SampleClient] AI deciding action based on sum: {sum}, log: {log}, actions: {actions}, others_info: {others_info}, round_num: {round_num}")
        
        # モデルがロードされていない場合はランダムな行動を選択
        if self.strategy_net is None:
            logger.warning("Strategy network is not loaded, using random action")
            return actions[1]
            
        # 全員の手札を配列に格納する
        now_round_num = round_num
        if now_round_num != self.previous_round_num:
            self.previous_round_num = now_round_num

            #黒背景の０を引いたときに山札をリセットする
            if(self.is_shuffle_card):
                self.deck.reset()
                self.is_shuffle_card = False

            self.expect_cards = [player["card_info"] for player in others_info] # 他プレイヤーの手札を格納する
            for card in self.expect_cards:
                logger.info(f"deck {self.deck.cards}")
                logger.info(f"expect_cards {self.expect_cards}")
                if len(self.deck.cards) > 0:
                  if card in self.deck.cards:
                     self.deck.cards.remove(card) # 場のカードを山札から削除する
                  else:
                     self.deck.reset()
                     self.deck.cards.remove(card) # 場のカードを山札から削除する
                else:
                    self.deck.reset()

            self.mycard = self.deck.top_show_card() # 自分の手札を引く
            self.expect_cards.append(self.mycard) # 自分の手札を格納する
            sorted(self.expect_cards, reverse=True) # 降順にソートする
            self.expect_sum = self.convert_card(self.expect_cards, False, self.deck)  # 場のカードの合計値を計算する
            
            try:
                self.strategy_net.total_sum = self.expect_sum
                #logger.info(f"Updated strategy network total_sum to {self.expect_sum}")
            except Exception as e:
                #logger.error(f"Error updating strategy network total_sum: {e}")
                return actions[1]
        
        # 状態の準備
        current_state = {
            "others_info": others_info,
            "legal_action": actions,
            "log": log,
            "sum": self.expect_sum,
            "round_num": round_num,
            "player_card": self.mycard,
            "Is_coyoted": self.is_coyoted  # 必要に応じて設定
        }
        
        # try:
        logger.info(f"current_state: {current_state}")
        select_action = make_decision(current_state, self.strategy_net)
        logger.info(f"select_action: {select_action}")
        
        if(actions[1] + 5 <= select_action):
            select_action = actions[1] + 5

        if(self.expect_sum + 10 <= actions[1]):
            select_action = -1
        
        return select_action
        
        # except Exception as e:
        #     logger.error(f"Error during model prediction: {e}")
        #     return actions[1]

      
