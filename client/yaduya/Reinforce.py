from torch import optim
import torch 
from schedulefree import RAdamScheduleFree
import random
import numpy as np
import torch.nn as nn
# Unionは型ヒントのために必要
from typing import Union
import copy
from .replay import ReplayBuffer
import logging

TOTAL_TIMESTEPS = 50000 # 総ステップ数
MAX_STEP = 500 # 1エピソードでの最大ステップ数
BUFFER_SIZE = 1000000 #バッファサイズ
BATCH_SIZE = 32 # バッチサイズ
LEARNING_RATE = 0.0001 # 学習率
DISCOUNT_RATE = 0.99 # 割引率
STATE_SIZE = 4 # 状態数
ACTION_SIZE = 2 # 行動数
TARGET_UPDATE_STEPS = 1000 # ターゲットネットワークの更新ステップ頻度
LOG_STEPS = 5000 # ログ出力のステップ頻度

# pathの指定
path = 'cAc/model_PreAI6_pre.pth'


# ログの設定
# ログの設定
logging.basicConfig(
  filename='logs/agent.log',   # ログファイルの名前
  level=logging.DEBUG,      # ログレベル（DEBUG以上のレベルが記録される）
)
class Net(torch.nn.Module):
    def __init__(self, state_size: int, action_size: int,hidden_size: int = 128):
        super(Net, self).__init__()
        self.fc1 = torch.nn.Linear(state_size, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, hidden_size)
        self.fc3 = torch.nn.Linear(hidden_size, action_size)

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = torch.relu(self.fc2(x))
        # シグモイド関数を使用して出力を0から1の範囲に制限
        x = torch.sigmoid(self.fc3(x))
        return x


class SelfAttention(nn.Module):
    def __init__(self,state_size: int, action_size: int,hidden_size: int = 128):
        super(SelfAttention, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
        self.attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=4,dropout=0.1)
        self.norm = nn.LayerNorm(hidden_size)
    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = x.unsqueeze(0) 
        atten_out, _ = self.attention(x, x, x)
        atten_out = self.norm(atten_out + x)
        atten_out = atten_out.squeeze(0)
        x = torch.sigmoid(self.fc3(atten_out))
        return x


class Agent:
    def __init__(self, state_size: int, action_size: int):
        self.state_size = state_size
        self.action_size = action_size

        self.lr = LEARNING_RATE
        self.gamma = DISCOUNT_RATE
        self.buffer_size = BUFFER_SIZE
        self.batch_size = BATCH_SIZE
        self.target_update = TARGET_UPDATE_STEPS

        self.epsilon_start = 0.05
        self.epsilon_end = 0.0001
        self.epsilon_decay = (self.epsilon_start - self.epsilon_end) / TOTAL_TIMESTEPS
        self.epsilon = self.epsilon_start

        self.replay = ReplayBuffer(self.buffer_size, self.batch_size)
        self.data = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.count = 0
        path = '/home/vyuma/dev/cAc/models/model_PreAI6_128_313.pth'
        self.original_qnet = Net(self.state_size, self.action_size).to(self.device)
        self.original_qnet.load_state_dict(torch.load(path))
        self.target_qnet = Net(self.state_size, self.action_size).to(self.device)
        self.target_qnet.load_state_dict(torch.load(path))
        self.sync_net()
        self.optimizer = RAdamScheduleFree(self.original_qnet.parameters(), self.lr)
        self.losses = []
    
    def get_action(self, state) -> int:
        print('self.epsilon', self.epsilon)
        if np.random.rand() <= self.epsilon:
            # ちょっとした細工をする
            if state[0] == 0:
                return 0
            elif state[0] == 1:
                return 1
            else:
                return random.randrange(self.action_size)
        else:
            state_np = np.array(state, dtype=np.float32)
            state = torch.tensor(state_np[np.newaxis, :].astype(np.float32), device=self.device)
            q_c = self.original_qnet(state)
            return q_c.detach().argmax().item()
    
    def get_action_static(self,state) -> int:
        state_np = np.array(state, dtype=np.float32)
        state = torch.tensor(state_np[np.newaxis, :].astype(np.float32), device=self.device)
        # evalモードにする
        self.original_qnet.eval()
        self.optimizer.eval()
        # 予測を行う
        q_c = self.original_qnet(state)
        return q_c.detach().argmax().item()
    
    
    def update(self) -> None:
        if len(self.replay.buffer) < self.batch_size:
            return
        self.original_qnet.train()
        self.optimizer.train()
        self.data = self.replay.get()
        
        q_c = self.original_qnet(self.data.state)
        q = q_c[np.arange(self.batch_size), self.data.action.cpu().numpy()]

        with torch.no_grad():
            next_q_c = self.target_qnet(self.data.next_state)
            next_q = next_q_c.max(1)[0]
            next_q.detach()
            target = self.data.reward + (1 - self.data.done) * self.gamma *  next_q

        loss_function = nn.MSELoss()
        loss = loss_function(q, target)
        self.optimizer.zero_grad()
        loss.backward()
        logging.debug(f"loss: {loss.item()}")
        self.losses.append(loss.item())
        self.optimizer.step()
    def plot_loss(self,count):
        import matplotlib.pyplot as plt
        plt.plot(self.losses)
        plt.xlabel('Episode')
        plt.ylabel('Loss')
        plt.title('Loss over Episodes')
        plt.savefig(f'figure/losses/loss_plot-{count}.png')
    
    def add_experience(
        self,
        state: np.ndarray,
        action: int,
        reward: Union[int, float],
        next_state: np.ndarray,
        done: bool,
        is_next: bool = False,
        ) -> None:
        self.replay.add(state, action, reward, next_state, done,is_next)
        
    def sync_net(self) -> None:
        self.target_qnet = copy.deepcopy(self.original_qnet)
        
    def set_epsilon(self) -> None:
        self.epsilon -= self.epsilon_decay
        
    def save_model(self,model_name) -> None:
        torch.save(self.original_qnet.state_dict(), f'models/model_{model_name}_128.pth')