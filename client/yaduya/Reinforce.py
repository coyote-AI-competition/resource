from torch import optim
import torch 
import random
import numpy as np
import torch.nn as nn
# Unionは型ヒントのために必要
from typing import Union
import copy
from .replay import ReplayBuffer
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

class Net(torch.nn.Module):
    def __init__(self, state_size: int, action_size: int):
        super(Net, self).__init__()
        self.fc1 = torch.nn.Linear(state_size, 64)
        self.fc2 = torch.nn.Linear(64, 64)
        self.fc3 = torch.nn.Linear(64, action_size)

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = torch.relu(self.fc2(x))
        # シグモイド関数を使用して出力を0から1の範囲に制限
        x = torch.sigmoid(self.fc3(x))
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

        self.epsilon_start = 0.5
        self.epsilon_end = 0.0001
        self.epsilon_decay = (self.epsilon_start - self.epsilon_end) / TOTAL_TIMESTEPS
        self.epsilon = self.epsilon_start

        self.replay = ReplayBuffer(self.buffer_size, self.batch_size)
        self.data = None

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.count = 0
        self.original_qnet = Net(self.state_size, self.action_size).to(self.device)
        
        self.target_qnet = Net(self.state_size, self.action_size).to(self.device)
        
        self.sync_net()

        self.optimizer = optim.Adam(self.original_qnet.parameters(), self.lr)
    
    
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
            
    def update(self) -> None:
        if len(self.replay.buffer) < self.batch_size:
            return
            
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
        print('loss', loss.item())
        self.optimizer.step()
        
        
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
        torch.save(self.original_qnet.state_dict(), f'model_{model_name}.pth')