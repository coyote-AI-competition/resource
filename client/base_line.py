from .client import Client
import random

class SampleClient(Client):
    def AI_player_action(self, others_info,sum, log, actions):
        # カスタムロジックを実装
        print(f"[SampleClient] AI deciding action based on sum: {sum}, log: {log}, actions: {actions}")
        # 例: ランダムにアクションを選択
        action = random.choice(actions)
        return action