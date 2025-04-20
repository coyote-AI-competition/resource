from .not_websocket_client import Client
import random

class SampleClient(Client):
    def AI_player_action(self,others_info, sum, log, actions, round_num):
        # カスタムロジックを実装
        print(f"[SampleClient] AI deciding action based on sum: {sum}, log: {log}, actions: {actions},others_info: {others_info}, round_num: {round_num}" )
        # 例: ランダムにアクションを選択
        action = random.choice(actions)
        print(f"[SampleClient] AI selected action: {action}")
        return action