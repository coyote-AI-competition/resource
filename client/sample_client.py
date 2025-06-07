from .client import Client
import random

class SampleClient(Client):
    def __init__(self,port=5000, room_id="test-room", player_name="player1", is_ai=False):
        super().__init__()
        
        
        
    def AI_player_action(self, others_info ,sum, log, actions):
        # カスタムロジックを実装
        print(f"[Yuma] AI deciding action based on sum: {sum}, log: {log}, actions: {actions}, others_info: {others_info}")
        
        # 例: ランダムにアクションを選択
        action = random.choice(actions)
        return action