from .not_websocket_client import Client
import random


class RLClient(Client):
    def __init__(self, player_name="player1", is_ai=False):
        super().__init__(player_name, is_ai)
    
    def AI_player_action(self,others_info, sum, log, actions, round_num):
        # カスタムロジックを実装
        print(f"[Yuma] AI deciding action based on sum: {sum}, " )
        print(f"log: {log},")
        print(f"actions: {actions}")
        print(f"others_info: {others_info}")
        print(f"round_num: {round_num}")
        
        action = random.choice(actions[:5])
        return action







