from server.arena import Arena
from client.sample_arena_client import SampleClient as SampleClient
from client.not_websocket_client import Client
import StrategyNetwork as sn
import encode_state as es

import random

class SampleClient(Client):
    def AI_player_action(self,others_info, sum, log, actions):
        # カスタムロジックを実装
        print(f"[SampleClient] AI deciding action based on sum: {sum}, log: {log}, actions: {actions},others_info: {others_info}")
        # 例: ランダムにアクションを選択

        return chosen_action



if __name__ == "__main__":

    predefs = [
        [SampleClient(player_name="PreAI1", is_ai=True), "PreAI1"],
        [SampleClient(player_name="PreAI2", is_ai=True), "PreAI2"],
        [SampleClient(player_name="PreAI3", is_ai=True), "PreAI3"],
        [SampleClient(player_name="PreAI4", is_ai=True), "PreAI4"],
        [SampleClient(player_name="PreAI5", is_ai=True), "PreAI5"]
    ]

    arena = Arena(total_matches=5, predefined_clients=predefs)
    arena.run()