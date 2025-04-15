from .not_websocket_client import Client
import numpy as np
import random
# from client.coyote_game import *

class CFRClient(Client):
    def AI_player_action(self, others_info, sum, log, actions):
        print(f"[CFRClient] AI deciding action based on sum: {sum}, log: {log}, actions: {actions}, others_info: {others_info}")
        return random.choice(actions)

# Statistical Class for evaluating the performance of the AI
class StatClient(Client):

    def AI_player_action(self, others_info, sum, log, actions):
        print(f"[StatClient] AI deciding action based on sum: {sum}, log: {log}, actions: {actions}, others_info: {others_info}")
        return random.choice(actions)