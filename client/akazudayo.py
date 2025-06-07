from .not_websocket_client import Client
class BaseLine(Client):
    def AI_player_action(self,sum,actions):
        if(len(actions) == 1):
            return actions[0]
        current_action = actions[1] - 1 
        if sum + 5 <= current_action:
            action = actions[0]
        else:
            action = actions[1]
        return action