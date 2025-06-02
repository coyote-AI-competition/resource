from .hamao_melo_library.main import MyLibrary
from .not_websocket_client import Client


class HamaoNiMeloMelo(Client):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.my_library = MyLibrary()

    def AI_player_action(self, others_info, sum, log, actions, round_num):
        try:
            return self.my_library.action(others_info, log, actions)

        except Exception as e:
            print(f"Error in AI_player_action: {e}")
            print(others_info)
            if len(actions) == 1:
                return -1
            else:
                return actions[1]
