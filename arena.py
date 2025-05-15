from server.arena import Arena
if __name__ == "__main__":
    # -- Example usage --
    #
    # もし事前に定義したクライアントを渡したい場合:
    #
    from client.sample_arena_client import PlusOneClient as PlusOneClient
    from client.sample_arena_client import ConstClient as ConstClient
    from client.yaduya_agent import PlayerN as PlayerN
    
    predefs = [
        [PlusOneClient(player_name="PreAI1", is_ai=True), "PreAI1"],
        [PlayerN(player_name="PreAI2", is_ai=True), "PreAI2"],
        [PlusOneClient(player_name="PreAI3", is_ai=True), "PreAI3"],
        [PlayerN(player_name="PreAI4", is_ai=True), "PreAI4"],
        [PlayerN(player_name="PreAI5", is_ai=True), "PreAI5"],
        [PlayerN(player_name="PreAI6", is_ai=True), "PreAI6"],
    ]
    
    arena = Arena(total_matches=5, predefined_clients=predefs)
    arena.run()
    #
    
    # 事前クライアントを指定せずに起動
    
    # arena = Arena()
    # arena.run()