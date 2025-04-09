from server.arena import Arena
if __name__ == "__main__":
      # -- Example usage --
    #
    # もし事前に定義したクライアントを渡したい場合:
    #
    from client.sample_arena_client import SampleClient as SampleClient
    from client.Takanori_Kotama import CFRClient as CFRClient

    predefs = [
        [CFRClient(player_name="PreAI1", is_ai=True), "PreAI1"],
        [SampleClient(player_name="PreAI2", is_ai=True), "PreAI2"],
        [SampleClient(player_name="PreAI2", is_ai=True), "PreAI3"],
        [SampleClient(player_name="PreAI2", is_ai=True), "PreAI4"],
        [SampleClient(player_name="PreAI2", is_ai=True), "PreAI5"]
    ]
    
    arena = Arena(total_matches=5, predefined_clients=predefs)
    arena.run()
    #
    
    # 事前クライアントを指定せずに起動
    
    # arena = Arena()
    # arena.run()