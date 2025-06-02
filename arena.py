from server.arena import Arena
if __name__ == "__main__":
    # -- Example usage --
    #
    # もし事前に定義したクライアントを渡したい場合:
    #
    from client.Back_file import SampleClient as BackClient
    from client.yaduya_agent import PlayerReinforce as PlayerReinforce
    from client.MJ import ClientMJ as ClientMJ
    from client.Kuron import Kuron as ClientKuron
    from client.hamao_ni_melomelo import HamaoNiMeloMelo as ClientHamaoNiMeloMelo   
    from client.akazudayo import BaseLine as ClientBaseLine
    
    predefs = [
        
        [BackClient(player_name="PreAI1", is_ai=True), "PreAI1"],
        [PlayerReinforce(player_name="PreAI2", is_ai=True), "PreAI2"],
        [ClientMJ(player_name="PreAI3", is_ai=True), "PreAI3"],
        [ClientKuron(player_name="PreAI4", is_ai=True), "PreAI4"],
        [ClientHamaoNiMeloMelo(player_name="PreAI5", is_ai=True), "PreAI5"],
        [ClientBaseLine(player_name="PreAI6", is_ai=True), "PreAI6"],
    ]
    
    arena = Arena(total_matches=5, predefined_clients=predefs)
    arena.run()
    #
    
    # 事前クライアントを指定せずに起動
    
    # arena = Arena()
    # arena.run()