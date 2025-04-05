from client.sample_client import SampleClient

if __name__ == "__main__":
    is_player = input("Are you a player? (y/N): ").strip().lower()
    if is_player == 'y':
        room_id = input("Enter room ID (default: test-room): ") or "test-room"
        port = input("Enter server port (default: 5000): ") or 5000
        is_ai = input("Are you an AI? (y/N): ").strip().lower()
        if is_ai == 'y':
            player_name = input("Enter player name (default: player1): ") or "player1"
            client = SampleClient(int(port), room_id, player_name, is_ai=True)
            is_ai = True
        else:
            player_name = input("Enter player name (default: player1): ") or "player1"
            client = SampleClient(int(port), room_id, player_name, is_ai=False)
        client.connect()
        client.join_room()
        client.run()
    else:
        port = input("Enter server port (default: 5000): ") or 5000
        client = SampleClient(port=int(port))
        client.connect()
        client.observer()
        