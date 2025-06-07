from client.sample_client import SampleClient
from client.akazudayo import BaseLine as BaseLine
import threading
import time
import logging

# ログ設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_client(client, name):
    """クライアントを実行するスレッド関数"""
    try:
        logger.info(f"[{name}] 接続中...")
        client.connect()
        time.sleep(1)  # 接続の確立を待つ
        
        logger.info(f"[{name}] ルーム参加中...")
        client.join_room()
        time.sleep(1)  # ルーム参加の処理を待つ
        
        logger.info(f"[{name}] 実行開始")
        client.run()
    except Exception as e:
        logger.error(f"[{name}] エラー発生: {e}", exc_info=True)

if __name__ == "__main__":
    is_player = input("Are you a player? (y/N): ").strip().lower()
    if is_player == 'y':
        room_id = input("Enter room ID (default: test-room): ") or "test-room"
        port = input("Enter server port (default: 5000): ") or 5000
        is_ai = input("Are you an AI? (y/N): ").strip().lower()
        
        if is_ai == 'y':
            # AI数を選択
            ai_count = input("How many AI clients? (1-4, default: 4): ")
            ai_count = int(ai_count) if ai_count.isdigit() else 4
            ai_count = min(4, max(1, ai_count))
            
            # クライアント設定
            clients = []
            for i in range(ai_count):
                client_name = f"SampleAI{i}"
                clients.append((BaseLine(int(port), room_id, client_name, is_ai=True), client_name))
            
            # スレッド作成と実行
            threads = []
            for client, name in clients:
                thread = threading.Thread(target=run_client, args=(client, name))
                thread.daemon = True  # メインプログラム終了時にスレッドも終了
                threads.append(thread)
            
            # スレッドの開始（少し間隔をあける）
            for thread in threads:
                thread.start()
                time.sleep(2)  # クライアント間の接続タイミングをずらす
                
            # スレッドの終了を待機（Ctrl+Cで中断可能）
            try:
                for thread in threads:
                    thread.join()
            except KeyboardInterrupt:
                logger.info("プログラムが中断されました")
                
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