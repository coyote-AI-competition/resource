import random
class ReservoirBuffer:
    def __init__(self, max_size=5000):#10000
        self.buffer = []
        self.max_size = max_size
        
    def add(self, item):
        if len(self.buffer) < self.max_size:
            self.buffer.append(item)# バッファが埋まってないとき追加
        else:
            # ランダム置換によるリザーバーサンプリング
            idx = random.randint(0, len(self.buffer) - 1)
            self.buffer[idx] = item
    
    def sample(self, batch_size):
        if len(self.buffer) < batch_size:
            return self.buffer # バッファ内のアイテムが要求数より少ない場合、全て返す
        return random.sample(self.buffer, batch_size) # バッファからランダムにbatch_size個取得