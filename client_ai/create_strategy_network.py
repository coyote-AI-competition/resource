from StrategyNetwork import StrategyNetwork 
def create_strategy_network(self):
    """戦略ネットワークを作成"""
    # 入力サイズはゲーム状態に合わせて調整
    input_size = self.input_size  # 適切なサイズに調整
    output_size = 141  # -1〜139の宣言値
    
    return StrategyNetwork(self.total_sum, input_size, output_size)