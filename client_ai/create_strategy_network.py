from StrategyNetwork import StrategyNetwork 
def create_strategy_network():
    """戦略ネットワークを作成"""
    # 入力サイズはゲーム状態に合わせて調整
    input_size = 318  # 適切なサイズに調整
    output_size = 141  # -1〜139の宣言値
    
    return StrategyNetwork(input_size, output_size)