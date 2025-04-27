from StrategyNetwork import StrategyNetwork 
def create_strategy_network():
    """戦略ネットワークを作成"""
    # 入力サイズはゲーム状態に合わせて調整
    input_size = 304  # 適切なサイズに調整
    output_size = 141  # 0〜140の宣言値
    
    return StrategyNetwork(input_size, output_size)