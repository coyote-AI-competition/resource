import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from collections import defaultdict
import random
import os
import json
from tqdm import tqdm

# 既存のコードからインポート
from .encode_state import encode_state
from .StrategyNetwork import StrategyNetwork
from .reservoirbuffer import ReservoirBuffer
from .create_advantage_network import create_advantage_network

# 学習評価のためのクラス
class CFRTrainingEvaluator:
    def __init__(self, strategy_net, advantage_net):
        """学習の評価を行うクラス"""
        self.strategy_net = strategy_net
        self.advantage_net = advantage_net
        self.current_state = []  # 現在の状態を保持
        self.history = {
            'advantage_loss': [],
            'strategy_accuracy': [],
            'declaration_vs_sum_ratio': [],
            'epoch': [],
            'declarations': [],  # 宣言値の履歴
            'actual_sums': [],  # 実際の合計値の履歴
            'over_declaration_ratios': []  # 超過宣言の割合
        }
    
    def log_metrics(self, epoch, advantage_loss=None, strategy_accuracy=None, 
                   declaration_vs_sum_ratio=None):
        """各指標をログに記録"""
        self.history['epoch'].append(epoch)
        
        if advantage_loss is not None:
            self.history['advantage_loss'].append(advantage_loss)
        
        if strategy_accuracy is not None:
            self.history['strategy_accuracy'].append(strategy_accuracy)
        
        if declaration_vs_sum_ratio is not None:
            self.history['declaration_vs_sum_ratio'].append(declaration_vs_sum_ratio)
            
    
    def update_current_state(self, state):
        """現在の状態を更新"""
        self.current_state = state
        
        # 履歴の更新
        declarations = [s['selectaction'] for s in state]
        actual_sums = [s['sum'] for s in state]
        
        # 超過宣言の割合を計算
        over_declarations = []
        for d, s in zip(declarations, actual_sums):
            if d > s:
                over_declarations.append((d - s) / s)
            else:
                over_declarations.append(0)
        
        self.history['declarations'].append(declarations)
        self.history['actual_sums'].append(actual_sums)
        self.history['over_declaration_ratios'].append(np.mean(over_declarations))
    
    def evaluate_declaration_accuracy(self, test_states, num_samples=100):
        """宣言値と実際の合計値の比較を評価"""
        correct_declarations = 0
        total_ratio = 0
        overestimation_count = 0
        
        # テスト状態から最大numSamplesをランダムに選択
        sample_size = min(num_samples, len(test_states))
        sampled_states = random.sample(test_states, sample_size)
        
        for state in sampled_states:

            #declaration = max(action_probs.items(), key=lambda x: x[1])[0]
            declaration = state['selectaction']
            # 実際の合計値
            actual_sum = state['sum']
            
            # 宣言値と実際の合計値の比率
            if actual_sum > 0:
                ratio = declaration / actual_sum
                total_ratio += ratio
                
                # 過大宣言しているか確認
                if declaration > actual_sum:
                    overestimation_count += 1
                
                # 宣言値が実際の合計±20%以内なら正確と見なす
                if 0.8 <= ratio <= 1.2:
                    correct_declarations += 1
        
        avg_ratio = total_ratio / sample_size if sample_size > 0 else 0
        accuracy = correct_declarations / sample_size if sample_size > 0 else 0
        overestimation_rate = overestimation_count / sample_size if sample_size > 0 else 0
        
        return avg_ratio, accuracy, overestimation_rate
    
    def plot_all_metrics(self):
        """ゲーム全体を通しての指標の可視化"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. 損失関数の推移（ゲームごと）
        if self.history['advantage_loss']:
            axes[0, 0].plot(range(len(self.history['advantage_loss'])), self.history['advantage_loss'], 'b-')
            axes[0, 0].set_title('Loss per Game')
            axes[0, 0].set_xlabel('Game Number')
            axes[0, 0].set_ylabel('Loss Value')
            axes[0, 0].grid(True)
        
        # 2. 場の合計を超えた宣言の割合（ゲームごと）
        if self.current_state:
            declarations = [s['selectaction'] for s in self.current_state]
            actual_sums = [s['sum'] for s in self.current_state]
            over_declarations = []
            
            # ゲームごとの超過宣言の割合を計算
            for d, s in zip(declarations, actual_sums):
                if d > s:
                    over_declarations.append(1)
                else:
                    over_declarations.append(0)
            
            # 移動平均を計算（10ゲーム単位）
            window_size = min(10, len(over_declarations))
            if window_size > 0:
                moving_avg = np.convolve(over_declarations, 
                                       np.ones(window_size)/window_size, 
                                       mode='valid')
                axes[0, 1].plot(range(len(moving_avg)), moving_avg, 'r-')
                axes[0, 1].set_title('Over-Declaration Rate (10-game Moving Average)')
                axes[0, 1].set_xlabel('Game Number')
                axes[0, 1].set_ylabel('Rate of Over-Declaration')
                axes[0, 1].set_ylim([0, 1])
                axes[0, 1].grid(True)
        
        # 3. 宣言値と実際の合計値の推移
        if self.current_state:
            game_numbers = range(len(declarations))
            axes[1, 0].plot(game_numbers, declarations, 'b.', label='Declared Value', alpha=0.5)
            axes[1, 0].plot(game_numbers, actual_sums, 'r.', label='Actual Sum', alpha=0.5)
            
            # トレンドラインを追加
            z_decl = np.polyfit(game_numbers, declarations, 1)
            z_sum = np.polyfit(game_numbers, actual_sums, 1)
            p_decl = np.poly1d(z_decl)
            p_sum = np.poly1d(z_sum)
            
            axes[1, 0].plot(game_numbers, p_decl(game_numbers), 'b-', label='Declaration Trend')
            axes[1, 0].plot(game_numbers, p_sum(game_numbers), 'r-', label='Sum Trend')
            axes[1, 0].set_title('Declaration vs Actual Sum Over Games')
            axes[1, 0].set_xlabel('Game Number')
            axes[1, 0].set_ylabel('Value')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
        
        # 4. 宣言値と実際の合計値の差分の分布
        if self.current_state:
            differences = [d - s for d, s in zip(declarations, actual_sums)]
            axes[1, 1].hist(differences, bins=30, alpha=0.7, color='purple')
            axes[1, 1].axvline(x=0, color='r', linestyle='--', label='No Difference')
            axes[1, 1].set_title('Distribution of Declaration - Actual Sum')
            axes[1, 1].set_xlabel('Difference (Declaration - Actual Sum)')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        return fig
    
    def plot_declaration_distribution(self, test_states, num_samples=100):
        """宣言値の分布と実際の合計値の分布を比較"""
        declarations = []
        actual_sums = []
        
        # テスト状態からサンプリング
        sample_size = min(num_samples, len(test_states))
        sampled_states = random.sample(test_states, sample_size)
        
        for state in sampled_states:
            # 状態をエンコード
            encoded_state = encode_state(state)
            
            # 戦略ネットワークで宣言値を予測
            action_probs = self.strategy_net.prediction(encoded_state, state['legal_action'])
            
            # 確率に基づいて宣言値をサンプリング
            actions = list(action_probs.keys())
            probs = list(action_probs.values())
            
            if actions and probs:
                declaration = np.random.choice(actions, p=np.array(probs)/sum(probs))
                declarations.append(declaration)
                actual_sums.append(state['sum'])
        
        # 結果をプロット
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # ヒストグラム
        ax.hist(actual_sums, bins=20, alpha=0.5, label='Actual Sum', color='blue')
        ax.hist(declarations, bins=20, alpha=0.5, label='Declarations', color='red')
        
        # 過大宣言の割合を計算
        overestimations = sum(1 for d, s in zip(declarations, actual_sums) if d > s)
        overestimation_rate = overestimations / len(declarations) if declarations else 0
        
        ax.set_title(f'Declaration vs Actual Sum Distribution\nOverestimation Rate: {overestimation_rate:.2f}')
        ax.set_xlabel('Value')
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.grid(True)
        
        return fig, overestimation_rate

# メイン実行関数
def evaluate_cfr_training(self,current_state,iterations=50):
    """CFR学習の評価を実行"""
    # ネットワークの作成
    input_size = self.input_size
    output_size = 141
    advantage_net = self.advantage_net
    strategy_net = self.strategy_net
    
    # 評価用のインスタンス
    evaluator = CFRTrainingEvaluator(strategy_net, advantage_net)

    # 学習と評価のループ
    for i in tqdm(range(iterations)):
        # サンプル状態からのバッチ作成
        batch_size = 32
        batch_indices = np.random.choice(len(current_state), batch_size)
        batch_states = [current_state[idx] for idx in batch_indices]
        
        # エンコード
        encoded_states = np.array([encode_state(state) for state in batch_states])
        
        # 形状を(None, 318)に調整
        if len(encoded_states.shape) == 3:
            encoded_states = encoded_states.reshape(-1, self.input_size)  # (32, 1, 318) → (32, 318)
        
        # アドバンテージネットワークの更新シミュレーション
        advantage_loss = advantage_net.evaluate(
            encoded_states, 
            self.policy_targets,  
            verbose=0
        )[0]
        
        # 戦略ネットワークの精度計算
        correct_predictions = 0
        total_predictions = len(batch_states)
        
        for state, encoded_state in zip(batch_states, encoded_states):
            # 実際に選択されたアクション
            actual_action = state['selectaction']
            
            # ネットワークの予測を取得
            action_probs = strategy_net.prediction(encoded_state, state['legal_action'])
            
            # 最も確率の高いアクションを選択
            # 選択された値が
            predicted_action = max(action_probs.items(), key=lambda x: x[1])[0]
            if predicted_action == 0 and actual_action == -1:  # コヨーテの場合
                correct_predictions += 1
            elif predicted_action == actual_action + 1:  # 通常アクションの場合
                correct_predictions += 1
        
        strategy_accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        
        # 宣言値と実際の合計の比率を評価
        declaration_ratio, _, overestimation_rate = evaluator.evaluate_declaration_accuracy(current_state)
        
        # 指標の記録
        evaluator.log_metrics(
            epoch=i,
            advantage_loss=advantage_loss,
            strategy_accuracy=strategy_accuracy,
            declaration_vs_sum_ratio=declaration_ratio,
        )
    
    # 結果の可視化
    metrics_fig = evaluator.plot_all_metrics()
    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'save_picture')
    os.makedirs(save_dir, exist_ok=True)
    metrics_fig.savefig(os.path.join(save_dir, 'cfr_training_metrics.png'))
    
    # 宣言値の分布を可視化
    dist_fig, overestimation_rate = evaluator.plot_declaration_distribution(current_state)
    dist_fig.savefig(os.path.join(save_dir, 'declaration_distribution.png'))
    
    print(f"Training evaluation complete. Final overestimation rate: {overestimation_rate:.2f}")
    return evaluator

# モデルの推論パフォーマンスを視覚化する関数
def visualize_model_prediction(strategy_net, test_states, num_samples=10):
    """モデルの推論結果を視覚化"""
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.flatten()
    
    # ランダムにサンプルを選択
    sample_size = min(num_samples, len(test_states))
    sampled_states = random.sample(test_states, sample_size)
    
    for i, state in enumerate(sampled_states):
        if i >= len(axes):
            break
            
        # 状態をエンコード
        encoded_state = encode_state(state)
        
        # 戦略ネットワークで宣言値を予測
        action_probs = strategy_net.prediction(encoded_state, state['legal_action'])
        
        # 実際の合計値
        actual_sum = state['sum']
        
        # 行動確率の上位5つをプロット
        sorted_actions = sorted(action_probs.items(), key=lambda x: x[1], reverse=True)[:5]
        actions = [a[0] for a in sorted_actions]
        probs = [a[1] for a in sorted_actions]
        
        axes[i].bar(actions, probs)
        axes[i].axvline(x=actual_sum, color='r', linestyle='--', label=f'Actual Sum: {actual_sum}')
        axes[i].set_title(f'State {i+1}')
        axes[i].set_xlabel('Declaration Value')
        axes[i].set_ylabel('Probability')
        axes[i].legend()
    
    plt.tight_layout()
    return fig

