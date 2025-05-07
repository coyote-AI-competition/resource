import pyspiel
import numpy as np
from open_spiel.python.algorithms import external_sampling_mccfr
from open_spiel.python.algorithms import get_all_states
from open_spiel.python.algorithms import deep_cfr
import random
import pickle
import time
import tensorflow
import os
import tqdm
import matplotlib.pyplot as plt
import numpy as np
from pyspiel import PlayerId

import tensorflow.compat.v1 as tf
from game_setting import *


_NUM_PLAYERS = 6  # プレイヤー数

if __name__ == "__main__":

    # cfrで数理モデルを作成
    # Check for TensorFlow GPU access

    game = CoyoteGame() # ゲームのインスタンスを作成

    dirname = os.path.dirname(__file__)

    # while True:
    # 保存先パス
    ckpt_path = os.path.join(dirname, f"checkpoints_{_NUM_PLAYERS}", "deep_cfr.ckpt")
    # for _ in tqdm.tqdm(range(100000)):
    #     # セッション開始
    with tf.Session() as sess:

        # DeepCFRSolver 構築
        solver = deep_cfr.DeepCFRSolver(
            game=game,
            session=sess,
            policy_network_layers=[256, 256],
            advantage_network_layers=[256, 256],
            num_iterations=5,
            num_traversals=1000,
            learning_rate=1e-4,
            batch_size_advantage=32,
            batch_size_strategy=32,
            memory_capacity=1e6
        )

        # 変数初期化
        sess.run(tf.global_variables_initializer())

        # saver 準備（solver後じゃないと変数が構築されてない）
        saver = tf.train.Saver()

        # チェックポイントが存在すれば復元
        if os.path.exists(ckpt_path + ".index"):
            print("✅ Checkpoint found. Restoring model...")
            saver.restore(sess, ckpt_path)
            print("✅ Model restored!")
            # epochの復元
            with open(os.path.join(dirname, f"epoch_{_NUM_PLAYERS}.txt"), "r") as f:
                epoch = int(f.read())
            print("Epoch:", epoch)

            # Advantage lossesの復元
            with open(os.path.join(dirname, f"advantage_losses_{_NUM_PLAYERS}.npy"), "rb") as f:
                advantage_losses_history = np.load(f)
            # print("Advantage losses:", advantage_losses_history[-1])

            # Policy lossの復元
            with open(os.path.join(dirname, f"policy_loss_{_NUM_PLAYERS}.npy"), "rb") as f:
                policy_loss_history = np.load(f)
            print("Policy loss:", policy_loss_history[-1])
        
        else:

            # チェックポイントが存在しない場合は新規作成
            policy_loss_history = np.array([])
            advantage_losses_history = None


        # 学習（追加訓練）
        print("🚀 Starting DeepCFR training...")
        # 進捗表示
        for i in tqdm.tqdm(range(100000)):
            # 1イテレーションの学習
            _, advantage_losses, policy_loss = solver.solve()

            # 保存
            saver.save(sess, ckpt_path)
            print("💾 Model saved to:", ckpt_path)

            with open(os.path.join(dirname, f"epoch_{_NUM_PLAYERS}.txt"), "w") as f:
                f.write(str(i + 1))

            advantage_losses_ls = np.array([advantage_losses[i] for i in range(_NUM_PLAYERS)]).reshape(-1, _NUM_PLAYERS)

            if advantage_losses_history is None:
                advantage_losses_history = advantage_losses_ls
            else:
                advantage_losses_history = np.vstack((advantage_losses_history, advantage_losses_ls))

            # historyに追加
            # advantage_losses_history = np.append(advantage_losses_history, advantage_losses)
            policy_loss_history = np.append(policy_loss_history, policy_loss)

            # 保存
            np.save(os.path.join(dirname, f"advantage_losses_{_NUM_PLAYERS}.npy"), advantage_losses_history)
            np.save(os.path.join(dirname, f"policy_loss_{_NUM_PLAYERS}.npy"), policy_loss_history)

            # plot loss
            for i in range(_NUM_PLAYERS):
                plt.plot(advantage_losses_history[:,i], label=f"Player_{i}_Advantage_Loss")
            plt.xlabel("Iteration")
            plt.ylabel("Loss")
            plt.yscale('log')
            plt.title("DeepCFR Advantage Loss")
            plt.legend()
            plt.savefig(os.path.join(dirname, f"advantage_loss_{_NUM_PLAYERS}.png"))
            plt.close()
            
            # plot policy loss
            plt.plot(policy_loss_history, label="Policy Loss")
            plt.xlabel("Iteration")
            plt.yscale('log')
            plt.ylabel("Loss")
            plt.title("DeepCFR Loss")
            plt.legend()
            plt.savefig(os.path.join(dirname, f"policy_loss_{_NUM_PLAYERS}.png"))
            plt.close()

        print("✅ Training complete!")