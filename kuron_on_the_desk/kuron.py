from flask import Flask, request, jsonify
import os
import json
import tensorflow.compat.v1 as tf
import numpy as np
import logging
from open_spiel.python.algorithms import deep_cfr
import pyspiel

import game_set.game_setting_2
import game_set.game_setting_3
import game_set.game_setting_4
import game_set.game_setting_5
import game_set.game_setting_6

# compat.v1を使用するための設定
tf.disable_v2_behavior()

dir = os.path.dirname(__file__)
file_path = os.path.join(dir, "server.log")


logging.basicConfig(filename=file_path, level=logging.INFO)
logger = logging.getLogger(__name__)


game_sets = [
    game_set.game_setting_2.CoyoteGame(),
    game_set.game_setting_3.CoyoteGame(),
    game_set.game_setting_4.CoyoteGame(),
    game_set.game_setting_5.CoyoteGame(),
    game_set.game_setting_6.CoyoteGame()
]

game_state = [
    game_set.game_setting_2.CoyoteState,
    game_set.game_setting_3.CoyoteState,
    game_set.game_setting_4.CoyoteState,
    game_set.game_setting_5.CoyoteState,
    game_set.game_setting_6.CoyoteState
]

app = Flask(__name__)

@app.route('/')
def index():
    return "Welcome to the Kuron on the Desk!"

@app.route('/kuron')
def kuron():

    load_state = [game_state[i](game_sets[i]) for i in range(len(game_sets))]

    # クエリパラメータを取得
    player_num = request.args.get('player_num', type=int)
    current_declaration = request.args.get('current_declaration', type=int)
    used_card = request.args.get('used_card', type=str)
    others_card = request.args.get('others_card', type=str)

    used_card = json.loads(used_card)
    others_card = json.loads(others_card)

    load_state[player_num-2].set_initial_state(player_num, current_declaration, used_card, others_card)

    logger.info(f"player_num: {player_num}, current_declaration: {current_declaration}, used_card: {used_card}, others_card: {others_card}")

    dirname = os.path.dirname(__file__)

    # 保存先パス
    ckpt_path = os.path.join(dirname, "client", f"checkpoints_{player_num}", "deep_cfr.ckpt")

    with tf.Session() as sess:

        # DeepCFRSolver 構築
        solver = deep_cfr.DeepCFRSolver(
            game=game_sets[player_num-2],
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
            logger.info(f"model_{player_num} Checkpoint found. Restoring model...")
            saver.restore(sess, ckpt_path)
            logger.info(f"model_{player_num} Model restored successfully!")        
        else:
            # チェックポイントが存在しない場合はerror
            print("❌ Checkpoint not found. Exiting...")
            exit(1)

        average_policy = solver.action_probabilities(load_state[player_num-2])

        json_average_policy = {}

        for key in average_policy.keys():
            json_average_policy[str(key)] = str(float(average_policy[key]))


    return jsonify(json_average_policy)



if __name__ == "__main__":
    app.run(host="0.0.0.0", port=49155)
