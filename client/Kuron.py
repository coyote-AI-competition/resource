from .not_websocket_client import Client
import numpy as np
import random
import json
import os
import tensorflow as tf
import pyspiel
import requests
import random
from dotenv import load_dotenv


dotenv_path = os.path.join(os.path.dirname(__file__), "..", '.env')

load_dotenv(dotenv_path)

class Kuron(Client):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.life_dict = dict()  # Initialize life_ls to None
        self.last_other_info = None  # Initialize last_other_info to None
        self.used_card_ls = []  # Initialize used_card to None

    def AI_player_action(self, others_info, sum, log, actions, round_num):

        if len(actions) == 1:
            # 1つの行動しかない場合、その行動を選択
            return actions[0]                

        # Estimate the used card based on others_info
        used_card = self.used_card_estimate(others_info)
        others_card = [info["card_info"] for info in others_info]

        print(f"[Kuron] Current Declaration: {actions[1]-1}")

        average_policy = requests.get(f"{os.getenv("API_URL")}:{os.getenv("PORT")}/kuron?player_num={len(others_info)+1}&current_declaration={actions[1]-1}&used_card={str(used_card)}&others_card={str(others_card)}").json()
        print(f"[Kuron] Average policy: {average_policy}")

        rnd = random.random()

        print(f"[Kuron] Random number: {rnd}")

        # # 行動確率に基づいて行動を選択
        if rnd < float(average_policy["0"]):
            return actions[1]  # %の確率で最初の行動を選択
        else:
            return actions[0]  # %の確率で2番目の行動を選択
    
    def used_card_estimate(self, others_info):
        #Estimate the used card based on others_info, sum, log, actions

        # if self.life_ls exists:
        if len(self.life_dict) == 0:
            for info in others_info:
                print(f"[Kuron] Life info: {info}")
                self.life_dict[info["name"]] = info["life"]
            self.last_other_info = others_info  # Store the initial state of others_info

        now_life_dict = {info["name"] :info["life"] for info in others_info}
        # Check if the life_dict has changed
        if self.life_dict != now_life_dict:
            print(f"[Kuron] Life before: {self.life_dict}")
            self.life_dict = dict()  # Reset life_dict for the new game state
            for info in others_info:
                self.life_dict[info["name"]] = info["life"]
            
            for info in self.last_other_info:
                self.used_card_ls.append(info["card_info"])

            if 101 in self.used_card_ls:
                self.used_card_ls = []

            self.last_other_info = others_info  # Update last_other_info to the current state
            print(f"[Kuron] Life after: {self.life_dict}")

        visible_cards = [info["card_info"] for info in others_info]
        print(f"[Kuron] Used card: {self.used_card_ls + visible_cards}")

        return self.used_card_ls + visible_cards
