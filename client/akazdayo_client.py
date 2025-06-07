import os
import requests
import json
from .client import Client
from google import genai
from pydantic import BaseModel


class CoyoteResponse(BaseModel):
    action: int


class AkazdayoClient(Client):
    def __init__(
        self,
        player_name="GeminiAI",
        is_ai=True,
        llm_model="gemini-2.5-flash-preview-05-20",
    ):
        super().__init__(player_name=player_name, is_ai=is_ai)
        api_key = os.getenv("GEMINI_API_KEY")
        if api_key:
            self.model = genai.Client(api_key=api_key)
            self.llm_model = llm_model
            self.use_api = True
        else:
            print("Warning: GEMINI_API_KEY not found. Using fallback strategy.")
            self.model = None
            self.llm_model = None
            self.use_api = False

    def AI_player_action(self, others_info, sum, log, actions, round_num):
        prompt = self._build_prompt(others_info, sum, log, actions, round_num)
        try:
            response = self.model.models.generate_content(
                model=self.llm_model,
                contents=prompt,
                config={
                    "response_mime_type": "application/json",
                    "response_schema": CoyoteResponse,
                },
            )
            action = self._parse_and_validate_response(response.text, actions)
            print(f"[GeminiClient] API response: {response.text}")
            print(f"[GeminiClient] Selected action: {action}")
            return action
        except Exception as e:
            print(f"[GeminiClient] Error: {e}")
            # 安全策としてCOYOTE宣言（-1）または最小値
            return -1 if -1 in actions else min(actions)

    def _build_prompt(self, others_info, sum, log, actions, round_num):
        # 必要な情報を日本語でプロンプト化
        prompt = (
            "あなたはコヨーテゲームのAIです。以下の状況から最適なアクションを1つ選び、"
            '必ずJSON形式（例: {"action": 23}）で返答してください。\n'
            f"others_info: {json.dumps(others_info, ensure_ascii=False)}\n"
            f"sum: {sum}\n"
            f"log: {json.dumps(log, ensure_ascii=False)}\n"
            f"actions: {actions}\n"
            f"round_num: {round_num}\n"
            '出力例: {"action": 23}\n'
            "必ずactionsの中から選択してください。"
        )
        return prompt

    def _parse_and_validate_response(self, response_text, actions):
        # GeminiのStructured Output(JSON)を厳密に検証・パース
        try:
            obj = json.loads(response_text)
            action = obj.get("action")
            if action in actions:
                return action
        except Exception as e:
            print(f"[GeminiClient] Parse error: {e}")
        # 不正な場合はCOYOTE宣言（-1）または最小値
        return -1 if -1 in actions else min(actions)
