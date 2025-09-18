# src/utils/ai_provider.py
from dataclasses import dataclass
from typing import List, Dict, Any
import os

@dataclass
class ChatResult:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    provider: str = ""
    model: str = ""

class AIProvider:
    def __init__(self, provider: str = "openai", api_key: str = None):
        self.provider = provider.lower().strip()
        self.api_key = api_key
        if self.provider == "openai":
            import openai
            self.client = openai
            self.api_key = self.api_key or os.getenv("OPENAI_API_KEY")
            if self.api_key: self.client.api_key = self.api_key
        elif self.provider == "gemini":
            from google import generativeai as genai
            self.client = genai
            self.api_key = self.api_key or os.getenv("GOOGLE_API_KEY")
            if self.api_key: self.client.configure(api_key=self.api_key)
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    def chat(self, model: str, messages: List[Dict[str,Any]], **kwargs) -> ChatResult:
        if self.provider == "openai":
            # Works with openai<=0.28 style. If you're on >=1.x, adapt to client.chat.completions.create(...)
            resp = self.client.ChatCompletion.create(model=model, messages=messages, **kwargs)
            text = resp.choices[0].message["content"]
            usage = getattr(resp, "usage", {}) or {}
            in_tok = int(getattr(usage, "prompt_tokens", usage.get("prompt_tokens", 0)) or 0)
            out_tok = int(getattr(usage, "completion_tokens", usage.get("completion_tokens", 0)) or 0)
            tot_tok = int(getattr(usage, "total_tokens", usage.get("total_tokens", in_tok + out_tok)) or (in_tok + out_tok))
            return ChatResult(text, in_tok, out_tok, tot_tok, "openai", model)

        if self.provider == "gemini":
            model_obj = self.client.GenerativeModel(model)
            prompt = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
            resp = model_obj.generate_content(prompt)
            text = getattr(resp, "text", "") or ""
            um = getattr(resp, "usage_metadata", None)
            in_tok = int(getattr(um, "prompt_token_count", 0) or 0) if um else 0
            out_tok = int(getattr(um, "candidates_token_count", 0) or 0) if um else 0
            tot_tok = int(getattr(um, "total_token_count", in_tok + out_tok) or (in_tok + out_tok)) if um else (in_tok + out_tok)
            return ChatResult(text, in_tok, out_tok, tot_tok, "gemini", model)

        raise ValueError(f"Unsupported provider: {self.provider}")
