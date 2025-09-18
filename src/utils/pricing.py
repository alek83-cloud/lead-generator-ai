# src/utils/pricing.py
from dataclasses import dataclass, field
from typing import Dict, Any
from datetime import datetime

# USD cost per 1K tokens; tweak as needed
RATES_USD_PER_1K = {
    "default": {"input": 0.03, "output": 0.06},  # fallback if model not found
    "openai": {
        "gpt-4": {"input": 0.03, "output": 0.06},
        "gpt-4o-mini": {"input": 0.003, "output": 0.015},
    },
    "gemini": {
        "gemini-1.5-pro": {"input": 0.007, "output": 0.021},
        "gemini-1.5-flash": {"input": 0.00035, "output": 0.00053},
    },
}

@dataclass
class ModelUsage:
    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    timestamp: datetime = field(default_factory=datetime.now)
    def calculate_cost(self, rates: Dict[str,Dict[str,Dict[str,float]]] = None) -> float:
        rates = rates or RATES_USD_PER_1K
        p, m = (self.provider or "default").lower(), self.model
        r = rates.get(p, {}).get(m, rates["default"])
        return round((self.input_tokens/1000)*r["input"] + (self.output_tokens/1000)*r["output"], 6)

class ModelsPricing:
    def __init__(self, provider: str = "openai", model: str = "gpt-4"):
        self.provider, self.model = provider.lower(), model
        self.input_tokens = 0; self.output_tokens = 0; self.total_cost = 0.0
    def set_model(self, provider: str, model: str):
        self.provider, self.model = provider.lower(), model
    def track_usage(self, input_tokens=0, output_tokens=0):
        self.input_tokens += input_tokens; self.output_tokens += output_tokens
        usage = ModelUsage(self.provider, self.model, input_tokens, output_tokens)
        self.total_cost += usage.calculate_cost()
    @property
    def total_tokens(self): return self.input_tokens + self.output_tokens
    def get_usage_summary(self) -> Dict[str,Any]:
        return {"total_tokens": self.total_tokens, "input_tokens": self.input_tokens, "output_tokens": self.output_tokens, "total_cost": round(self.total_cost, 6)}
