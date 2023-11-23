from typing import Callable
from abc import ABCMeta, abstractmethod

from transformers import pipeline


def sentiment_function(
    s: str,
    sentiment_pipeline: Callable[[str], float] = pipeline(
        "text-classification", model="martin-ha/toxic-comment-model"
    ),
) -> float | list:
    """Toxicity model into a"""
    text_data = [
        d["score"] if d["label"] == "toxic" else 1 - d["score"]
        for d in sentiment_pipeline(s)
    ]
    return text_data[0] if len(text_data) == 1 else text_data


class RewardEngine(metaclass=ABCMeta):
    @abstractmethod
    def calculate_reward(
        state: str, state_sentiment: str, action: str, terminal: bool
    ) -> float:
        pass


class RelativeToxicityRewardEngine(RewardEngine):
    def __init__(
        self, toxicity_function: Callable[[str], float], terminal_scalar: int = 5
    ):
        self.model: Callable[[str], float] = toxicity_function
        self.terminal_scalar: int = terminal_scalar

    def calculate_reward(
        state: str, state_toxicity: str, response: str, action: str, terminal: bool
    ) -> float:
        response_toxicity = self.model(response)
        reward = response_toxicity - state_toxicity
        if terminal:
            return self.terminal_scalar * reward
        else:
            return reward
