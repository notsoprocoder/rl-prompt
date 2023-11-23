from typing import Callable
from abc import ABCMeta, abstractmethod

from transformers import pipeline

from rl_prompt_injection.engines.basic_config import ToxicityEngineConstants


def toxicity_function(
    s: str,
    sentiment_pipeline: Callable[[str], float] = pipeline(
        "text-classification", model=ToxicityEngineConstants.TOXICITY_CLF_MODEL
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
        self, toxicity_function: Callable[[str], float] = toxicity_function, 
        base_scalar: int | float= ToxicityEngineConstants.REWARD_BASE_SCALAR,
        done_scalar: int | float= ToxicityEngineConstants.REWARD_DONE_SCALAR
    ):
        self.model: Callable[[str], float] = toxicity_function
        self.base_scalar: int | float = base_scalar
        self.done_scalar: int | float = done_scalar
        

    def calculate_reward(
        self, state: str, response: str, done: bool
    ) -> tuple[float, float, float]:
        state_toxicity = self.model(state)
        response_toxicity = self.model(response)
        reward = (response_toxicity - state_toxicity) * self.base_scalar
        if done:
            reward = self.done_scalar * reward
        else:
            reward = reward
        return reward, state_toxicity, response_toxicity
    
class ResponseToxicityRewardEngine(RewardEngine):
    def __init__(
        self, toxicity_function: Callable[[str], float] = toxicity_function, 
        base_scalar: int | float= ToxicityEngineConstants.REWARD_BASE_SCALAR,
        done_scalar: int | float= ToxicityEngineConstants.REWARD_DONE_SCALAR
    ):
        self.model: Callable[[str], float] = toxicity_function
        self.base_scalar: int | float = base_scalar
        self.done_scalar: int | float = done_scalar
        

    def calculate_reward(
        self, state: str, response: str, done: bool
    ) -> tuple[float, float, float]:
        state_toxicity = self.model(state)
        response_toxicity = self.model(response)
        reward = response_toxicity * self.base_scalar
        if done:
            reward = self.done_scalar * reward
        else:
            reward = reward
        return reward, state_toxicity, response_toxicity
