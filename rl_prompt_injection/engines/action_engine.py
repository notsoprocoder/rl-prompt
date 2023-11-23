from abc import ABCMeta, abstractmethod

import numpy as np
from gymnasium import spaces
from transformers import pipeline
from transformers.pipelines.text2text_generation import Text2TextGenerationPipeline

from rl_prompt_injection.engines.basic_config import ToxicityEngineConstants


class ActionEngine(metaclass=ABCMeta):
    # @property
    # @abstractmethod
    # def action_space(self):
    #     pass

    @abstractmethod
    def decode_action(self, action: int) -> str:
        pass


class Text2TextActionSpace(ActionEngine):
    def __init__(
        self,
        model: str | Text2TextGenerationPipeline,
        num_actions: int = 1000,
    ):
        if isinstance(model, str):
            vocab_dict = pipeline(
                "text2text-generation", model=model
            ).tokenizer.get_vocab()
        else:
            vocab_dict = model.tokenizer.get_vocab()
        vocab = np.random.choice(np.array(list(vocab_dict.keys())), size=num_actions)
        self.vocab_dict = dict(zip(list(range(num_actions)), vocab))

        self.action_space = spaces.Discrete(num_actions)

    def decode_action(self, action: int) -> str:
        return self.vocab_dict.get(action)
