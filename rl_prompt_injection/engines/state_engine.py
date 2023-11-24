from abc import ABCMeta, abstractmethod

import numpy as np
from gymnasium import spaces
from sentence_transformers import SentenceTransformer

from rl_prompt_injection.engines.basic_config import ToxicityEngineConstants


class StateEngine(metaclass=ABCMeta):
    @abstractmethod
    def encode_state(self, s: str) -> np.array:
        pass

    # @property
    # @abstractmethod
    # def observation_space(self):
    #     pass


class SentenceTransformerStateEngine(StateEngine):
    def __init__(
        self,
        embedding_model: str
        | SentenceTransformer = ToxicityEngineConstants.STATE_EMBEDDING_MODEL,
    ):
        if isinstance(embedding_model, str):
            self.model: SentenceTransformer = SentenceTransformer(embedding_model)
        else:
            self.model: SentenceTransformer = embedding_model

        self.observation_space: spaces.Box = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.get_embedding_dims(self.model),),
            dtype=np.float32,
        )

    def encode_state(self, s: str) -> np.array:
        return self.model.encode(s)

    @staticmethod
    def get_embedding_dims(model: SentenceTransformer) -> int:
        if hasattr(model, "_modules"):
            for key, module in model._modules.items():
                if hasattr(module, "word_embedding_dimension"):
                    return module.word_embedding_dimension
        return -1
