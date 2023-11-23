from abc import ABCMeta, abstractmethod

import torch
import numpy as np
from gym import spaces
from sentence_transformers import SentenceTransformer


class StateEngine(metaclass=ABCMeta):
    @abstractmethod
    def encode_state(self, s: str) -> torch.Tensor:
        pass

    @property
    @abstractmethod
    def observation_space(self):
        pass


class SentenceTransformerStateEngine(StateEngine):
    def __init__(
        self,
        embedding_model: str
        | SentenceTransformer = "sentence-transformers/all-MiniLM-L12-v2",
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

    def encode_state(self, s: str) -> torch.Tensor:
        return torch.Tensor(self.model.encode(s))

    @staticmethod
    def get_embedding_dims(model: SentenceTransformer) -> int:
        if hasattr(model, "_modules"):
            for key, module in model._modules.items():
                if hasattr(module, "word_embedding_dimension"):
                    return module.word_embedding_dimension
        return -1
