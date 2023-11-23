from abc import ABCMeta, abstractmethod

from gym import spaces
from transformers import pipeline
from transformers.pipelines.text2text_generation import Text2TextGenerationPipeline


class ActionEngine(metaclass=ABCMeta):
    @property
    @abstractmethod
    def action_space(self):
        pass

    @abstractmethod
    def encode_action(self, action: int) -> str:
        pass


class Text2TextActionSpace(ActionEngine):
    def __init__(
        self,
        model: str | Text2TextGenerationPipeline,
        num_actions: int = 1000,
        *args,
        **kwargs
    ):
        super(Text2TextActionSpace, self).__init__()
        if isinstance(model, str):
            vocab_dict = pipeline(
                "text2text-generation", model="google/flan-t5-base"
            ).tokenizer.get_vocab()
        else:
            vocab_dict = model.tokenizer.get_vocab()
        vocab = np.random.choice(np.array(list(vocab_dict.keys())), size=num_actions)
        self.vocab_dict = dict(zip(list(range(num_actions)), vocab))

        self.action_space = spaces.Discrete(num_actions)

    def decode_action(self, action: int) -> str:
        return self.vocab_dict.get(action)
