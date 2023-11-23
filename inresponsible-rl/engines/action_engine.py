from abc import ABCMeta

from gym import spaces
from transformers import pipeline
from transformers.pipelines.text2text_generation import Text2TextGenerationPipeline


class ActionEngine(ABCMeta):
    def transform_action(self, action: int) -> str:
        return self.vocab_dict.get(action)


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
