from dataclasses import dataclass


@dataclass
class DataConstants:
    SOURCE: str = (
        "https://www.kaggle.com/datasets/jp797498e/twitter-entity-sentiment-analysis/"
    )
    TRAIN_DATA_PATH: str = "data/data.csv"
    VALID_DATA_PATH: str = "twitter_validation.csv"


@dataclass
class ExperimentConstants:
    NUM_TEXTS: int = 1000
    SPLIT: float = 0.7
    PROMPT_LEN: int = 20
    FEATURE_MODEL: str = "sentence-transformers/all-MiniLM-L12-v2"
