from dataclasses import dataclass


@dataclass
class DataConstants:
    SOURCE: str = (
        "https://www.kaggle.com/datasets/jp797498e/twitter-entity-sentiment-analysis/"
    )
    TRAIN_DATA_PATH: str = "data/data.csv"
    VALID_DATA_PATH: str = "twitter_validation.csv"


@dataclass
class BasicConfig:
    LOG_INTERVAL: int = 200


@dataclass
class ExperimentConstants:
    NUM_ACTIONS: int = 250
    MODEL = "google/flan-t5-base"
    INSTRUCTION_PROMPT = ""
    TIMESTEPS: int = 25000
    OUTPUT_DIR: str = "output/no-base-text"
