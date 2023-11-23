from dataclasses import dataclass


@dataclass
class ToxicityEngineConstants:
    STATE_EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L12-v2"
    TOXICITY_CLF_MODEL: str = "martin-ha/toxic-comment-model"
    # TEXT_GEN_MODEL: str = "google/flan-t5-base"

    REWARD_BASE_SCALAR: int | float = 100
    REWARD_DONE_SCALAR: int | float = 5
