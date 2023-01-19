from dataclasses import dataclass


@dataclass()
class TrainConfig:
    pass


@dataclass()
class ModelConfig:
    pass


@dataclass()
class Config:
    train_config: TrainConfig
    model_config: ModelConfig
