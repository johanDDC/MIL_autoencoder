from dataclasses import dataclass


@dataclass()
class OptimizerConfig:
    optimizer_name: str
    lr: float


@dataclass()
class TrainConfig:
    optimizer_config: OptimizerConfig


@dataclass()
class ModelConfig:
    pass


@dataclass()
class Config:
    train_config: TrainConfig
    model_config: ModelConfig
