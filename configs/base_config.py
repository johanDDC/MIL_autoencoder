import torch

from dataclasses import dataclass
from typing import Union


@dataclass()
class SchedulerConfig:
    name: str


@dataclass()
class OptimizerConfig:
    name: str
    lr: float


@dataclass()
class TrainConfig:
    optimizer_config: OptimizerConfig
    scheduler_config: Union[None, SchedulerConfig]
    loss: torch.nn.Module


@dataclass()
class ModelConfig:
    use_pretrained: bool
    pretrained_path: Union[None, str]


@dataclass()
class Config:
    train_config: TrainConfig
    model_config: ModelConfig
