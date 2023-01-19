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
    train_batch_size = 16
    eval_batch_size = 16
    num_epoches = 10

    checkpoint_path = "checkpoints"

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
