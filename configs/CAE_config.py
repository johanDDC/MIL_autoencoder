import torch.nn as nn

from configs.base_config import OptimizerConfig, TrainConfig, ModelConfig, Config, SchedulerConfig, ClassifierConfig


class CAETrainConfig(TrainConfig):
    train_batch_size = 32
    eval_batch_size = 32
    num_epoches = 100

    checkpoint_path = "checkpoints/cae"


class CAEModelConfig(ModelConfig):
    n_layers = 3
    scale_factor = 2
    in_channels = 3
    start_num_filters = 16

    negative_slope = .2

class CAECosineLR(SchedulerConfig):
    T_max = 1562


CAEConfig = Config(
    CAETrainConfig(
        optimizer_config=OptimizerConfig("Adam", 1e-4),
        scheduler_config=CAECosineLR("CosineAnnealingLR", "step"),
        loss=nn.MSELoss
    ),
    CAEModelConfig(),
    ClassifierConfig(num_classes=100, num_epoches=10, input_dim=192, intermediate_dim=150)
)
