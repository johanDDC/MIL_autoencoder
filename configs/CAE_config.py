import torch.nn as nn

from configs.base_config import OptimizerConfig, TrainConfig, ModelConfig, Config


class CAETrainConfig(TrainConfig):
    train_batch_size = 32
    eval_batch_size = 32
    num_epoches = 100

    checkpoint_path = "checkpoints/cae"


class CAEModelConfig(ModelConfig):
    n_layers = 3
    scale_factor = 2
    in_channels = 3
    start_num_filters = 24

    negative_slope = .2


CAEConfig = Config(
    CAETrainConfig(
        optimizer_config=OptimizerConfig("Adam", 1e-4),
        scheduler_config=None,
        loss=nn.MSELoss
    ),
    CAEModelConfig()
)
