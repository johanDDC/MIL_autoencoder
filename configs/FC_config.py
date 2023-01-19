import torch.nn as nn

from configs.base_config import OptimizerConfig, TrainConfig, ModelConfig, Config


class FCTrainConfig(TrainConfig):
    train_batch_size = 128
    eval_batch_size = 128

    num_epoches = 100

    checkpoint_path = "checkpoints/fc"


class FCModelConfig(ModelConfig):
    use_pretrained = False
    pretrained_path = ""

    n_layers = 3
    scale_factor = 2
    negative_slope = .2


FCConfig = Config(
    FCTrainConfig(
        optimizer_config=OptimizerConfig("Adam", 1e-4),
        scheduler_config=None,
        loss=nn.MSELoss
    ),
    FCModelConfig()
)
