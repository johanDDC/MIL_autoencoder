import torch.nn as nn

from configs.base_config import OptimizerConfig, TrainConfig, ModelConfig, Config, SchedulerConfig, ClassifierConfig


class SimpleTrainConfig(TrainConfig):
    train_batch_size = 128
    eval_batch_size = 128
    num_epoches = 100

    checkpoint_path = "checkpoints/simple"


class SimpleModelConfig(ModelConfig):
    pass
    # encoder_n_layers = 4
    # decoder_n_layers = 4
    # encoder_scale_factor = 2
    # decoder_scale_factor = 2
    # in_channels = 3
    # start_num_filters = 6
    #
    # negative_slope = .2

class SimpleCosineLR(SchedulerConfig):
    T_max = 1562

class SimpleOptimizerConfig(OptimizerConfig):
    weight_decay = 5e-2
    betas=(0.9, 0.999)


SimConfig = Config(
    SimpleTrainConfig(
        optimizer_config=SimpleOptimizerConfig("AdamW", lr=4e-3),
        scheduler_config=SimpleCosineLR("CosineAnnealingLR", "step"),
        loss=nn.MSELoss
    ),
    SimpleModelConfig(),
    ClassifierConfig(num_classes=100, num_epoches=10, input_dim=192, intermediate_dim=150)
)
