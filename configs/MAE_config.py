import math
import torch

from configs.base_config import OptimizerConfig, TrainConfig, ModelConfig, Config, SchedulerConfig

NUM_EPOCHES = 2000
NUM_WARMUP_EPOCHES = 200
MASK_RATIO = .75


class MAETrainConfig(TrainConfig):
    train_batch_size = 128
    eval_batch_size = 128
    num_epoches = NUM_EPOCHES

    checkpoint_path = "checkpoints/mae"


class CAEModelConfig(ModelConfig):
    img_size = 32
    patch_size = 2
    in_channels = 3
    inner_dim = 768
    hidden_dim = 192
    patch_ratio = MASK_RATIO

    encoder_n_layers = 6
    encoder_n_heads = 3

    decoder_n_layers = 2
    decoder_n_heads = 3


class MAEOptimizerConfig(OptimizerConfig):
    weight_decay = .05


class MAEScheduler(SchedulerConfig):
    lr_lambda = lambda self, epoch: min((epoch + 1) / (NUM_WARMUP_EPOCHES + 1e-7),
                                  0.5 * (math.cos(epoch / NUM_EPOCHES * math.pi) + 1))


MAEConfig = Config(
    MAETrainConfig(
        optimizer_config=MAEOptimizerConfig("AdamW", 2e-4),
        scheduler_config=MAEScheduler("LambdaLR", "step"),
        loss=lambda: lambda predictions_and_mask, targets: torch.mean(
            (predictions_and_mask[0] - targets) ** 2 * predictions_and_mask[1]) / MASK_RATIO
    ),
    CAEModelConfig()
)
