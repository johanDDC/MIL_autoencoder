from configs.base_config import *


class FCTrainConfig(TrainConfig):
    train_batch_size = 128
    eval_batch_size = 128

    num_epoches = 100
    learning_rate = 1e-4

    checkpoint_path = "checkpoints/fc"


class FCModelConfig(ModelConfig):
    use_pretrained = False
    pretrained_path = ""

    n_layers = 3
    scale_factor = 2
    negative_slope = .2


FCConfig = Config(FCTrainConfig(), FCModelConfig())
