import ml_collections

from configs._base import get_base_pretrain_config


def get_config() -> ml_collections.ConfigDict:
    config = get_base_pretrain_config()

    # Optimizer (SimCLR uses linear LR scaling: 0.3 * batch_size / 256)
    config.optimizer = "sgd"
    config.learning_rate = 0.3
    config.momentum = 0.9
    config.weight_decay = 1e-6

    return config
