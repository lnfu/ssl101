import ml_collections

from configs._base import get_base_pretrain_config


def get_config() -> ml_collections.ConfigDict:
    config = get_base_pretrain_config()

    # Optimizer
    config.optimizer = "adamw"
    config.learning_rate = 1e-3
    config.weight_decay = 1e-4

    return config
