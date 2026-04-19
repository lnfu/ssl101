import ml_collections

from configs.adamw import get_config as _get_base_config


def get_config() -> ml_collections.ConfigDict:
    config = _get_base_config()
    config.augment = True
    return config
