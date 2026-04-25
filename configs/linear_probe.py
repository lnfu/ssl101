import ml_collections
from ml_collections import config_dict


def get_config() -> ml_collections.ConfigDict:
    config = ml_collections.ConfigDict()

    # Model
    config.num_classes = 10

    # Data
    config.dataset = "stl10"
    config.batch_size = 256

    # Pretrained checkpoint to load (must be set on the command line).
    # Plain placeholder (not required_placeholder) because config_flags locks
    # the config during --config=... parse, before --config.X=Y overrides are
    # applied; required placeholders fail at that lock step. main() validates.
    config.pretrain_checkpoint_dir = "checkpoints"
    config.pretrain_run_name = config_dict.placeholder(str)
    config.pretrain_epoch = config_dict.placeholder(int)

    # Optimizer (SimCLR linear eval: SGD + momentum, no weight decay)
    config.learning_rate = 0.1
    config.momentum = 0.9
    config.num_epochs = 90
    config.warmup_epochs = 0

    # Training
    config.seed = 42

    # Logging
    config.wandb_project = "ssl101-lab-03"
    config.dry_run = False

    return config
