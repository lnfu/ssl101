import ml_collections


def get_config() -> ml_collections.ConfigDict:
    config = ml_collections.ConfigDict()

    # Model
    config.model = "SimCLR"

    # Data
    config.dataset = "stl10"
    config.batch_size = 256

    # Optimizer
    config.optimizer = "adamw"
    config.learning_rate = 1e-3
    config.warmup_epochs = 10
    config.num_epochs = 200
    config.weight_decay = 1e-4

    # SimCLR
    config.temperature = 0.5
    config.augment = True

    # Training
    config.seed = 42
    config.checkpoint_dir = "checkpoints"
    config.checkpoint_every_epochs = 10

    # Logging
    config.wandb_project = "ssl101-lab-03"
    config.dry_run = False

    return config
