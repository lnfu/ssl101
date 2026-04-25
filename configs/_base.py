import ml_collections


def get_base_pretrain_config() -> ml_collections.ConfigDict:
    """Shared defaults for SimCLR pretraining configs (sgd, adamw, ...)."""
    config = ml_collections.ConfigDict()

    # Model
    config.model = "SimCLR"

    # Data
    config.dataset = "stl10"
    config.batch_size = 256

    # Schedule (optimizer-specific fields are set by each child config)
    config.warmup_epochs = 10
    config.num_epochs = 200

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
