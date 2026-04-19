import ml_collections


def get_config() -> ml_collections.ConfigDict:
    config = ml_collections.ConfigDict()

    # Model
    config.model = "ResNet18"
    config.num_classes = 10

    # Data
    config.dataset = "cifar10"
    config.batch_size = 128

    # Optimizer
    config.optimizer = "adamw"
    config.learning_rate = 1e-3
    config.warmup_epochs = 5
    config.num_epochs = 100
    config.weight_decay = 1e-2

    # Augmentation
    config.augment = False

    # Training
    config.seed = 42
    config.log_every_steps = 10
    config.checkpoint_dir = "checkpoints"
    config.checkpoint_every_epochs = 5

    # Logging
    config.wandb_project = "ssl101-lab-02"
    config.dry_run = False

    return config
