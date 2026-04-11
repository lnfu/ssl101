import ml_collections


def get_config() -> ml_collections.ConfigDict:
    config = ml_collections.ConfigDict()

    # Model
    config.model = "MLP"
    config.hidden_dim = 128
    config.num_classes = 10

    # Data
    config.dataset = "mnist"
    config.batch_size = 256

    # Optimizer
    config.learning_rate = 0.05
    config.momentum = 0.9
    config.num_epochs = 80

    # Training
    config.seed = 42
    config.log_every_steps = 10
    config.checkpoint_dir = "checkpoints"
    config.checkpoint_every_epochs = 5

    # Logging
    config.wandb_project = "ssl101"

    return config
