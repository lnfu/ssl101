import ml_collections


def get_config():
    config = ml_collections.ConfigDict()

    # Model
    config.model = "ResNet18"

    # Data
    config.dataset = "cifar10"
    config.batch_size = 128

    # Optimizer
    config.learning_rate = 0.1
    config.momentum = 0.9
    config.warmup_epochs = 5.0
    config.num_epochs = 100.0
    config.weight_decay = 1e-4

    # Training
    config.log_every_steps = 100
    config.half_precision = False

    return config
