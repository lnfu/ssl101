import logging
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import wandb

from checkpoint import make_checkpointer, save_checkpoint
from configs.default import get_config
from input_pipeline import create_dataset, get_split_size
from models import accuracy, cross_entropy_loss, init_params

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def make_train_step(optimizer: optax.GradientTransformation):
    """Return a jit-compiled train step closed over the optimizer."""

    @jax.jit
    def _step(
        params: dict,
        opt_state: optax.OptState,
        X: jax.Array,
        y: jax.Array,
    ) -> tuple[dict, optax.OptState, jax.Array]:
        loss, grads = jax.value_and_grad(cross_entropy_loss)(params, X, y)
        updates, new_opt_state = optimizer.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, loss

    return _step


def _load_test_set(batch_size: int) -> tuple[jax.Array, jax.Array]:
    """Load the full test set into two concatenated JAX arrays."""
    xs, ys = [], []
    for batch in create_dataset("test", batch_size):
        xs.append(batch["image"])
        ys.append(batch["label"])
    return jnp.array(np.concatenate(xs)), jnp.array(np.concatenate(ys))


def main() -> None:
    config = get_config()

    # Init params & optimizer
    key = jax.random.PRNGKey(config.seed)
    params = init_params(
        key, input_dim=784, hidden_dim=config.hidden_dim, num_classes=config.num_classes
    )
    optimizer = optax.sgd(config.learning_rate, momentum=config.momentum)
    opt_state = optimizer.init(params)
    train_step = make_train_step(optimizer)

    # Wandb (init first so the checkpoint dir can mirror the wandb run name)
    wandb.init(project=config.wandb_project, config=config.to_dict())
    run_name = Path(wandb.run.dir).parent.name
    checkpointer, run_dir = make_checkpointer(config.checkpoint_dir, run_name)

    # Pre-load test set once
    logger.info("Loading test set...")
    test_X, test_y = _load_test_set(config.batch_size)

    # Training loop
    train_ds = create_dataset("train", config.batch_size, seed=config.seed)
    steps_per_epoch = get_split_size("train") // config.batch_size
    num_epochs = int(config.num_epochs)

    logger.info(
        "Training: %d epochs × %d steps/epoch (batch=%d)",
        num_epochs,
        steps_per_epoch,
        config.batch_size,
    )

    global_step = 0
    for epoch in range(1, num_epochs + 1):
        epoch_loss = 0.0

        for _ in range(steps_per_epoch):
            batch = next(train_ds)
            X_batch = jnp.array(batch["image"])
            y_batch = jnp.array(batch["label"])
            params, opt_state, loss = train_step(params, opt_state, X_batch, y_batch)
            epoch_loss += float(loss)
            global_step += 1

        test_acc = float(accuracy(params, test_X, test_y))
        train_acc = float(accuracy(params, X_batch, y_batch))  # last batch proxy

        metrics = {
            "epoch": epoch,
            "loss": epoch_loss / steps_per_epoch,
            "train_acc": train_acc,
            "test_acc": test_acc,
        }
        wandb.log(metrics, step=epoch)
        logger.info(
            "Epoch %2d/%d | loss: %.4f | train_acc: %.4f | test_acc: %.4f",
            epoch,
            num_epochs,
            metrics["loss"],
            train_acc,
            test_acc,
        )

        if epoch % config.checkpoint_every_epochs == 0:
            save_checkpoint(checkpointer, run_dir, epoch, params, opt_state)

    wandb.finish()


if __name__ == "__main__":
    main()
