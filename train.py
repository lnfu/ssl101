import logging
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import wandb
from absl import app
from flax import nnx
from ml_collections import config_flags

from augmentations import augment_batch, make_augmentation
from checkpoint import make_checkpointer, save_checkpoint
from input_pipeline import create_dataset, get_split_size
from models import ResNet18

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

_CONFIG = config_flags.DEFINE_config_file("config", "configs/sgd.py")


def make_train_step(augment_fn):
    @nnx.jit
    def train_step(
        model: ResNet18,
        optimizer: nnx.Optimizer,
        X: jax.Array,
        y: jax.Array,
        rng: jax.Array,
    ) -> jax.Array:
        X = augment_fn(rng, X)

        def loss_fn(model: ResNet18) -> jax.Array:
            logits = model(X)
            return optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()

        loss, grads = nnx.value_and_grad(loss_fn)(model)
        optimizer.update(model, grads)
        return loss

    return train_step


@nnx.jit
def compute_accuracy(model: ResNet18, X: jax.Array, y: jax.Array) -> jax.Array:
    logits = model(X, use_running_average=True)
    return jnp.mean(jnp.argmax(logits, axis=-1) == y)


def _load_test_set(batch_size: int) -> tuple[jax.Array, jax.Array]:
    xs, ys = [], []
    for batch in create_dataset("test", batch_size):
        xs.append(batch["image"])
        ys.append(batch["label"])
    return jnp.array(np.concatenate(xs)), jnp.array(np.concatenate(ys))


def main(_) -> None:
    config = _CONFIG.value

    model = ResNet18(num_classes=config.num_classes, rngs=nnx.Rngs(config.seed))

    steps_per_epoch = get_split_size("train") // config.batch_size
    num_epochs = int(config.num_epochs)
    warmup_steps = int(config.warmup_epochs * steps_per_epoch)

    lr_schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=config.learning_rate,
        warmup_steps=warmup_steps,
        decay_steps=steps_per_epoch * num_epochs,
    )
    if config.optimizer == "sgd":
        tx = optax.chain(
            optax.add_decayed_weights(config.weight_decay),
            optax.sgd(lr_schedule, momentum=config.momentum),
        )
    elif config.optimizer == "adamw":
        tx = optax.adamw(lr_schedule, weight_decay=config.weight_decay)
    else:
        raise ValueError(f"Unknown optimizer {config.optimizer!r}")
    optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)

    if config.augment:
        chain = make_augmentation()

        def augment_fn(rng: jax.Array, x: jax.Array) -> jax.Array:
            return augment_batch(chain, rng, x)
    else:

        def augment_fn(rng: jax.Array, x: jax.Array) -> jax.Array:
            return x

    train_step = make_train_step(augment_fn)

    if not config.dry_run:
        wandb.init(project=config.wandb_project, config=config.to_dict())
        run_name = Path(wandb.run.dir).parent.name
        checkpointer, run_dir = make_checkpointer(config.checkpoint_dir, run_name)
    else:
        run_name = "dry_run"
        checkpointer, run_dir = None, None

    logger.info("Loading test set...")
    test_X, test_y = _load_test_set(config.batch_size)

    logger.info(
        "Training: %d epochs × %d steps/epoch (batch=%d, augment=%s)",
        num_epochs,
        steps_per_epoch,
        config.batch_size,
        config.augment,
    )

    rng = jax.random.key(config.seed)
    for epoch in range(1, num_epochs + 1):
        epoch_loss = 0.0
        num_steps = 0

        for batch in create_dataset(
            "train", config.batch_size, seed=config.seed + epoch
        ):
            X_batch = jnp.array(batch["image"])
            y_batch = jnp.array(batch["label"])
            rng, step_rng = jax.random.split(rng)
            loss = train_step(model, optimizer, X_batch, y_batch, step_rng)
            epoch_loss += float(loss)
            num_steps += 1

        test_acc = float(compute_accuracy(model, test_X, test_y))
        train_acc = float(compute_accuracy(model, X_batch, y_batch))

        metrics = {
            "epoch": epoch,
            "loss": epoch_loss / num_steps,
            "train_acc": train_acc,
            "test_acc": test_acc,
        }
        if not config.dry_run:
            wandb.log(metrics, step=epoch)
        logger.info(
            "Epoch %2d/%d | loss: %.4f | train_acc: %.4f | test_acc: %.4f",
            epoch,
            num_epochs,
            metrics["loss"],
            train_acc,
            test_acc,
        )

        if not config.dry_run and epoch % config.checkpoint_every_epochs == 0:
            save_checkpoint(checkpointer, run_dir, epoch, model, optimizer)

    if not config.dry_run:
        wandb.finish()


if __name__ == "__main__":
    app.run(main)
