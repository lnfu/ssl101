import logging
from pathlib import Path

import jax
import jax.numpy as jnp
import optax
import wandb
from absl import app
from flax import nnx
from ml_collections import config_flags

from augmentations import augment_batch, make_augmentation
from checkpoint import make_checkpointer, save_checkpoint
from input_pipeline import create_dataset, get_split_size
from models import SimCLR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

_CONFIG = config_flags.DEFINE_config_file("config", "configs/sgd.py")


def _nt_xent_loss(
    z1: jax.Array, z2: jax.Array, temperature: float
) -> tuple[jax.Array, jax.Array]:
    batch_size = z1.shape[0]
    z1 = z1 / (jnp.linalg.norm(z1, axis=-1, keepdims=True) + 1e-8)
    z2 = z2 / (jnp.linalg.norm(z2, axis=-1, keepdims=True) + 1e-8)
    z = jnp.concatenate([z1, z2], axis=0)  # (2B, D)

    sim = (z @ z.T) / temperature  # (2B, 2B)
    sim = sim - jnp.eye(2 * batch_size) * 1e9  # mask self-similarity

    targets = jnp.concatenate(
        [jnp.arange(batch_size, 2 * batch_size), jnp.arange(0, batch_size)]
    )

    loss = optax.softmax_cross_entropy_with_integer_labels(sim, targets).mean()
    contrastive_acc = jnp.mean(jnp.argmax(sim, axis=-1) == targets)
    return loss, contrastive_acc


def make_train_step(augment_fn, temperature: float):
    @nnx.jit
    def train_step(
        model: SimCLR,
        optimizer: nnx.Optimizer,
        X: jax.Array,
        rng: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        rng1, rng2 = jax.random.split(rng)
        x1 = augment_fn(rng1, X)
        x2 = augment_fn(rng2, X)

        def loss_fn(model: SimCLR) -> tuple[jax.Array, jax.Array]:
            z1 = model(x1)
            z2 = model(x2)
            return _nt_xent_loss(z1, z2, temperature)

        (loss, acc), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)
        optimizer.update(model, grads)
        return loss, acc

    return train_step


def main(_) -> None:
    config = _CONFIG.value

    model = SimCLR(rngs=nnx.Rngs(config.seed))

    steps_per_epoch = get_split_size("unlabeled") // config.batch_size
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

    train_step = make_train_step(augment_fn, float(config.temperature))

    if not config.dry_run:
        wandb.init(project=config.wandb_project, config=config.to_dict())
        assert wandb.run is not None
        run_name = Path(wandb.run.dir).parent.name
        checkpointer, run_dir = make_checkpointer(config.checkpoint_dir, run_name)
    else:
        run_name = "dry_run"
        checkpointer, run_dir = None, None

    logger.info(
        "SimCLR pretraining: %d epochs × %d steps/epoch (batch=%d, temperature=%g)",
        num_epochs,
        steps_per_epoch,
        config.batch_size,
        config.temperature,
    )

    rng = jax.random.key(config.seed)
    for epoch in range(1, num_epochs + 1):
        epoch_loss = 0.0
        epoch_acc = 0.0
        num_steps = 0

        for batch in create_dataset(
            "unlabeled", config.batch_size, seed=config.seed + epoch
        ):
            X_batch = jnp.array(batch["image"])
            rng, step_rng = jax.random.split(rng)
            loss, acc = train_step(model, optimizer, X_batch, step_rng)
            epoch_loss += float(loss)
            epoch_acc += float(acc)
            num_steps += 1

        metrics = {
            "epoch": epoch,
            "loss": epoch_loss / num_steps,
            "contrastive_acc": epoch_acc / num_steps,
        }
        if not config.dry_run:
            wandb.log(metrics, step=epoch)
        logger.info(
            "Epoch %2d/%d | loss: %.4f | contrastive_acc: %.4f",
            epoch,
            num_epochs,
            metrics["loss"],
            metrics["contrastive_acc"],
        )

        if not config.dry_run and epoch % config.checkpoint_every_epochs == 0:
            assert checkpointer is not None and run_dir is not None
            save_checkpoint(checkpointer, run_dir, epoch, model, optimizer)

    if not config.dry_run:
        wandb.finish()


if __name__ == "__main__":
    app.run(main)
