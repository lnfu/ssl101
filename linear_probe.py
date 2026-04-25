import logging
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import optax
import orbax.checkpoint as ocp
import wandb
from absl import app
from flax import nnx
from ml_collections import config_flags

from augmentations import (
    augment_batch,
    make_normalize_only,
    make_probe_train_augmentation,
)
from checkpoint import restore_checkpoint
from input_pipeline import compute_num_steps, create_dataset
from models import LinearClassifier, SimCLR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

_CONFIG = config_flags.DEFINE_config_file("config", "configs/linear_probe.py")


def make_train_step(augment_fn):
    @nnx.jit
    def train_step(
        model: LinearClassifier,
        optimizer: nnx.Optimizer,
        X: jax.Array,
        y: jax.Array,
        rng: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        X = augment_fn(rng, X)

        def loss_fn(model: LinearClassifier) -> tuple[jax.Array, jax.Array]:
            logits = model(X)
            loss = optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()
            acc = jnp.mean(jnp.argmax(logits, axis=-1) == y)
            return loss, acc

        (loss, acc), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)
        optimizer.update(model, grads)
        return loss, acc

    return train_step


def make_eval_step(augment_fn):
    @nnx.jit
    def eval_step(
        model: LinearClassifier,
        X: jax.Array,
        y: jax.Array,
        rng: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        # `rng` is unused by the eval augmentation (normalize only) but kept
        # in the signature for interface symmetry with train_step.
        X = augment_fn(rng, X)
        logits = model(X)
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()
        acc = jnp.mean(jnp.argmax(logits, axis=-1) == y)
        return loss, acc

    return eval_step


def main(_) -> None:
    config = _CONFIG.value

    if config.pretrain_run_name is None or config.pretrain_epoch is None:
        raise ValueError(
            "Must set --config.pretrain_run_name=<run> and "
            "--config.pretrain_epoch=<epoch>"
        )

    rngs = nnx.Rngs(config.seed)

    simclr = SimCLR(rngs=rngs)
    pretrain_dir = str(
        Path(config.pretrain_checkpoint_dir).resolve() / config.pretrain_run_name
    )
    checkpointer = ocp.StandardCheckpointer()
    restored_epoch = restore_checkpoint(
        checkpointer, pretrain_dir, config.pretrain_epoch, simclr
    )
    logger.info(
        "Loaded SimCLR checkpoint: %s @ epoch %d",
        config.pretrain_run_name,
        restored_epoch,
    )

    model = LinearClassifier(
        simclr.backbone,
        num_classes=config.num_classes,
        rngs=rngs,
        freeze_backbone=True,
    )

    # drop_remainder=True for train (stable JIT shapes), False for eval (cover
    # every test sample).
    steps_per_epoch = compute_num_steps("train", config.batch_size, drop_remainder=True)
    num_epochs = int(config.num_epochs)
    decay_steps = steps_per_epoch * num_epochs

    if config.warmup_epochs > 0:
        warmup_steps = int(config.warmup_epochs * steps_per_epoch)
        lr_schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=config.learning_rate,
            warmup_steps=warmup_steps,
            decay_steps=decay_steps,
        )
    else:
        lr_schedule = optax.cosine_decay_schedule(
            init_value=config.learning_rate,
            decay_steps=decay_steps,
        )

    tx = optax.sgd(lr_schedule, momentum=config.momentum)
    optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)

    train_aug_fn = partial(augment_batch, make_probe_train_augmentation())
    eval_aug_fn = partial(augment_batch, make_normalize_only())

    train_step = make_train_step(train_aug_fn)
    eval_step = make_eval_step(eval_aug_fn)

    if not config.dry_run:
        wandb.init(
            project=config.wandb_project,
            config=config.to_dict(),
            job_type="linear_probe",
        )

    logger.info(
        "Linear probe: %d epochs × %d steps/epoch (batch=%d, lr=%g)",
        num_epochs,
        steps_per_epoch,
        config.batch_size,
        config.learning_rate,
    )

    rng = jax.random.key(config.seed)
    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc, train_steps = 0.0, 0.0, 0
        for batch in create_dataset(
            "train",
            config.batch_size,
            seed=config.seed + epoch,
            drop_remainder=True,
        ):
            X = jnp.array(batch["image"])
            y = jnp.array(batch["label"])
            rng, step_rng = jax.random.split(rng)
            loss, acc = train_step(model, optimizer, X, y, step_rng)
            train_loss += float(loss)
            train_acc += float(acc)
            train_steps += 1

        eval_loss, eval_acc, eval_steps = 0.0, 0.0, 0
        for batch in create_dataset(
            "test",
            config.batch_size,
            seed=0,
            shuffle=False,
            drop_remainder=False,
        ):
            X = jnp.array(batch["image"])
            y = jnp.array(batch["label"])
            rng, step_rng = jax.random.split(rng)
            loss, acc = eval_step(model, X, y, step_rng)
            eval_loss += float(loss)
            eval_acc += float(acc)
            eval_steps += 1

        metrics = {
            "epoch": epoch,
            "train/loss": train_loss / train_steps,
            "train/acc": train_acc / train_steps,
            "test/loss": eval_loss / eval_steps,
            "test/acc": eval_acc / eval_steps,
        }
        if not config.dry_run:
            wandb.log(metrics, step=epoch)
        logger.info(
            "Epoch %2d/%d | train_loss: %.4f acc: %.4f | test_loss: %.4f acc: %.4f",
            epoch,
            num_epochs,
            metrics["train/loss"],
            metrics["train/acc"],
            metrics["test/loss"],
            metrics["test/acc"],
        )

    if not config.dry_run:
        wandb.finish()


if __name__ == "__main__":
    app.run(main)
