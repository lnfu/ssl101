import logging
from pathlib import Path

import orbax.checkpoint as ocp
from flax import nnx

logger = logging.getLogger(__name__)


def make_checkpointer(
    checkpoint_dir: str, run_name: str
) -> tuple[ocp.StandardCheckpointer, str]:
    run_dir = str(Path(checkpoint_dir).resolve() / run_name)
    Path(run_dir).mkdir(parents=True, exist_ok=True)
    return ocp.StandardCheckpointer(), run_dir


def save_checkpoint(
    checkpointer: ocp.StandardCheckpointer,
    run_dir: str,
    epoch: int,
    model: nnx.Module,
    optimizer: nnx.Optimizer,
) -> None:
    path = str(Path(run_dir) / f"epoch_{epoch:06d}")
    _, model_state = nnx.split(model)
    checkpointer.save(path, {"model_state": model_state, "epoch": epoch})
    logger.info("Checkpoint saved: %s", path)


def restore_checkpoint(
    checkpointer: ocp.StandardCheckpointer,
    run_dir: str,
    epoch: int,
    model: nnx.Module,
) -> int:
    path = str(Path(run_dir) / f"epoch_{epoch:06d}")
    _, state = nnx.split(model)
    restored = checkpointer.restore(path, {"model_state": state, "epoch": 0})
    nnx.update(model, restored["model_state"])
    return int(restored["epoch"])
