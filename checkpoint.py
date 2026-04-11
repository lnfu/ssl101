import logging
from pathlib import Path

import optax
import orbax.checkpoint as ocp

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
    params: dict,
    opt_state: optax.OptState,
) -> None:
    path = str(Path(run_dir) / f"epoch_{epoch:06d}")
    checkpointer.save(
        path,
        {
            "params": params,
            "opt_state": opt_state,
            "epoch": epoch,
        },
    )
    logger.info("Checkpoint saved: %s", path)


def restore_checkpoint(
    checkpointer: ocp.StandardCheckpointer,
    run_dir: str,
    epoch: int,
    params: dict,
    opt_state: optax.OptState,
) -> tuple[dict, optax.OptState, int]:
    path = str(Path(run_dir) / f"epoch_{epoch:06d}")
    restored = checkpointer.restore(
        path,
        {
            "params": params,
            "opt_state": opt_state,
            "epoch": 0,
        },
    )
    return restored["params"], restored["opt_state"], restored["epoch"]
