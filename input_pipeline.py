import functools
import logging
from pathlib import Path
from typing import Iterator

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

_CACHE_DIR = Path(__file__).parent / ".cache"
_TEST_SIZE = 0.1


@functools.lru_cache(maxsize=1)
def _load_mnist() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load MNIST, using local disk cache if available.

    Returns:
        (X_train, X_test, y_train, y_test) with pixel values in [0, 1].
    """
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_x = _CACHE_DIR / "mnist_x.npy"
    cache_y = _CACHE_DIR / "mnist_y.npy"

    if cache_x.exists() and cache_y.exists():
        logger.info("Loading MNIST from cache: %s", _CACHE_DIR)
        X = np.load(cache_x)
        y = np.load(cache_y)
    else:
        logger.info("Downloading MNIST from OpenML...")
        mnist = fetch_openml("mnist_784", version=1, as_frame=False)
        X = mnist.data.astype(np.float32) / 255.0
        y = mnist.target.astype(np.int32)
        np.save(cache_x, X)
        np.save(cache_y, y)
        logger.info("MNIST cached to %s", _CACHE_DIR)

    return train_test_split(X, y, test_size=_TEST_SIZE, random_state=0)


def create_dataset(
    split: str,
    batch_size: int,
    seed: int = 0,
) -> Iterator[dict]:
    """Yield batched MNIST samples for the given split.

    The training split cycles indefinitely with per-epoch shuffling.
    The test split yields exactly one pass through the data.

    Args:
        split: One of ``"train"`` or ``"test"``.
        batch_size: Number of samples per batch. The last incomplete batch
            of each epoch is dropped.
        seed: PRNG seed for shuffle randomness (train split only).

    Yields:
        Dicts with keys:
            - ``"image"``: ``float32`` array of shape ``(B, 784)``
            - ``"label"``: ``int32`` array of shape ``(B,)``

    Raises:
        ValueError: If ``split`` is not ``"train"`` or ``"test"``.
    """
    if split not in ("train", "test"):
        raise ValueError(f"split must be 'train' or 'test', got {split!r}")

    X_train, X_test, y_train, y_test = _load_mnist()
    X = X_train if split == "train" else X_test  # already (N, 784)
    y = y_train if split == "train" else y_test

    n = len(X)
    rng = np.random.default_rng(seed)

    while True:
        indices = rng.permutation(n) if split == "train" else np.arange(n)

        for start in range(0, n - batch_size + 1, batch_size):
            batch_idx = indices[start : start + batch_size]
            yield {
                "image": X[batch_idx],
                "label": y[batch_idx],
            }

        if split != "train":
            break


def get_split_size(split: str) -> int:
    """Return the number of examples in the given split.

    Args:
        split: One of ``"train"`` or ``"test"``.

    Returns:
        Number of examples.
    """
    X_train, X_test, _, _ = _load_mnist()
    return len(X_train) if split == "train" else len(X_test)
