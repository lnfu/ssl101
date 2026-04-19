import logging
import os
import pickle
import tarfile
import urllib.request
from collections.abc import Iterator

import numpy as np

logger = logging.getLogger(__name__)

_CIFAR10_TRAIN_SIZE = 50000
_CIFAR10_TEST_SIZE = 10000
_CIFAR10_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
_CIFAR10_DIR = os.path.join(os.path.dirname(__file__), "data")

_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}


def _download_cifar10() -> str:
    os.makedirs(_CIFAR10_DIR, exist_ok=True)
    dest = os.path.join(_CIFAR10_DIR, "cifar-10-python.tar.gz")
    if not os.path.exists(os.path.join(_CIFAR10_DIR, "cifar-10-batches-py")):
        logger.info("Downloading CIFAR-10 (~170 MB) to %s ...", _CIFAR10_DIR)
        urllib.request.urlretrieve(_CIFAR10_URL, dest)
        with tarfile.open(dest) as f:
            f.extractall(_CIFAR10_DIR)
    return os.path.join(_CIFAR10_DIR, "cifar-10-batches-py")


def _load_batch(path: str) -> tuple[np.ndarray, np.ndarray]:
    with open(path, "rb") as f:
        d = pickle.load(f, encoding="bytes")
    X = (
        d[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1).astype(np.float32)
        / 255.0
    )
    y = np.array(d[b"labels"], dtype=np.int32)
    return X, y


def _load_cifar10_split(split: str) -> tuple[np.ndarray, np.ndarray]:
    if split in _cache:
        return _cache[split]

    data_dir = _download_cifar10()
    if split == "train":
        batches = [os.path.join(data_dir, f"data_batch_{i}") for i in range(1, 6)]
        parts = [_load_batch(b) for b in batches]
        X = np.concatenate([p[0] for p in parts])
        y = np.concatenate([p[1] for p in parts])
    else:
        X, y = _load_batch(os.path.join(data_dir, "test_batch"))

    _cache[split] = (X, y)
    return _cache[split]


def create_dataset(
    split: str,
    batch_size: int,
    seed: int = 0,
) -> Iterator[dict[str, np.ndarray]]:
    if split not in ("train", "test"):
        raise ValueError(f"Unknown split {split!r}. Expected 'train' or 'test'.")

    X, y = _load_cifar10_split(split)

    indices = np.random.default_rng(seed).permutation(len(X))
    X, y = X[indices], y[indices]

    for start in range(0, len(X), batch_size):
        yield {
            "image": X[start : start + batch_size],
            "label": y[start : start + batch_size],
        }


def get_split_size(split: str) -> int:
    if split == "train":
        return _CIFAR10_TRAIN_SIZE
    if split == "test":
        return _CIFAR10_TEST_SIZE
    raise ValueError(f"Unknown split {split!r}. Expected 'train' or 'test'.")
