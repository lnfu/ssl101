import logging
import os
import tarfile
import urllib.request
from collections.abc import Iterator

import numpy as np

logger = logging.getLogger(__name__)

_STL10_TRAIN_SIZE = 5000
_STL10_TEST_SIZE = 8000
_STL10_UNLABELED_SIZE = 100000
_STL10_URL = "https://ai.stanford.edu/~acoates/stl10/stl10_binary.tar.gz"
_STL10_DIR = os.path.join(os.path.dirname(__file__), "data")

_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}


def _download_stl10() -> str:
    os.makedirs(_STL10_DIR, exist_ok=True)
    dest = os.path.join(_STL10_DIR, "stl10_binary.tar.gz")
    if not os.path.exists(os.path.join(_STL10_DIR, "stl10_binary")):
        logger.info("Downloading STL-10 (~2.5 GB) to %s ...", _STL10_DIR)
        urllib.request.urlretrieve(_STL10_URL, dest)
        with tarfile.open(dest) as f:
            f.extractall(_STL10_DIR)
    return os.path.join(_STL10_DIR, "stl10_binary")


def _load_images(path: str) -> np.ndarray:
    with open(path, "rb") as f:
        data = np.fromfile(f, dtype=np.uint8)
    # STL-10 stores images column-major: reshape as (N, 3, 96, 96) then
    # transpose (0, 3, 2, 1) to get (N, 96, 96, 3) in row-major RGB.
    return data.reshape(-1, 3, 96, 96).transpose(0, 3, 2, 1).astype(np.float32) / 255.0


def _load_labels(path: str) -> np.ndarray:
    with open(path, "rb") as f:
        # STL-10 labels are 1-indexed (1..10); shift to 0..9.
        return np.fromfile(f, dtype=np.uint8).astype(np.int32) - 1


def _load_stl10_split(split: str) -> tuple[np.ndarray, np.ndarray]:
    if split in _cache:
        return _cache[split]

    data_dir = _download_stl10()
    if split == "train":
        X = _load_images(os.path.join(data_dir, "train_X.bin"))
        y = _load_labels(os.path.join(data_dir, "train_y.bin"))
    elif split == "test":
        X = _load_images(os.path.join(data_dir, "test_X.bin"))
        y = _load_labels(os.path.join(data_dir, "test_y.bin"))
    elif split == "unlabeled":
        X = _load_images(os.path.join(data_dir, "unlabeled_X.bin"))
        y = np.full(len(X), -1, dtype=np.int32)
    else:
        raise ValueError(
            f"Unknown split {split!r}. Expected 'train', 'test', or 'unlabeled'."
        )

    _cache[split] = (X, y)
    return _cache[split]


def create_dataset(
    split: str,
    batch_size: int,
    seed: int = 0,
) -> Iterator[dict[str, np.ndarray]]:
    if split not in ("train", "test", "unlabeled"):
        raise ValueError(
            f"Unknown split {split!r}. Expected 'train', 'test', or 'unlabeled'."
        )

    X, y = _load_stl10_split(split)

    indices = np.random.default_rng(seed).permutation(len(X))
    X, y = X[indices], y[indices]

    for start in range(0, len(X), batch_size):
        yield {
            "image": X[start : start + batch_size],
            "label": y[start : start + batch_size],
        }


def get_split_size(split: str) -> int:
    if split == "train":
        return _STL10_TRAIN_SIZE
    if split == "test":
        return _STL10_TEST_SIZE
    if split == "unlabeled":
        return _STL10_UNLABELED_SIZE
    raise ValueError(
        f"Unknown split {split!r}. Expected 'train', 'test', or 'unlabeled'."
    )
