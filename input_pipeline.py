import logging
import tarfile
import urllib.request
from functools import lru_cache
from pathlib import Path

import grain.python as grain
import numpy as np

logger = logging.getLogger(__name__)

_STL10_URL = "https://ai.stanford.edu/~acoates/stl10/stl10_binary.tar.gz"
_STL10_DIR = Path(__file__).parent / "data"

_VALID_SPLITS = ("train", "test", "unlabeled")
_SPLIT_SIZES = {
    "train": 5000,
    "test": 8000,
    "unlabeled": 100000,
}


def _validate_split(split: str) -> None:
    if split not in _VALID_SPLITS:
        raise ValueError(f"Unknown split {split!r}. Expected one of {_VALID_SPLITS}.")


def _download_stl10() -> Path:
    _STL10_DIR.mkdir(parents=True, exist_ok=True)
    archive = _STL10_DIR / "stl10_binary.tar.gz"
    extracted = _STL10_DIR / "stl10_binary"
    if not extracted.exists():
        logger.info("Downloading STL-10 (~2.5 GB) to %s ...", _STL10_DIR)
        urllib.request.urlretrieve(_STL10_URL, archive)
        with tarfile.open(archive) as f:
            f.extractall(_STL10_DIR)
    return extracted


def _load_images(path: Path) -> np.ndarray:
    # STL-10 stores images column-major as raw uint8. We keep storage as uint8
    # (~4x memory savings over float32) and cast to float32 per-sample in the
    # data source's __getitem__.
    data = np.fromfile(path, dtype=np.uint8)
    return data.reshape(-1, 3, 96, 96).transpose(0, 3, 2, 1).copy()


def _load_labels(path: Path) -> np.ndarray:
    # STL-10 labels are 1-indexed (1..10); shift to 0..9.
    return np.fromfile(path, dtype=np.uint8).astype(np.int32) - 1


class STL10Source:
    """Grain RandomAccessDataSource for an STL-10 split.

    Images are held in memory as uint8; labels as int32. Float32 conversion
    happens per-sample in __getitem__.
    """

    def __init__(self, split: str) -> None:
        _validate_split(split)
        self._split = split
        data_dir = _download_stl10()
        if split == "unlabeled":
            self._images = _load_images(data_dir / "unlabeled_X.bin")
            self._labels = np.full(len(self._images), -1, dtype=np.int32)
        else:
            self._images = _load_images(data_dir / f"{split}_X.bin")
            self._labels = _load_labels(data_dir / f"{split}_y.bin")

    def __len__(self) -> int:
        return len(self._images)

    def __getitem__(self, index: int) -> dict[str, np.ndarray]:
        return {
            "image": self._images[index].astype(np.float32) / 255.0,
            "label": self._labels[index],
        }

    def __repr__(self) -> str:
        # Required by grain for checkpointing support.
        return f"STL10Source(split={self._split!r})"


@lru_cache(maxsize=3)
def _get_source(split: str) -> STL10Source:
    return STL10Source(split)


def create_dataset(
    split: str,
    batch_size: int,
    *,
    seed: int = 0,
    shuffle: bool = True,
    drop_remainder: bool = False,
) -> grain.DataLoader:
    _validate_split(split)
    source = _get_source(split)
    sampler = grain.IndexSampler(
        num_records=len(source),
        shuffle=shuffle,
        seed=seed,
        shard_options=grain.NoSharding(),
        num_epochs=1,
    )
    return grain.DataLoader(
        data_source=source,
        sampler=sampler,
        operations=[
            grain.Batch(batch_size=batch_size, drop_remainder=drop_remainder),
        ],
        worker_count=0,
    )


def get_split_size(split: str) -> int:
    _validate_split(split)
    return _SPLIT_SIZES[split]


def compute_num_steps(
    split: str, batch_size: int, *, drop_remainder: bool = True
) -> int:
    """Exact number of batches ``create_dataset`` will yield for one epoch.

    create_dataset does not drop the incomplete final batch by default, so
    plain floor division (``size // batch_size``) understates the true step
    count whenever the split size is not a multiple of batch_size.
    """
    size = get_split_size(split)
    if drop_remainder:
        return size // batch_size
    return -(-size // batch_size)  # ceil division
