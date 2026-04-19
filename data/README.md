# Data

## Auto download

Run any training script and the data will be downloaded here automatically.

## Manual download

1. Download from https://ai.stanford.edu/~acoates/stl10/stl10_binary.tar.gz
2. Extract into this folder so the layout looks like:

```
data/
└── stl10_binary/
    ├── train_X.bin
    ├── train_y.bin
    ├── test_X.bin
    ├── test_y.bin
    ├── unlabeled_X.bin
    ├── class_names.txt
    └── fold_indices.txt
```
