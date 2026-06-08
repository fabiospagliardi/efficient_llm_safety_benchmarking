import json
from pathlib import Path

import numpy as np


def load_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(records: list[dict], path: Path) -> None:
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def split_fold(
    records: list[dict],
    n_folds: int,
    fold_idx: int,
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    """Return (train, test) for the given fold index out of n_folds."""
    if not (0 <= fold_idx < n_folds):
        raise ValueError(f"fold_idx must be in [0, {n_folds - 1}], got {fold_idx}")

    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(records)).tolist()

    fold_sizes = [len(records) // n_folds] * n_folds
    for i in range(len(records) % n_folds):
        fold_sizes[i] += 1

    folds: list[list[int]] = []
    start = 0
    for size in fold_sizes:
        folds.append(indices[start : start + size])
        start += size

    test_idx = set(folds[fold_idx])
    train = [records[i] for i in range(len(records)) if i not in test_idx]
    test = [records[i] for i in test_idx]
    return train, test


def split_random_sample(
    records: list[dict],
    sample_idx: int,
    test_size: int,
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    """Return (train, test) for a single reproducible random draw identified by sample_idx."""
    if test_size >= len(records):
        raise ValueError(f"test_size ({test_size}) must be smaller than len(records) ({len(records)})")

    rng = np.random.default_rng((seed, sample_idx))
    test_idx = set(rng.choice(len(records), size=test_size, replace=False).tolist())
    test = [records[i] for i in test_idx]
    train = [records[i] for i in range(len(records)) if i not in test_idx]
    return train, test


def split_random_samples(
    records: list[dict],
    n_samples: int,
    test_size: int = 10,
    seed: int = 42,
) -> list[tuple[list[dict], list[dict]]]:
    """Return a list of (train, test) tuples from random subsampling without replacement."""
    if test_size >= len(records):
        raise ValueError(f"test_size ({test_size}) must be smaller than len(records) ({len(records)})")

    rng = np.random.default_rng(seed)
    splits = []
    for _ in range(n_samples):
        test_idx = set(rng.choice(len(records), size=test_size, replace=False).tolist())
        test = [records[i] for i in test_idx]
        train = [records[i] for i in range(len(records)) if i not in test_idx]
        splits.append((train, test))
    return splits
