import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from subset_selection.split import load_jsonl, write_jsonl, split_fold


ROOT = Path(__file__).resolve().parents[1]
IRT_SCRIPT = ROOT / "irt-fit" / "fit_irt_model.py"


def build_eval_df(records: list[dict]) -> pd.DataFrame:  # columns: model, item_id, score
    rows = [
        {
            "model": record["subject_id"],
            "item_id": item_id,
            "score": score,
        }
        for record in records
        for item_id, score in record["responses"].items()
    ]
    return pd.DataFrame(rows)


FitResult = tuple[pd.DataFrame, pd.DataFrame]  # (parameters, abilities)


def fit_irt(
    train_records: list[dict],
    fold_tag: str = "fold",
    device: str = "cpu",
    seed: int = 0,
    epochs: int = 1000,
) -> FitResult:
    fit_results_dir = IRT_SCRIPT.parent / "fit_results"
    with tempfile.TemporaryDirectory() as td:
        tmp_path = Path(td) / f"{fold_tag}_data.jsonl"
        write_jsonl(train_records, tmp_path)
        _run_fit(tmp_path, device=device, seed=seed, epochs=epochs)

    stem = tmp_path.stem
    parameters_path = fit_results_dir / f"{stem.replace('data', 'parameters')}.csv"
    abilities_path  = fit_results_dir / f"{stem.replace('data', 'abilities')}.csv"

    return pd.read_csv(parameters_path), pd.read_csv(abilities_path)


def _run_fit(input_path: Path, device: str, seed: int, epochs: int) -> subprocess.CompletedProcess:
    cmd = [
        sys.executable,
        str(IRT_SCRIPT),
        "--input_path", str(input_path),
        "--device", device,
        "--seed", str(seed),
        "--epochs", str(epochs),
    ]
    return subprocess.run(cmd, cwd=IRT_SCRIPT.parent, check=True)


def run_cross_validation(
    data_path: Path,
    n_folds: int = 5,
    fold_indices: Optional[list[int]] = None,
    device: str = "cpu",
    seed: int = 42,
    epochs: int = 1000,
    output_root: Optional[Path] = None,
) -> list[dict]:
    records = load_jsonl(data_path)
    fold_indices = fold_indices if fold_indices is not None else list(range(n_folds))

    results = []
    for fold_idx in fold_indices:
        train, test = split_fold(records, n_folds=n_folds, fold_idx=fold_idx, seed=seed)
        print(
            f"[fold {fold_idx}/{n_folds}] "
            f"train={len(train)} subjects, test={len(test)} subjects"
        )

        fold_out = (output_root / f"fold_{fold_idx}") if output_root else None
        completed = fit_irt(
            train,
            output_dir=fold_out,
            device=device,
            seed=seed,
            epochs=epochs,
        )
        results.append(
            {
                "fold_idx": fold_idx,
                "n_train": len(train),
                "n_test": len(test),
                "returncode": completed.returncode,
            }
        )

    return results
