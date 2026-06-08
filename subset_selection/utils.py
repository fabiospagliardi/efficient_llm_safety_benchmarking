from __future__ import annotations

import numpy as np
import pandas as pd

from subset_selection.evaluation import evaluate_accuracy, evaluate_abilities, evaluate_items_sweep, build_response_matrix
# from static_subset.subset_predictors import RandomForestSubsetPredictor

def select_items_ranked(parameters_df, method, n_quantiles=None, random_state=None) -> list[int]:
    """
    Return an ordered list of integer positions into parameters_df (best item first)
    according to the specified selection method.

    Supported methods:
      "total_fisher"           – sort by total_fisher descending
      "marginal_fisher"        – sort by marginal_fisher_contribution descending
      "total_fisher_bquant"    – total_fisher descending, then round-robin by b quantile
      "marginal_fisher_bquant" – marginal_fisher_contribution descending, round-robin by b quantile
      "random"                 – random shuffle (requires random_state)
    """
    METHOD_COLUMN = {
        "total_fisher":           ("total_fisher", False),
        "marginal_fisher":        ("marginal_fisher_contribution", False),
        "total_fisher_bquant":    ("total_fisher", False),
        "marginal_fisher_bquant": ("marginal_fisher_contribution", False),
    }

    if method == "random":
        rng = np.random.default_rng(random_state)
        positions = list(range(len(parameters_df)))
        rng.shuffle(positions)
        return positions

    if method not in METHOD_COLUMN:
        raise ValueError(f"Unknown method: {method!r}")

    sort_column, ascending = METHOD_COLUMN[method]
    df = parameters_df.reset_index(drop=True)
    sorted_df = df.sort_values(sort_column, ascending=ascending)

    if method.endswith("_bquant"):
        if n_quantiles is None:
            raise ValueError(f"n_quantiles is required for method {method!r}")
        sorted_df = sorted_df.copy()
        sorted_df["_quantile"] = pd.qcut(sorted_df["b"], q=n_quantiles, labels=False, duplicates="drop")
        groups = [group.index.tolist() for _, group in sorted_df.groupby("_quantile", sort=True)]
        max_len = max(len(g) for g in groups)
        reordered = [g[pos] for pos in range(max_len) for g in groups if pos < len(g)]
        return reordered

    return sorted_df.index.tolist()


class RandomMethod:
    """
    High-level API for random item selection + RF-predicted accuracy.

    Serves as a null-hypothesis baseline: items are shuffled randomly using
    fold_idx as the seed so each fold gets a different but reproducible shuffle.

    fit() selects items and prepares train model signatures.
    evaluate() runs the per-K sweep and RF-predicted accuracy estimation.

    After fit(), ranked_indices is available for items_fold_df rank columns.
    """

    def __init__(self, max_items: int, n_points: int) -> None:
        self.max_items = max_items
        self.n_points = n_points
        self.ranked_indices: dict[str, list[int]] | None = None
        self._train_matrices: dict[str, np.ndarray] | None = None
        self._ac_train: np.ndarray | None = None
        self._ab_train: np.ndarray | None = None

    def fit(
        self,
        train_df: pd.DataFrame,
        parameters_df: pd.DataFrame,
        fold_idx: int = 0,
    ) -> "RandomMethod":
        self.ranked_indices = {
            "random": select_items_ranked(parameters_df, "random", random_state=fold_idx),
        }
        self._ac_train, _ = evaluate_accuracy(train_df, len(parameters_df))
        self._ab_train, _ = evaluate_abilities(train_df, parameters_df)
        self._train_matrices = {
            "random": build_response_matrix(train_df, self.ranked_indices["random"], parameters_df, self.max_items)[0]
        }
        return self

    def evaluate(
        self,
        test_df: pd.DataFrame,
        parameters_df: pd.DataFrame,
        models_test: list[str],
        fold_idx: int,
    ) -> list[dict]:
        if self.ranked_indices is None:
            raise RuntimeError("Call fit() before evaluate().")

        indices = self.ranked_indices["random"]
        ab, ab_std, ac, ac_std, k_vals = evaluate_items_sweep(
            indices, test_df, parameters_df, self.max_items, self.n_points
        )
        test_matrix, _ = build_response_matrix(test_df, indices, parameters_df, self.max_items)
        train_matrix = self._train_matrices["random"]

        rows = []
        for k_idx, K in enumerate(k_vals):
            # predictor_ac = RandomForestSubsetPredictor().fit(train_matrix[:, :K], self._ac_train)
            # predicted_ac = predictor_ac.predict(test_matrix[:, :K])
            # predictor_ab = RandomForestSubsetPredictor().fit(train_matrix[:, :K], self._ab_train)
            # predicted_ab = predictor_ab.predict(test_matrix[:, :K])
            for m_idx, model in enumerate(models_test):
                rows.append({
                    "fold": fold_idx, "method": "random", "K": K,
                    "model": model, "split": "test",
                    "ability": ab[m_idx, k_idx], "ability_std": ab_std[m_idx, k_idx],
                    "accuracy": ac[m_idx, k_idx], "accuracy_std": ac_std[m_idx, k_idx],
                    # "accuracy_pred": predicted_ac[m_idx], "ability_pred": predicted_ab[m_idx],
                })

        return rows