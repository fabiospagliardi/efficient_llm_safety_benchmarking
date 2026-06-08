from __future__ import annotations

import numpy as np
import pandas as pd

from subset_selection.evaluation import evaluate_accuracy, evaluate_items_sweep, build_response_matrix, evaluate_abilities
# from static_subset.subset_predictors import RandomForestSubsetPredictor


def select_items_disco(train_df: pd.DataFrame, parameters_df: pd.DataFrame) -> list[int]:
    """
    Rank items by binary entropy of empirical difficulty (DISCO).

    For each item, compute p_i = fraction of training models that answered correctly,
    then H_i = -p_i * log(p_i) - (1 - p_i) * log(1 - p_i).

    Items are returned ordered by H_i descending (highest entropy first).
    Returns an ordered list of integer positions into parameters_df.
    """
    item_ids = parameters_df["item_ids"]

    # Compute empirical correctness
    p = (
        train_df
        .assign(item_id=lambda x: x["item_id"].astype(str))
        .groupby("item_id")["score"]
        .mean()
        .rename("p")
    )

    # Preserve original positional index explicitly
    df = (
        parameters_df
        .reset_index()  # creates column "index" = original position
        .rename(columns={"index": "pos"})
        .assign(item_ids=lambda x: x["item_ids"].astype(str))
        .merge(p, left_on="item_ids", right_index=True, how="left")
    )

    # Handle missing items
    df["p"] = df["p"].fillna(0.5)

    p_clipped = np.clip(df["p"], 1e-12, 1 - 1e-12)
    df["entropy"] = -p_clipped * np.log(p_clipped) - (1 - p_clipped) * np.log(1 - p_clipped)

    # Return original positions sorted by entropy
    return df.sort_values("entropy", ascending=False)["pos"].tolist()


class DiscoSubsetMethod:
    """
    High-level API for DISCO item selection + RF-predicted accuracy.

    Ranks items by binary entropy of empirical difficulty, then estimates
    full-dataset accuracy via a Random Forest trained on per-item signatures.

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
    ) -> "DiscoSubsetMethod":
        self.ranked_indices = {
            "disco": select_items_disco(train_df, parameters_df),
        }
        self._ac_train, _ = evaluate_accuracy(train_df, len(parameters_df))
        self._ab_train, _ = evaluate_abilities(train_df, parameters_df)
        self._train_matrices = {
            method: build_response_matrix(train_df, indices, parameters_df, self.max_items)[0]
            for method, indices in self.ranked_indices.items()
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

        rows = []
        for method, indices in self.ranked_indices.items():
            ab, ab_std, ac, ac_std, k_vals = evaluate_items_sweep(
                indices, test_df, parameters_df, self.max_items, self.n_points
            )
            test_matrix, _ = build_response_matrix(test_df, indices, parameters_df, self.max_items)
            train_matrix = self._train_matrices[method]

            for k_idx, K in enumerate(k_vals):
                # predictor = RandomForestSubsetPredictor().fit(train_matrix[:, :K], self._ac_train)
                # predicted_ac = predictor.predict(test_matrix[:, :K])

                # predictor_ab = RandomForestSubsetPredictor().fit(train_matrix[:, :K], self._ab_train)
                # predicted_ab = predictor_ab.predict(test_matrix[:, :K])

                for m_idx, model in enumerate(models_test):
                    rows.append({
                        "fold": fold_idx, "method": method, "K": K,
                        "model": model, "split": "test",
                        "ability": ab[m_idx, k_idx], "ability_std": ab_std[m_idx, k_idx],
                        "accuracy": ac[m_idx, k_idx], "accuracy_std": ac_std[m_idx, k_idx],
                        # "accuracy_pred": predicted_ac[m_idx], "ability_pred": predicted_ab[m_idx],
                    })

        return rows