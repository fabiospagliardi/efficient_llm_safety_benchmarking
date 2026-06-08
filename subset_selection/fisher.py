from __future__ import annotations

import numpy as np
import pandas as pd
import tqdm

from subset_selection.evaluation import evaluate_accuracy, evaluate_items_sweep, build_response_matrix, evaluate_abilities
from subset_selection.utils import select_items_ranked
# from static_subset.subset_predictors import RandomForestSubsetPredictor

def sigmoid_stable(
    z: np.ndarray,
) -> np.ndarray:  # same shape as z
    return np.where(z >= 0, 1.0 / (1.0 + np.exp(-z)), np.exp(z) / (1.0 + np.exp(z)))


def fisher_information(
    theta: np.ndarray | float,  # shape (n_thetas, 1) when vectorised, or scalar
    a: np.ndarray,              # shape (n_items,)
    b: np.ndarray,              # shape (n_items,)
    D: float = 1.0,
) -> np.ndarray:  # shape (n_thetas, n_items) when theta is (n_thetas, 1), else (n_items,)

    z = D * a * (theta - b)
    P = sigmoid_stable(z)
    return (D ** 2) * (a ** 2) * (P * (1.0 - P))


def greedy_select(
    info_matrix: np.ndarray,  # shape (n_thetas, n_items)
    pair_select: bool = False,
) -> list[float]:  # length n_items; greedy marginal objective per item (insertion order)
    n_thetas, n_items = info_matrix.shape
    marginal_fisher_by_item = [np.nan] * n_items
    remaining_mask = np.ones(n_items, dtype=bool)
    current_tif = np.zeros(n_thetas)

    step = 0
    pbar = tqdm.tqdm(total=n_items, desc="Greedy select: Iterating over items.")

    while step < n_items:
        remaining_idx = np.where(remaining_mask)[0]
        if len(remaining_idx) == 0:
            break

        if not pair_select or len(remaining_idx) < 2:
            candidate_tifs = current_tif[:, None] + info_matrix
            candidate_tifs = np.maximum(candidate_tifs, 1e-12)
            candidate_objs = np.sum(1.0 / np.sqrt(candidate_tifs), axis=0)
            candidate_objs[~remaining_mask] = np.inf

            best = int(np.argmin(candidate_objs))
            marginal_fisher_by_item[best] = candidate_objs[best]
            remaining_mask[best] = False
            current_tif += info_matrix[:, best]
            step += 1
            pbar.update(1)
        else:
            best_obj = np.inf
            best_i, best_j = remaining_idx[0], remaining_idx[1]

            for ii, i in enumerate(remaining_idx):
                for j in remaining_idx[ii + 1:]:
                    pair_tif = current_tif + info_matrix[:, i] + info_matrix[:, j]
                    pair_tif = np.maximum(pair_tif, 1e-12)
                    obj = np.sum(1.0 / np.sqrt(pair_tif))
                    if obj < best_obj:
                        best_obj = obj
                        best_i, best_j = i, j

            marginal_fisher_by_item[best_i] = best_obj
            marginal_fisher_by_item[best_j] = best_obj
            remaining_mask[best_i] = False
            remaining_mask[best_j] = False
            current_tif += info_matrix[:, best_i] + info_matrix[:, best_j]
            step += 2
            pbar.update(2)

    pbar.close()
    return marginal_fisher_by_item



# ---------------------------------------------------------------------------
# High-level methodology API
# ---------------------------------------------------------------------------

class FisherMethod:
    """
    High-level API for Fisher-information-based item selection + RF-predicted accuracy.

    Covers total Fisher, marginal Fisher, and their difficulty-quantile variants.

    fit() selects items for all strategies and prepares train model signatures.
    evaluate() runs the per-K sweep and RF-predicted accuracy estimation.

    After fit(), ranked_indices is available for items_fold_df rank columns.
    """

    def __init__(self, max_items: int, n_points: int, n_quantiles: int) -> None:
        self.max_items = max_items
        self.n_points = n_points
        self.n_quantiles = n_quantiles
        self.ranked_indices: dict[str, list[int]] | None = None
        self._train_matrices: dict[str, np.ndarray] | None = None
        self._ac_train: np.ndarray | None = None
        self._ab_train: np.ndarray | None = None

    def fit(self, train_df: pd.DataFrame, parameters_df: pd.DataFrame) -> "FisherMethod":
        self.ranked_indices = {
            "total_fisher":           select_items_ranked(parameters_df, "total_fisher"),
            "marginal_fisher":        select_items_ranked(parameters_df, "marginal_fisher"),
            "total_fisher_bquant":    select_items_ranked(parameters_df, "total_fisher_bquant", n_quantiles=self.n_quantiles),
            "marginal_fisher_bquant": select_items_ranked(parameters_df, "marginal_fisher_bquant", n_quantiles=self.n_quantiles),
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
                # predictor_ac = RandomForestSubsetPredictor().fit(train_matrix[:, :K], self._ac_train)
                # predicted_ac = predictor_ac.predict(test_matrix[:, :K])

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