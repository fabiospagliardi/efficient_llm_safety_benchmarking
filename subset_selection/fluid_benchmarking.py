from __future__ import annotations

import numpy as np
import pandas as pd
import tqdm

from fluid_benchmarking.engine import run_fluid_benchmarking
from subset_selection.evaluation import evaluate_abilities
from subset_selection.fisher import fisher_information


class FluidBenchmarkingMethod:
    """
    High-level API for fluid (adaptive) benchmarking via maximum Fisher information item selection.

    Unlike static methods, each model receives a different adaptive item sequence.
    fit() computes the mean training-model ability, used as start_ability for each test model.
    evaluate() runs run_fluid_benchmarking() per test model and returns rows at log-spaced K values.

    ranked_indices is not populated (no fixed item ranking exists for items_fold_df).
    """

    def __init__(
        self,
        max_items: int,
        n_points: int,
        method: str = "mle",
        D: float = 1.0,
        mu0: float = 0.0,
        sigma0: float = 1.0,
        theta_range: tuple[float, float] = (-8.0, 8.0),
        tol: float = 1e-6,
        max_iter: int = 100,
    ) -> None:
        self.max_items = max_items
        self.n_points = n_points
        self.method = method
        self.D = D
        self.mu0 = mu0
        self.sigma0 = sigma0
        self.theta_range = theta_range
        self.tol = tol
        self.max_iter = max_iter
        self._start_ability: float = 0.0
        self._fitted = False

    def fit(self, train_df: pd.DataFrame, parameters_df: pd.DataFrame) -> "FluidBenchmarkingMethod":
        abilities, _ = evaluate_abilities(train_df, parameters_df)
        self._start_ability = float(np.mean(abilities))
        self._fitted = True
        return self

    def evaluate(
        self,
        test_df: pd.DataFrame,
        parameters_df: pd.DataFrame,
        models_test: list[str],
        fold_idx: int,
    ) -> list[dict]:
        if not self._fitted:
            raise RuntimeError("Call fit() before evaluate().")

        params = parameters_df.reset_index(drop=True)
        irt_model = params[["a", "b"]].values
        item_ids = params["item_ids"].astype(str).tolist()

        n_items = len(params)
        n_items_available = min(self.max_items, n_items)
        k_vals = sorted(set(int(round(x)) for x in np.geomspace(1, n_items_available, self.n_points)))

        rows = []
        for model in tqdm.tqdm(models_test, desc="Fluid benchmarking"):
            model_df = test_df[test_df["model"] == model].assign(
                item_id=lambda x: x["item_id"].astype(str)
            )
            score_map = dict(zip(model_df["item_id"], model_df["score"]))
            lm_responses = np.array([score_map.get(iid, np.nan) for iid in item_ids])

            n_valid = int(np.sum(~np.isnan(lm_responses)))
            if n_valid < n_items:
                print(f"Model {model}: {n_items - n_valid} items missing in test_df, skipping.")
                continue

            result = run_fluid_benchmarking(
                lm_responses=lm_responses,
                irt_model=irt_model,
                start_ability=self._start_ability,
                n_max=n_items_available,
                method=self.method,
                D=self.D,
                mu0=self.mu0,
                sigma0=self.sigma0,
                theta_range=self.theta_range,
                tol=self.tol,
                max_iter=self.max_iter,
            )
            items_fb = result["items_fb"]
            abilities_fb = result["abilities_fb"]
            n_administered = len(items_fb)

            for K in k_vals:
                if K > n_administered:
                    continue

                administered = items_fb[:K]
                ab = abilities_fb[K - 1]

                a_vals = irt_model[administered, 0]
                b_vals = irt_model[administered, 1]
                total_fi = fisher_information(ab, a_vals, b_vals, D=self.D).sum()
                ab_std = float(1.0 / np.sqrt(max(total_fi, 1e-12)))

                ac = float(np.mean(lm_responses[administered]))
                ac_std = float(np.sqrt(max(ac * (1 - ac) / K, 1e-12)))

                rows.append({
                    "fold": fold_idx,
                    "method": "fluid_benchmarking",
                    "K": K,
                    "model": model,
                    "split": "test",
                    "ability": ab,
                    "ability_std": ab_std,
                    "accuracy": ac,
                    "accuracy_std": ac_std,
                })

        return rows
