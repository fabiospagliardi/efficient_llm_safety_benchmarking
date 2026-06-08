"""
Anchor Point Benchmark Distillation
=====================================
Implements the Anchor Point Selection (APS) method from:

    Vivek et al. (2024) "Anchor Points: Benchmarking Models with Much Fewer Examples"
    https://arxiv.org/abs/2404.13744

Key idea: cluster all benchmark questions by their correlation structure (computed
over a set of *source* models), select one representative "anchor" per cluster via
K-Medoids (PAM), and weight each anchor by its cluster size.  Any new (*target*)
model evaluated only on the B anchors receives an estimated full-benchmark score via
the APW (Anchor Point Weighted) formula.
"""

from __future__ import annotations

from collections.abc import Sequence

import kmedoids
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tqdm

from subset_selection.evaluation import evaluate_abilities, evaluate_accuracy
# from static_subset.subset_predictors import RandomForestSubsetPredictor


# ---------------------------------------------------------------------------
# Adapter: wraps kmedoids.fasterpam result in the sklearn KMedoids interface
# ---------------------------------------------------------------------------

class _KMedoidsAdapter:
    def __init__(self, n_clusters: int, random_state: int | None = None) -> None:
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.labels_: np.ndarray | None = None
        self.medoid_indices_: np.ndarray | None = None

    def fit(self, D: np.ndarray) -> "_KMedoidsAdapter":
        result = kmedoids.fasterpam(D, self.n_clusters, random_state=self.random_state)
        self.labels_ = np.asarray(result.labels)
        self.medoid_indices_ = np.asarray(result.medoids)
        return self


def _make_kmedoids(n_clusters: int, random_state: int) -> "_KMedoidsAdapter":
    return _KMedoidsAdapter(n_clusters=n_clusters, random_state=random_state)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class AnchorPointBenchmark:
    """
    Distills a binary evaluation benchmark into B anchor questions using
    the Anchor Point Selection (APS) method (Vivek et al., 2024).

    Terminology from the paper
    --------------------------
    Source models  : LLMs used during fit() to learn the correlation structure.
    Target models  : New LLMs to be evaluated cheaply via anchor questions only.
    Anchor points  : The B representative questions selected as cluster medoids.
    APW score      : Anchor Point Weighted score — the weighted sum used to estimate
                     a target model's full-benchmark accuracy.
    """

    def __init__(self, B: int = 50, random_state: int = 42) -> None:
        """
        Parameters
        ----------
        B : int
            Number of anchor questions to select.
        random_state : int
            Seed for K-Medoids (PAM) reproducibility.
        """
        self.B = B
        self.random_state = random_state

        # Populated by fit()
        self.anchor_indices: np.ndarray | None = None   # shape (B,)
        self.weights: np.ndarray | None = None           # shape (B,), sums to 1
        self.cluster_labels: np.ndarray | None = None   # shape (n_questions,)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, results_matrix: np.ndarray) -> "AnchorPointBenchmark":
        """
        Fit the anchor point model on a set of source models.

        Steps (following Section 3 of Vivek et al., 2024):
          1. Compute pairwise Pearson correlations between questions using the
             pattern of correct/incorrect answers across source models as signal.
          2. Convert correlations to distances: d = 1 − r.
          3. Run K-Medoids (PAM) on the precomputed distance matrix to choose
             B anchor questions (medoids).
          4. Assign APW scores: weight_b = |cluster_b| / n_questions.

        Parameters
        ----------
        results_matrix : np.ndarray, shape (n_source_llms, n_questions)
            Binary matrix; 1 = correct, 0 = incorrect, for source models.

        Returns
        -------
        self
        """
        n_llms, n_questions = results_matrix.shape

        if self.B > n_questions:
            raise ValueError(
                f"B={self.B} exceeds the number of questions ({n_questions})."
            )
        if self.B > n_llms:
            import warnings
            warnings.warn(
                f"B={self.B} > n_source_llms={n_llms}; correlation estimates may be noisy."
            )

        # --- Step 1: Pairwise Pearson correlations between questions ---
        # Transpose so each row is a question vector over source models.
        # np.corrcoef returns an (n_questions, n_questions) symmetric matrix.
        corr_matrix = np.corrcoef(results_matrix.T)

        # Questions with zero variance (always correct or always wrong) yield NaN
        # correlations; treat them as uncorrelated (r = 0 → distance = 1).
        corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)

        # --- Step 2: Correlation → distance (d = 1 − r) ---
        # Maps r ∈ [−1, 1] to d ∈ [0, 2].
        dist_matrix = 1.0 - corr_matrix

        # Guard against tiny floating-point violations of the [0, 2] range.
        dist_matrix = np.clip(dist_matrix, 0.0, 2.0)

        # Self-distance must be exactly 0 for precomputed-metric K-Medoids.
        np.fill_diagonal(dist_matrix, 0.0)

        # --- Step 3: K-Medoids (PAM) — select B anchor questions ---
        # Each medoid becomes an anchor point; questions cluster around B centers.
        kmedoids = _make_kmedoids(self.B, self.random_state)
        kmedoids.fit(dist_matrix)

        self.cluster_labels = kmedoids.labels_          # (n_questions,)
        self.anchor_indices = kmedoids.medoid_indices_  # (B,)

        # --- Step 4: APW weights — proportional to cluster size ---
        # Weight of anchor b = fraction of all questions in cluster b.
        # This ensures the APW score equals the full-benchmark score in expectation.
        cluster_sizes = np.bincount(self.cluster_labels, minlength=self.B)
        self.weights = cluster_sizes / n_questions  # (B,), sums to 1.0

        return self

    def fit_precomputed_dist(self, dist_matrix: np.ndarray) -> "AnchorPointBenchmark":
        """
        Fit using a precomputed (n_questions, n_questions) distance matrix.

        Identical to fit() but skips the Pearson correlation step.  Use this
        when sweeping many B values on the same dataset to avoid recomputing
        the correlation matrix on every call.

        Parameters
        ----------
        dist_matrix : np.ndarray, shape (n_questions, n_questions)
            Symmetric non-negative distance matrix, e.g. produced by
            ``dist = np.clip(1 - np.corrcoef(train_matrix.T), 0, 2)``.

        Returns
        -------
        self
        """
        n_questions = len(dist_matrix)
        if self.B > n_questions:
            raise ValueError(
                f"B={self.B} exceeds the number of questions ({n_questions})."
            )

        kmedoids = _make_kmedoids(self.B, self.random_state)
        kmedoids.fit(dist_matrix)

        self.cluster_labels = kmedoids.labels_
        self.anchor_indices = kmedoids.medoid_indices_

        cluster_sizes = np.bincount(self.cluster_labels, minlength=self.B)
        self.weights = cluster_sizes / n_questions

        return self

    def score(self, new_llm_anchor_results: np.ndarray) -> float:
        """
        Estimate the full-benchmark score for a target model using only its
        results on the B anchor questions (APW score formula).

            APW_score = Σ_b  anchor_result_b × weight_b

        Parameters
        ----------
        new_llm_anchor_results : np.ndarray, shape (B,)
            Binary results (1 = correct, 0 = incorrect) on the B anchor questions.

        Returns
        -------
        float
            Estimated benchmark accuracy in [0, 1].
        """
        self._check_fitted()
        if len(new_llm_anchor_results) != self.B:
            raise ValueError(
                f"Expected {self.B} anchor results, got {len(new_llm_anchor_results)}."
            )
        # APW score: weighted dot product of binary anchor results and cluster weights
        return float(np.dot(new_llm_anchor_results, self.weights))

    def validate(self, held_out_results: np.ndarray) -> dict:
        """
        Measure estimation accuracy on a held-out set of target models.

        For each held-out LLM:
          - APW (estimated) score = weighted sum over anchor questions only
          - True score            = mean accuracy over all questions
        then report Mean Absolute Error (MAE) and per-model breakdowns.

        Parameters
        ----------
        held_out_results : np.ndarray, shape (n_held_out_llms, n_questions)
            Full binary result matrix for target (held-out) models.

        Returns
        -------
        dict with keys
            'mae'              : float   — mean absolute error across all held-out LLMs
            'estimated_scores' : ndarray — APW scores, shape (n_held_out_llms,)
            'true_scores'      : ndarray — mean accuracy, shape (n_held_out_llms,)
            'errors'           : ndarray — per-LLM absolute error
        """
        self._check_fitted()

        # Extract only the anchor columns for each held-out (target) model
        anchor_results = held_out_results[:, self.anchor_indices]  # (n_held_out, B)

        # APW score for every target model (vectorised dot product)
        estimated_scores = anchor_results @ self.weights  # (n_held_out,)

        # True benchmark score = fraction of all questions answered correctly
        true_scores = held_out_results.mean(axis=1)  # (n_held_out,)

        errors = np.abs(estimated_scores - true_scores)
        mae = float(errors.mean())

        print(f"Validation MAE: {mae:.4f}  (n={len(held_out_results)} held-out models)")
        return {
            "mae": mae,
            "estimated_scores": estimated_scores,
            "true_scores": true_scores,
            "errors": errors,
        }

    def plot_error_vs_B(
        self,
        results_matrix: np.ndarray,
        b_range: Sequence[int],
        held_out_fraction: float = 0.2,
        ax: plt.Axes | None = None,
    ) -> plt.Axes:
        """
        Plot estimation MAE as a function of the number of anchor points B.

        Internally splits results_matrix into source models (used for fit) and
        held-out target models (used for validation), then sweeps over b_range.
        Use this curve to choose an appropriate B for your dataset.

        Parameters
        ----------
        results_matrix : np.ndarray, shape (n_llms, n_questions)
            Combined binary result matrix (source + held-out LLMs together).
        b_range : Sequence[int]
            Values of B to evaluate, e.g. range(5, 101, 5).
        held_out_fraction : float
            Fraction of LLMs to reserve as held-out target models (default 0.2).
        ax : matplotlib.axes.Axes, optional
            Axes to draw on; a new figure is created if None.

        Returns
        -------
        matplotlib.axes.Axes
        """
        n_llms = results_matrix.shape[0]
        n_held_out = max(1, int(n_llms * held_out_fraction))

        # Reproducible random split into source and held-out (target) models
        rng = np.random.default_rng(self.random_state)
        held_out_idx = rng.choice(n_llms, size=n_held_out, replace=False)
        source_idx = np.setdiff1d(np.arange(n_llms), held_out_idx)

        source_matrix = results_matrix[source_idx]
        held_out_matrix = results_matrix[held_out_idx]

        maes = []
        b_values = list(b_range)
        for b in b_values:
            # Fresh fit for each B; the same held-out split ensures a fair comparison
            model = AnchorPointBenchmark(B=b, random_state=self.random_state)
            model.fit(source_matrix)
            result = model.validate(held_out_matrix)
            maes.append(result["mae"])
            print(f"  B={b:4d} | MAE={result['mae']:.4f}")

        if ax is None:
            _, ax = plt.subplots(figsize=(8, 5))

        ax.plot(b_values, maes, marker="o", linewidth=1.5)
        ax.set_xlabel("Number of anchor points (B)")
        ax.set_ylabel("Mean Absolute Error (MAE)")
        ax.set_title("Estimation Error vs Number of Anchor Points (APS)")
        ax.grid(True, linestyle="--", alpha=0.5)

        return ax

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check_fitted(self) -> None:
        if self.anchor_indices is None:
            raise RuntimeError("Call fit() before using this method.")


# ---------------------------------------------------------------------------
# Item selection helper
# ---------------------------------------------------------------------------

def select_anchor_items(
    train_df: pd.DataFrame,
    parameters_df: pd.DataFrame,
    max_items: int,
    n_points: int,
    random_state: int = 42,
) -> dict[int, tuple[list[int], "AnchorPointBenchmark"]]:
    """
    Select anchor items via K-Medoids on Pearson-correlation distances of training responses.

    Returns {B: (sorted_positions, fitted_AnchorPointBenchmark)} where positions index into
    parameters_df.  The fitted object is needed to compute APW scores on new models.
    The Pearson distance matrix is precomputed once and reused for all B values.
    Returns {} when no valid B values exist.
    """
    item_ids = parameters_df["item_ids"].astype(str).tolist()
    n_items = len(item_ids)

    train_matrix = (
        train_df
        .assign(item_id=lambda x: x["item_id"].astype(str))
        .pivot(index="model", columns="item_id", values="score")
        .reindex(columns=item_ids, fill_value=0)
        .astype(float)
        .values
    )

    corr = np.corrcoef(train_matrix.T)
    corr = np.nan_to_num(corr, nan=0.0)
    dist = np.clip(1.0 - corr, 0.0, 2.0)
    np.fill_diagonal(dist, 0.0)

    n_items_available = min(max_items, n_items)
    b_values = sorted(set(int(round(x)) for x in np.geomspace(1, n_items_available, n_points)))
    if not b_values:
        return {}

    result = {}
    for B in tqdm.tqdm(b_values, desc="Anchor point: selecting items"):
        apb = AnchorPointBenchmark(B=B, random_state=random_state)
        apb.fit_precomputed_dist(dist)
        positions = sorted(apb.anchor_indices)
        result[B] = (positions, apb)

    return result


# ---------------------------------------------------------------------------
# High-level methodology API
# ---------------------------------------------------------------------------

class AnchorPointMethod:
    """
    High-level API for the Anchor Point methodology.

    fit() selects anchor items on training models via K-Medoids.
    evaluate() returns performance rows for IRT-based accuracy
    ('anchor_point') and APW-weighted prediction ('anchor_point_predicted').

    After fit(), anchor_items_by_B is available for items_fold_df construction.
    """

    def __init__(self, max_items: int, n_points: int, random_state: int = 42) -> None:
        self.max_items = max_items
        self.n_points = n_points
        self.random_state = random_state
        self.anchor_items_by_B: dict[int, tuple[list[int], AnchorPointBenchmark]] | None = None
        self._ac_train: np.ndarray | None = None
        self._ab_train: np.ndarray | None = None

    def fit(self, train_df: pd.DataFrame, parameters_df: pd.DataFrame) -> "AnchorPointMethod":
        self.anchor_items_by_B = select_anchor_items(
            train_df, parameters_df, self.max_items, self.n_points, self.random_state
        )
        self._ac_train, _ = evaluate_accuracy(train_df, len(parameters_df))
        self._ab_train, _ = evaluate_abilities(train_df, parameters_df)

        params_str = parameters_df.copy().assign(item_ids=lambda x: x["item_ids"].astype(str))
        item_ids = params_str["item_ids"].tolist()
        train_str = train_df.assign(item_id=lambda x: x["item_id"].astype(str))
        train_matrix_all = (
            train_str
            .pivot(index="model", columns="item_id", values="score")
            .sort_index()
            .reindex(columns=item_ids, fill_value=0)
            .astype(float)
            .values
        )
        self._train_matrices: dict[int, np.ndarray] = {
            B: train_matrix_all[:, apb.anchor_indices]
            for B, (_, apb) in self.anchor_items_by_B.items()
        }
        return self

    def evaluate(
        self,
        test_df: pd.DataFrame,
        parameters_df: pd.DataFrame,
        models_test: list[str],
        fold_idx: int,
    ) -> list[dict]:
        if self.anchor_items_by_B is None:
            raise RuntimeError("Call fit() before evaluate().")

        n_test = len(models_test)
        test_str = test_df.assign(item_id=lambda x: x["item_id"].astype(str))
        params_str = parameters_df.copy().assign(item_ids=lambda x: x["item_ids"].astype(str))

        item_ids = params_str["item_ids"].tolist()
        test_matrix_all = (
            test_str
            .pivot(index="model", columns="item_id", values="score")
            .sort_index()
            .reindex(columns=item_ids, fill_value=0)
            .astype(float)
            .values
        )

        rows = []
        for B, (positions, apb) in sorted(self.anchor_items_by_B.items()):
            params_subset = params_str.iloc[positions]
            anchor_ids = set(params_subset["item_ids"])
            test_subset = test_str[test_str["item_id"].isin(anchor_ids)]

            if len(params_subset) == 0:
                continue

            thetas, thetas_std = evaluate_abilities(test_subset, params_subset)
            if len(thetas) != n_test:
                continue

            acc, acc_std = evaluate_accuracy(test_subset, B)
            if len(acc) != n_test:
                continue

            apw_scores = test_matrix_all[:, apb.anchor_indices] @ apb.weights

            test_matrix_B = test_matrix_all[:, apb.anchor_indices]
            train_matrix_B = self._train_matrices[B]
            # predictor_ac = RandomForestSubsetPredictor().fit(train_matrix_B, self._ac_train)
            # predicted_ac = predictor_ac.predict(test_matrix_B)
            # predictor_ab = RandomForestSubsetPredictor().fit(train_matrix_B, self._ab_train)
            # predicted_ab = predictor_ab.predict(test_matrix_B)

            for m_idx, model in enumerate(models_test):
                rows.append({
                    "fold": fold_idx, "method": "anchor_point", "K": B,
                    "model": model, "split": "test",
                    "ability": thetas[m_idx], "ability_std": thetas_std[m_idx],
                    "accuracy": acc[m_idx], "accuracy_std": acc_std[m_idx],
                    "ap_pred": apw_scores[m_idx],
                    # "accuracy_pred": predicted_ac[m_idx],
                    # "ability_pred": predicted_ab[m_idx],
                })

        return rows


# ---------------------------------------------------------------------------
# Synthetic data example — demonstrates the full APS workflow
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(0)

    def _sigmoid(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-x))

    # Simulate a 1PL IRT benchmark: 60 source + 20 target LLMs, 200 questions.
    # Each LLM has a latent ability θ; each question has a latent difficulty β.
    # P(LLM_i answers Q_j correctly) = sigmoid(θ_i − β_j)
    n_source, n_target, n_questions = 60, 20, 200

    abilities_source = rng.normal(0.0, 1.0, size=n_source)  # source model abilities (θ)
    abilities_target = rng.normal(0.0, 1.0, size=n_target)  # target model abilities (θ)
    difficulties     = rng.normal(0.0, 1.0, size=n_questions)  # question difficulties (β)

    source_probs = _sigmoid(abilities_source[:, None] - difficulties[None, :])
    target_probs = _sigmoid(abilities_target[:, None] - difficulties[None, :])

    source_results = rng.binomial(1, source_probs).astype(float)  # (60, 200)
    target_results = rng.binomial(1, target_probs).astype(float)  # (20, 200)

    # --- 1. Fit on source models (learn correlation structure + pick anchors) ---
    print("=" * 58)
    print("Step 1: Fitting AnchorPointBenchmark (B=20) on source models")
    print("=" * 58)
    apb = AnchorPointBenchmark(B=20, random_state=42)
    apb.fit(source_results)
    print(f"Anchor indices : {apb.anchor_indices}")
    print(f"Weights (sum={apb.weights.sum():.4f}): {np.round(apb.weights, 3)}")

    # --- 2. Score a single new target LLM using only its anchor results ---
    print("\n" + "=" * 58)
    print("Step 2: Scoring a single target model (APW formula)")
    print("=" * 58)
    single_llm    = target_results[0]
    anchor_answers = single_llm[apb.anchor_indices]
    estimated      = apb.score(anchor_answers)
    true_score     = single_llm.mean()
    print(f"  Estimated score (APW) : {estimated:.4f}")
    print(f"  True score            : {true_score:.4f}")
    print(f"  Absolute error        : {abs(estimated - true_score):.4f}")

    # --- 3. Validate across all held-out target models ---
    print("\n" + "=" * 58)
    print("Step 3: Validation on all target models")
    print("=" * 58)
    val = apb.validate(target_results)

    # --- 4. Plot estimation error vs B ---
    print("\n" + "=" * 58)
    print("Step 4: Plotting MAE vs number of anchor points B")
    print("=" * 58)
    all_results = np.vstack([source_results, target_results])  # (80, 200)
    ax = apb.plot_error_vs_B(
        results_matrix=all_results,
        b_range=range(5, 51, 5),
    )
    plt.tight_layout()
    plt.savefig("anchor_point_error_vs_B.png", dpi=150)
    print("\nSaved: anchor_point_error_vs_B.png")
    plt.show()
