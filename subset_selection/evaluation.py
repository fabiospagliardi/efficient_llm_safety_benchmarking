import numpy as np
import pandas as pd
import tqdm

def calculate_accuracy(eval_dataset: pd.DataFrame) -> np.ndarray:  # shape: (n_models,)
    """Returns mean accuracy (% correct) per model, in sorted model order."""
    lms = sorted(eval_dataset["model"].unique())
    accs = []
    for lm in lms:
        mask = eval_dataset["model"] == lm
        scores = eval_dataset[mask]["score"]
        if len(scores) == 0:
            print(f"No items evaluated for model {lm}. Skipping.")
            continue
        accs.append(scores.mean())
    return np.array(accs)


def ability_estimate(
    lm_responses: np.ndarray,
    irt_model: np.ndarray,
    method: str = "map",
    D: float = 1.0,
    mu0: float = 0.0,
    sigma0: float = 1.0,
    theta0: float | None = None,
    theta_range: tuple[float, float] = (-8.0, 8.0),
    tol: float = 1e-6,
    max_iter: int = 100,
) -> float:
    """Ability estimate for a 2PL IRT model using MAP with normal prior or MLE."""
    from subset_selection.fisher import sigmoid_stable

    method = method.lower()
    if method not in {"map", "mle"}:
        raise ValueError("method must be 'map' or 'mle'.")

    lm_responses = np.asarray(lm_responses, dtype=float)
    irt_model = np.asarray(irt_model, dtype=float)

    if irt_model.ndim != 2 or irt_model.shape[1] != 2:
        raise ValueError("irt_model must be an (n_items, 2) array with columns [a, b].")
    n_items = irt_model.shape[0]

    if lm_responses.shape != (n_items,):
        raise ValueError("lm_responses must be shape (n_items,) aligned with irt_model.")

    mask = ~np.isnan(lm_responses)
    a = irt_model[mask, 0]
    b = irt_model[mask, 1]
    lm_responses = lm_responses[mask]

    if method == "map":
        if sigma0 <= 0:
            raise ValueError("sigma0 must be positive for MAP.")
        inv_sigma2 = 1.0 / (sigma0 * sigma0)
    else:
        inv_sigma2 = 0.0

    low, high = float(theta_range[0]), float(theta_range[1])
    if low >= high:
        raise ValueError("theta_range must have low < high.")

    def score(theta: float) -> float:
        z = D * a * (theta - b)
        P = sigmoid_stable(z)
        return (mu0 - theta) * inv_sigma2 + D * np.sum(a * (lm_responses - P))

    def score_prime(theta: float) -> float:
        z = D * a * (theta - b)
        P = sigmoid_stable(z)
        return -inv_sigma2 - (D**2) * np.sum(a * a * P * (1.0 - P))

    theta = mu0 if theta0 is None else theta0
    theta = float(np.clip(theta, low, high))

    for _ in range(max_iter):
        T = score(theta)
        if np.abs(T) < tol:
            return theta
        Tp = score_prime(theta)
        if not np.isfinite(Tp) or Tp == 0.0:
            break
        new_theta = theta - T / Tp
        if new_theta < low or new_theta > high or not np.isfinite(new_theta):
            new_theta = float(np.clip(new_theta, low, high))
        T_abs = abs(T)
        for _bt in range(15):
            T_new = score(new_theta)
            if abs(T_new) < T_abs or not np.isfinite(T_new):
                break
            new_theta = 0.5 * (new_theta + theta)
        theta = new_theta

    sL = score(low)
    sH = score(high)
    if sL * sH <= 0:
        lo, hi = low, high
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            sM = score(mid)
            if abs(sM) < tol:
                return mid
            if sL * sM > 0:
                lo, sL = mid, sM
            else:
                hi = mid
        return 0.5 * (lo + hi)
    return high if (sL > 0 and sH > 0) else low


def calculate_abilities(eval_dataset: pd.DataFrame, parameters_dataset: pd.DataFrame) -> np.ndarray:  # shape: (n_models,)
    lms = sorted(eval_dataset["model"].unique())
    theta = []
    for lm in lms:
        mask = (eval_dataset["model"] == lm)
        lm_eval_results = (
            eval_dataset[mask]
            .copy()
            .assign(item_id=lambda x: x["item_id"].astype(str))
            .sort_values("item_id")
            .reset_index(drop=True)
        )
        irt_model = (
            parameters_dataset
            .copy()
            .assign(item_ids=lambda x: x["item_ids"].astype(str))
            .sort_values(by="item_ids")
            .reset_index(drop=True)
        )
        assert all(col in irt_model.columns for col in ['a', 'b'])
        assert len(lm_eval_results) == len(irt_model)

        if not (lm_eval_results["item_id"].to_numpy() == irt_model["item_ids"].to_numpy()).all():
            raise ValueError(
                f"Item mismatch after sorting, lm={lm}\n"
                f"len responses: {len(lm_eval_results)}\n"
                f"len irt: {len(irt_model)}\n"
                f"first eval item_ids: {lm_eval_results['item_id'].head().tolist()}\n"
                f"first irt item_ids: {irt_model['item_ids'].head().tolist()}\n"
            )
        ability = ability_estimate(
            lm_responses=np.array(lm_eval_results["score"]),
            irt_model=irt_model[["a", "b"]],
            method="map"
        )
        theta.append(ability)
    return np.array(theta)


def evaluate_abilities(
    subset_df: pd.DataFrame,
    parameters_subset_df: pd.DataFrame,
) -> tuple[
    np.ndarray,  # abilities, shape (n_models,)
    np.ndarray,  # abilities_std, shape (n_models,)
]:
    """Compute IRT abilities and Fisher-information std for all models in subset_df."""
    from subset_selection.fisher import fisher_information
    abilities = calculate_abilities(subset_df, parameters_subset_df)
    a = parameters_subset_df["a"].values
    b = parameters_subset_df["b"].values
    abilities_std = 1.0 / np.sqrt(
        np.sum(fisher_information(abilities[:, None], a, b), axis=1)
    )
    return abilities, abilities_std


def evaluate_accuracy(
    subset_df: pd.DataFrame,
    n_items: int,
) -> tuple[
    np.ndarray,  # accuracy, shape (n_models,)
    np.ndarray,  # accuracy_std, shape (n_models,)
]:
    """Compute mean accuracy and binomial std for all models in subset_df."""
    accuracy = calculate_accuracy(subset_df)
    accuracy_std = np.sqrt(np.clip(accuracy * (1 - accuracy) / n_items, 1e-12, None))
    return accuracy, accuracy_std


def evaluate_items_sweep(
    ordered_indices: list[int],
    test_df: pd.DataFrame,
    parameters_df: pd.DataFrame,
    max_items: int,
    n_points: int,
) -> tuple[
    np.ndarray,  # abilities,      shape (n_models, n_k_vals)
    np.ndarray,  # abilities_std,  shape (n_models, n_k_vals)
    np.ndarray,  # accuracy,       shape (n_models, n_k_vals)
    np.ndarray,  # accuracy_std,   shape (n_models, n_k_vals)
    list[int],   # k_vals:         K values evaluated (log-spaced from 1 to n_items_available)
]:
    """
    Sweep over K values log-spaced from 1 to n_items_available and evaluate both abilities and accuracy at each K.

    ordered_indices : ordered list of integer positions into parameters_df (best first).
    """
    n_items_available = min(max_items, len(ordered_indices))
    k_vals = sorted(set(int(round(x)) for x in np.geomspace(1, n_items_available, n_points)))

    n_llms = test_df["model"].nunique()
    abilities_out     = np.full((n_llms, len(k_vals)), np.nan)
    abilities_std_out = np.full((n_llms, len(k_vals)), np.nan)
    accuracy_out      = np.full((n_llms, len(k_vals)), np.nan)
    accuracy_std_out  = np.full((n_llms, len(k_vals)), np.nan)

    ordered_item_ids = parameters_df.iloc[ordered_indices]["item_ids"].astype(str).tolist()
    item_rank = {item_id: rank for rank, item_id in enumerate(ordered_item_ids)}

    test_sorted = test_df.assign(_rank=test_df["item_id"].astype(str).map(item_rank)).sort_values("_rank")
    cumulative_counts = test_sorted["_rank"].searchsorted(range(1, len(ordered_item_ids) + 1))

    for j, K in enumerate(tqdm.tqdm(k_vals, desc="Evaluating subset performance")):
        k = cumulative_counts[K - 1]
        test_subset = test_sorted.iloc[:k]
        params_subset = parameters_df.iloc[ordered_indices[:K]]

        ab, ab_std = evaluate_abilities(test_subset, params_subset)
        ac, ac_std = evaluate_accuracy(test_subset, K)

        abilities_out[:, j]     = ab
        abilities_std_out[:, j] = ab_std
        accuracy_out[:, j]      = ac
        accuracy_std_out[:, j]  = ac_std

    return abilities_out, abilities_std_out, accuracy_out, accuracy_std_out, k_vals


def build_response_matrix(
    df: pd.DataFrame,
    ordered_indices: list[int],
    parameters_df: pd.DataFrame,
    max_items: int,
) -> tuple[np.ndarray, list[str]]:
    """
    Build a binary response matrix (n_models, K_max) for the ranked items.

    Rows are in sorted model-name order; columns follow the item ranking.

    Returns:
        matrix : (n_models, K_max) float array of binary scores
        models  : sorted model names (length n_models)
    """
    n_items_available = min(max_items, len(ordered_indices))
    ordered_item_ids = (
        parameters_df.iloc[ordered_indices[:n_items_available]]["item_ids"]
        .astype(str)
        .tolist()
    )
    pivot = (
        df.assign(item_id=lambda x: x["item_id"].astype(str))
        .pivot(index="model", columns="item_id", values="score")
        .sort_index()
        .reindex(columns=ordered_item_ids, fill_value=0)
        .astype(float)
    )
    return pivot.values, pivot.index.tolist()


def compute_accuracy_sweep(
    ordered_indices: list[int],
    df: pd.DataFrame,
    parameters_df: pd.DataFrame,
    max_items: int,
    n_points: int,
) -> tuple[
    np.ndarray,  # accuracy_out, shape (n_models, n_k_vals); models in sorted name order
    list[int],   # k_vals: K values evaluated (log-spaced from 1 to n_items_available)
]:
    """
    Sweep over K values log-spaced from 1 to n_items_available and compute only accuracy for all models in df.

    Cheaper than evaluate_items_sweep (no IRT fitting), intended for train models
    where accuracy at each K is needed to fit a subset predictor.
    """
    n_items_available = min(max_items, len(ordered_indices))
    k_vals = sorted(set(int(round(x)) for x in np.geomspace(1, n_items_available, n_points)))

    n_llms = df["model"].nunique()
    accuracy_out = np.full((n_llms, len(k_vals)), np.nan)

    ordered_item_ids = parameters_df.iloc[ordered_indices]["item_ids"].astype(str).tolist()
    item_rank = {item_id: rank for rank, item_id in enumerate(ordered_item_ids)}

    df_sorted = df.assign(_rank=df["item_id"].astype(str).map(item_rank)).sort_values("_rank")
    cumulative_counts = df_sorted["_rank"].searchsorted(range(1, len(ordered_item_ids) + 1))

    for j, K in enumerate(k_vals):
        k = cumulative_counts[K - 1]
        accuracy_out[:, j] = calculate_accuracy(df_sorted.iloc[:k])

    return accuracy_out, k_vals
