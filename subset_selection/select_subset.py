import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import warnings; warnings.filterwarnings("ignore")

from subset_selection.split import load_jsonl, split_fold, split_random_sample
from subset_selection.irt import fit_irt, build_eval_df
from subset_selection.fisher import fisher_information, greedy_select, FisherMethod
from subset_selection.evaluation import evaluate_abilities, evaluate_accuracy
from subset_selection.anchor_point_benchmark import AnchorPointMethod
from subset_selection.disco import DiscoSubsetMethod
from subset_selection.utils import RandomMethod
from subset_selection.fluid_benchmarking import FluidBenchmarkingMethod

ROOT = Path(__file__).resolve().parents[1]


def filter_low_variance_items(
    records: list[dict],
    min_variance: float = 0.01,
) -> list[dict]:

    all_items = list(records[0]["responses"].keys())
    scores = np.array([[r["responses"][item] for item in all_items] for r in records], dtype=float)
    p = scores.mean(axis=0)
    variance = p * (1 - p)
    keep = set(item for item, v in zip(all_items, variance) if v >= min_variance)
    removed = len(all_items) - len(keep)
    if removed:
        print(f"Removed {removed} items with variance < {min_variance} ({len(keep)} remaining)")
    return [
        {**r, "responses": {k: v for k, v in r["responses"].items() if k in keep}}
        for r in records
    ]


def make_run_tag(benchmark: str, split_tag: str, n_quantiles: int, min_variance: float, n_points: int) -> str:
    return f"{benchmark}_{split_tag}_{n_quantiles}bquant_minvar{min_variance}_step{n_points}"


def save_results(
    item_dfs: list[pd.DataFrame | None],
    perf_dfs: list[pd.DataFrame],
    run_tag: str,
    max_items: int,
    append_perf: bool = False,
) -> None:
    new_perf = pd.concat(perf_dfs, ignore_index=True)
    perf_path = Path(f"{run_tag}_{max_items}items_performance.csv")

    if append_perf and perf_path.exists():
        existing = pd.read_csv(perf_path)
        existing = existing[existing["method"] != "fluid_benchmarking"]
        new_perf = pd.concat([existing, new_perf], ignore_index=True)

    new_perf.to_csv(perf_path, index=False)
    print(f"Wrote {perf_path}")

    non_none_items = [df for df in item_dfs if df is not None]
    if non_none_items:
        items_path = Path(f"{run_tag}_items.csv")
        pd.concat(non_none_items, ignore_index=True).to_csv(items_path, index=False)
        print(f"Wrote {items_path}")


def run_fold(
    records: list[dict],
    fold_idx: int,
    max_items: int,
    n_quantiles: int,
    run_tag: str,
    n_points: int = 20,
    n_folds: int | None = None,
    n_samples: int | None = None,
    test_size: int = 10,
    run_static: bool = True,
    run_fluid: bool = False,
) -> tuple[pd.DataFrame | None, pd.DataFrame]:  # items_fold_df (None if not run_static), perf_fold_df
    if n_folds == 0:
        train = test = records
    elif n_samples is not None:
        train, test = split_random_sample(records, sample_idx=fold_idx, test_size=test_size)
    else:
        train, test = split_fold(records, n_folds=n_folds, fold_idx=fold_idx)
    parameters_df, _ = fit_irt(train, fold_tag=f"{run_tag}_fold_{fold_idx}")
    # Train and test are transformed into a long dataframe with columns
    # Model | item | score (0,1)
    train, test = build_eval_df(train), build_eval_df(test)
    parameters_df = parameters_df.reset_index(drop=True)

    models_train = sorted(train["model"].unique())
    models_test = sorted(test["model"].unique())

    perf_rows = []
    items_fold_df = None

    if run_static:
        # ── Compute Fisher information (prerequisite for static methods) ──────

        ab_train, ab_train_std = evaluate_abilities(train, parameters_df)
        info_matrix_train = fisher_information(ab_train[:, None], parameters_df["a"].values, parameters_df["b"].values)
        parameters_df["total_fisher"] = info_matrix_train.sum(axis=0)
        parameters_df["marginal_fisher_contribution"] = greedy_select(info_matrix_train)

        # ── Fit all static methodologies on train ─────────────────────────────

        fisher_method = FisherMethod(max_items, n_points, n_quantiles)
        fisher_method.fit(train, parameters_df)

        disco_method = DiscoSubsetMethod(max_items, n_points)
        disco_method.fit(train, parameters_df)

        random_method = RandomMethod(max_items, n_points)
        random_method.fit(train, parameters_df, fold_idx=fold_idx)

        anchor_method = AnchorPointMethod(max_items, n_points)
        anchor_method.fit(train, parameters_df)

        # ── Full-item baselines ───────────────────────────────────────────────

        ac_train, ac_train_std = evaluate_accuracy(train, len(parameters_df))
        ab_all,   ab_all_std   = evaluate_abilities(test, parameters_df)
        ac_all,   ac_all_std   = evaluate_accuracy(test, len(parameters_df))

        for model, ab, ab_std, ac, ac_std, split in [
            *zip(models_train, ab_train, ab_train_std, ac_train, ac_train_std, ["train"] * len(models_train)),
            *zip(models_test,  ab_all,   ab_all_std,   ac_all,   ac_all_std,   ["test"]  * len(models_test)),
        ]:
            perf_rows.append({
                "fold": fold_idx, "method": "all_items", "K": len(parameters_df),
                "model": model, "split": split,
                "ability": ab, "ability_std": ab_std,
                "accuracy": ac, "accuracy_std": ac_std,
            })

        # ── Evaluate static methodologies on test ─────────────────────────────

        perf_rows += fisher_method.evaluate(test, parameters_df, models_test, fold_idx)
        perf_rows += disco_method.evaluate(test, parameters_df, models_test, fold_idx)
        perf_rows += random_method.evaluate(test, parameters_df, models_test, fold_idx)
        perf_rows += anchor_method.evaluate(test, parameters_df, models_test, fold_idx)

        # ── Build items_fold_df (wide format) ─────────────────────────────────

        items_fold_df = (
            parameters_df[["item_ids", "a", "b", "total_fisher", "marginal_fisher_contribution"]]
            .copy()
            .rename(columns={"item_ids": "item_id"})
        )
        items_fold_df["item_id"] = items_fold_df["item_id"].astype(str)

        for method, indices in {**fisher_method.ranked_indices, **disco_method.ranked_indices, **random_method.ranked_indices}.items():
            rank_map = {str(parameters_df.iloc[pos]["item_ids"]): rank for rank, pos in enumerate(indices, start=1)}
            items_fold_df[f"{method}_rank"] = items_fold_df["item_id"].map(rank_map)

        for B, (positions, _) in sorted(anchor_method.anchor_items_by_B.items()):
            selected = {str(parameters_df.iloc[pos]["item_ids"]) for pos in positions}
            items_fold_df[f"anchor_B_{B}"] = items_fold_df["item_id"].isin(selected).astype(int)

        items_fold_df["fold"] = fold_idx

    if run_fluid:
        # ── Fluid benchmarking ────────────────────────────────────────────────

        fluid_method = FluidBenchmarkingMethod(max_items, n_points)
        fluid_method.fit(train, parameters_df)
        perf_rows += fluid_method.evaluate(test, parameters_df, models_test, fold_idx)

    return items_fold_df, pd.DataFrame(perf_rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select static IRT-based benchmark subsets across cross-validation folds."
    )
    parser.add_argument(
        "--benchmark",
        required=True,
        help="Benchmark name, used to name the output CSV (e.g. helm_airbench2024).",
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Relative path from repo root to the input JSONL file.",
    )
    parser.add_argument(
        "--n_folds",
        type=int,
        default=None,
        help="Number of cross-validation folds. Mutually exclusive with --n_samples.",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=None,
        help="Number of random subsamples. Mutually exclusive with --n_folds.",
    )
    parser.add_argument(
        "--test_size",
        type=int,
        default=10,
        help="Number of items in each test set when using --n_samples (default: 10).",
    )
    parser.add_argument(
        "--max_items",
        type=int,
        default=3000,
        help="Maximum number of items to evaluate per method (default: 3000).",
    )
    parser.add_argument(
        "--min_variance",
        type=float,
        default=0.01,
        help="Remove items whose empirical variance p(1-p) is below this threshold (default: 0.01).",
    )
    parser.add_argument(
        "--n_quantiles",
        type=int,
        default=4,
        help="Number of b-difficulty quantiles for the bquant methods (default: 4).",
    )
    parser.add_argument(
        "--n_points",
        type=int,
        default=20,
        help="Number of K values (log-spaced from 1 to max_items) to evaluate in the sweep (default: 20).",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print benchmark stats (number of models and items) and exit.",
    )
    parser.add_argument(
        "--no_static",
        action="store_true",
        help="Skip all static subset methods (Fisher, DISCO, random, anchor).",
    )
    parser.add_argument(
        "--fluid",
        action="store_true",
        help="Run fluid (adaptive) benchmarking via maximum Fisher information item selection.",
    )
    args = parser.parse_args()

    input_path = ROOT / args.input
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    records = load_jsonl(input_path)
    records = filter_low_variance_items(records, min_variance=args.min_variance)

    if args.dry_run:
        n_models = len(records)
        n_items = len(records[0]["responses"])
        print(f"Benchmark : {args.benchmark}")
        print(f"Models    : {n_models}")
        print(f"Items     : {n_items}")
        return

    run_static = not args.no_static
    run_fluid = args.fluid

    if not run_static and not run_fluid:
        parser.error("Nothing to run: pass --fluid, or remove --no_static.")

    if args.n_folds is not None and args.n_samples is not None:
        parser.error("--n_folds and --n_samples are mutually exclusive")

    use_random = args.n_samples is not None
    if use_random:
        n_samples = args.n_samples
        split_tag = f"{n_samples}samples_test{args.test_size}"
        fold_indices = range(n_samples)
    else:
        n_folds = args.n_folds if args.n_folds is not None else 5
        split_tag = f"{n_folds}folds" if n_folds > 0 else "nofold"
        fold_indices = [0] if n_folds == 0 else range(n_folds)

    run_tag = make_run_tag(args.benchmark, split_tag, args.n_quantiles, args.min_variance, args.n_points)

    all_item_dfs, all_perf_dfs = [], []
    for fold_idx in fold_indices:
        print(f"\n=== {'Sample' if use_random else 'Fold'} {fold_idx} ===")
        items_df, perf_df = run_fold(
            records, fold_idx, args.max_items, args.n_quantiles, run_tag,
            n_points=args.n_points,
            n_folds=None if use_random else n_folds,
            n_samples=n_samples if use_random else None,
            test_size=args.test_size,
            run_static=run_static,
            run_fluid=run_fluid,
        )
        all_item_dfs.append(items_df)
        all_perf_dfs.append(perf_df)

    fluid_only = run_fluid and not run_static
    save_results(all_item_dfs, all_perf_dfs, run_tag, args.max_items, append_perf=fluid_only)


if __name__ == "__main__":
    main()
