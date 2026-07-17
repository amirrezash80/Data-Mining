import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score


ROOT = Path(__file__).resolve().parents[1]
PARAMS_PATH = ROOT / "params_phase2.yaml"

ARRAY_PATH = (
    ROOT
    / "data"
    / "processed"
    / "phase1_arrays.npz"
)

PHASE2_SAMPLE_PATH = (
    ROOT
    / "reports"
    / "phase2"
    / "evaluation_sample_indices.csv"
)

KMEANS_SEARCH_PATH = (
    ROOT
    / "reports"
    / "phase2"
    / "kmeans_search.csv"
)

REPORT_DIR = (
    ROOT
    / "reports"
    / "phase2"
)

FIGURE_DIR = (
    ROOT
    / "reports"
    / "figures"
    / "phase2"
)

PROCESSED_DIR = (
    ROOT
    / "data"
    / "processed"
)


def ensure_directories() -> None:
    for directory in [
        REPORT_DIR,
        FIGURE_DIR,
        PROCESSED_DIR,
    ]:
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(
                REPORT_DIR
                / "phase2_completion_step3.log",
                mode="w",
                encoding="utf-8",
            ),
        ],
        force=True,
    )


def setup_plots() -> None:
    sns.set_theme(
        style="whitegrid",
        context="notebook",
        palette="colorblind",
    )

    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 200,
            "figure.autolayout": True,
        }
    )


def load_yaml(
    path: Path,
) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        params = yaml.safe_load(file)

    if not isinstance(params, dict):
        raise ValueError(
            "Invalid params_phase2.yaml"
        )

    return params


def save_json(
    data: Any,
    path: Path,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            data,
            file,
            ensure_ascii=False,
            indent=2,
            default=str,
        )


def save_figure(
    filename: str,
) -> Path:
    path = FIGURE_DIR / filename

    plt.savefig(
        path,
        bbox_inches="tight",
    )

    plt.close()

    logging.info(
        "Saved figure: %s",
        path,
    )

    return path


def load_analysis_data(
    params: dict[str, Any],
) -> tuple[
    np.ndarray,
    np.ndarray,
]:
    if not ARRAY_PATH.exists():
        raise FileNotFoundError(
            f"Phase 1 arrays not found: {ARRAY_PATH}"
        )

    with np.load(
        ARRAY_PATH,
        allow_pickle=False,
    ) as arrays:
        required = {
            "X_pca",
            "train_indices",
        }

        missing = required - set(
            arrays.files
        )

        if missing:
            raise KeyError(
                f"Missing arrays: {sorted(missing)}"
            )

        x_all = arrays[
            "X_pca"
        ].astype(np.float64)

        train_indices = arrays[
            "train_indices"
        ].astype(np.int64)

    if PHASE2_SAMPLE_PATH.exists():
        sample_table = pd.read_csv(
            PHASE2_SAMPLE_PATH
        )

        if (
            "array_index"
            not in sample_table.columns
        ):
            raise KeyError(
                "array_index is missing from "
                "evaluation_sample_indices.csv"
            )

        selected_indices = sample_table[
            "array_index"
        ].to_numpy(dtype=np.int64)
    else:
        sample_size = min(
            int(
                params["sampling"][
                    "evaluation_size"
                ]
            ),
            len(train_indices),
        )

        random_seed = int(
            params["project"][
                "random_seed"
            ]
        )

        rng = np.random.default_rng(
            random_seed
        )

        selected_indices = rng.choice(
            train_indices,
            size=sample_size,
            replace=False,
        )

    selected_indices = np.unique(
        selected_indices
    )

    if np.any(selected_indices < 0):
        raise IndexError(
            "Negative sample index found"
        )

    if np.any(
        selected_indices >= len(x_all)
    ):
        raise IndexError(
            "Sample index exceeds X_pca"
        )

    matrix = x_all[
        selected_indices
    ]

    if not np.isfinite(matrix).all():
        raise ValueError(
            "Analysis matrix contains "
            "NaN or infinite values"
        )

    return (
        matrix,
        selected_indices,
    )


def determine_selected_k(
    params: dict[str, Any],
) -> int:
    if KMEANS_SEARCH_PATH.exists():
        results = pd.read_csv(
            KMEANS_SEARCH_PATH
        )

        required_columns = {
            "k",
            "silhouette",
        }

        missing = (
            required_columns
            - set(results.columns)
        )

        if missing:
            raise KeyError(
                "Missing K-Means search columns: "
                f"{sorted(missing)}"
            )

        valid = results.dropna(
            subset=["silhouette"]
        )

        if not valid.empty:
            selected = valid.sort_values(
                [
                    "silhouette",
                    "davies_bouldin",
                ],
                ascending=[
                    False,
                    True,
                ],
            ).iloc[0]

            return int(
                selected["k"]
            )

    return int(
        params["cluster_search"][
            "k_min"
        ]
    )


def jaccard_similarity(
    first: np.ndarray,
    second: np.ndarray,
) -> float:
    intersection = np.sum(
        first & second
    )

    union = np.sum(
        first | second
    )

    if union == 0:
        return 1.0

    return float(
        intersection / union
    )


def match_clusters_by_jaccard(
    reference_labels: np.ndarray,
    bootstrap_labels: np.ndarray,
) -> list[dict[str, Any]]:
    reference_clusters = np.unique(
        reference_labels
    )

    bootstrap_clusters = np.unique(
        bootstrap_labels
    )

    rows = []

    for reference_cluster in (
        reference_clusters
    ):
        reference_mask = (
            reference_labels
            == reference_cluster
        )

        best_cluster = None
        best_jaccard = -1.0

        for bootstrap_cluster in (
            bootstrap_clusters
        ):
            bootstrap_mask = (
                bootstrap_labels
                == bootstrap_cluster
            )

            score = jaccard_similarity(
                reference_mask,
                bootstrap_mask,
            )

            if score > best_jaccard:
                best_jaccard = score
                best_cluster = int(
                    bootstrap_cluster
                )

        rows.append(
            {
                "reference_cluster": int(
                    reference_cluster
                ),
                "matched_bootstrap_cluster": (
                    best_cluster
                ),
                "jaccard": float(
                    best_jaccard
                ),
                "reference_cluster_size": int(
                    reference_mask.sum()
                ),
            }
        )

    return rows


def update_coclustering_counts(
    counts: np.ndarray,
    labels: np.ndarray,
) -> None:
    for cluster_id in np.unique(
        labels
    ):
        members = np.flatnonzero(
            labels == cluster_id
        )

        counts[
            np.ix_(members, members)
        ] += 1


def run_bootstrap_for_k(
    x: np.ndarray,
    anchor_positions: np.ndarray,
    k: int,
    bootstrap_runs: int,
    n_init: int,
    random_seed: int,
    store_matrix: bool,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    np.ndarray ,
    np.ndarray,
]:
    anchor_x = x[
        anchor_positions
    ]

    reference_model = KMeans(
        n_clusters=k,
        init="k-means++",
        n_init=n_init,
        max_iter=300,
        random_state=random_seed,
    )

    reference_model.fit(x)

    reference_labels = (
        reference_model.predict(
            anchor_x
        ).astype(np.int32)
    )

    if store_matrix:
        coclustering_counts = np.zeros(
            (
                len(anchor_x),
                len(anchor_x),
            ),
            dtype=np.uint16,
        )
    else:
        coclustering_counts = None

    jaccard_rows = []
    run_rows = []

    number_of_records = len(x)

    for run_id in range(
        bootstrap_runs
    ):
        run_seed = (
            random_seed
            + 1000
            + 100 * k
            + run_id
        )

        rng = np.random.default_rng(
            run_seed
        )

        bootstrap_positions = rng.choice(
            number_of_records,
            size=number_of_records,
            replace=True,
        )

        bootstrap_x = x[
            bootstrap_positions
        ]

        model = KMeans(
            n_clusters=k,
            init="k-means++",
            n_init=n_init,
            max_iter=300,
            random_state=run_seed,
        )

        start = time.perf_counter()

        model.fit(
            bootstrap_x
        )

        predicted_labels = model.predict(
            anchor_x
        ).astype(np.int32)

        runtime = (
            time.perf_counter()
            - start
        )

        pairwise_ari = float(
            adjusted_rand_score(
                reference_labels,
                predicted_labels,
            )
        )

        matched_rows = (
            match_clusters_by_jaccard(
                reference_labels=(
                    reference_labels
                ),
                bootstrap_labels=(
                    predicted_labels
                ),
            )
        )

        run_jaccards = []

        for row in matched_rows:
            row.update(
                {
                    "k": int(k),
                    "bootstrap_run": int(
                        run_id + 1
                    ),
                    "seed": int(
                        run_seed
                    ),
                }
            )

            jaccard_rows.append(
                row
            )

            run_jaccards.append(
                row["jaccard"]
            )

        run_rows.append(
            {
                "k": int(k),
                "bootstrap_run": int(
                    run_id + 1
                ),
                "seed": int(
                    run_seed
                ),
                "pairwise_ari_to_reference": (
                    pairwise_ari
                ),
                "mean_cluster_jaccard": float(
                    np.mean(
                        run_jaccards
                    )
                ),
                "minimum_cluster_jaccard": float(
                    np.min(
                        run_jaccards
                    )
                ),
                "runtime_seconds": float(
                    runtime
                ),
                "unique_bootstrap_records": int(
                    len(
                        np.unique(
                            bootstrap_positions
                        )
                    )
                ),
            }
        )

        if coclustering_counts is not None:
            update_coclustering_counts(
                counts=(
                    coclustering_counts
                ),
                labels=predicted_labels,
            )

        logging.info(
            "Bootstrap k=%d run=%d/%d "
            "ARI=%.4f Jaccard=%.4f",
            k,
            run_id + 1,
            bootstrap_runs,
            pairwise_ari,
            float(
                np.mean(
                    run_jaccards
                )
            ),
        )

    run_results = pd.DataFrame(
        run_rows
    )

    jaccard_results = pd.DataFrame(
        jaccard_rows
    )

    if coclustering_counts is not None:
        probability_matrix = (
            coclustering_counts.astype(
                np.float32
            )
            / float(
                bootstrap_runs
            )
        )

        np.fill_diagonal(
            probability_matrix,
            1.0,
        )
    else:
        probability_matrix = None

    return (
        run_results,
        jaccard_results,
        probability_matrix,
        reference_labels,
    )


def create_k_stability_summary(
    run_results: pd.DataFrame,
    jaccard_results: pd.DataFrame,
) -> pd.DataFrame:
    run_summary = (
        run_results.groupby("k")
        .agg(
            bootstrap_runs=(
                "bootstrap_run",
                "count",
            ),
            ari_mean=(
                "pairwise_ari_to_reference",
                "mean",
            ),
            ari_std=(
                "pairwise_ari_to_reference",
                "std",
            ),
            ari_minimum=(
                "pairwise_ari_to_reference",
                "min",
            ),
            ari_maximum=(
                "pairwise_ari_to_reference",
                "max",
            ),
            mean_jaccard=(
                "mean_cluster_jaccard",
                "mean",
            ),
            jaccard_std=(
                "mean_cluster_jaccard",
                "std",
            ),
            minimum_jaccard=(
                "minimum_cluster_jaccard",
                "min",
            ),
            runtime_mean=(
                "runtime_seconds",
                "mean",
            ),
        )
        .reset_index()
    )

    cluster_summary = (
        jaccard_results.groupby(
            [
                "k",
                "reference_cluster",
            ]
        )
        .agg(
            cluster_jaccard_mean=(
                "jaccard",
                "mean",
            ),
            cluster_jaccard_std=(
                "jaccard",
                "std",
            ),
            cluster_jaccard_minimum=(
                "jaccard",
                "min",
            ),
            reference_cluster_size=(
                "reference_cluster_size",
                "first",
            ),
        )
        .reset_index()
    )

    minimum_cluster_stability = (
        cluster_summary.groupby("k")
        .agg(
            worst_cluster_mean_jaccard=(
                "cluster_jaccard_mean",
                "min",
            ),
            best_cluster_mean_jaccard=(
                "cluster_jaccard_mean",
                "max",
            ),
        )
        .reset_index()
    )

    return run_summary.merge(
        minimum_cluster_stability,
        on="k",
        how="left",
    )


def plot_k_stability(
    summary: pd.DataFrame,
) -> Path:
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(14, 5),
    )

    axes[0].errorbar(
        summary["k"],
        summary["ari_mean"],
        yerr=summary["ari_std"].fillna(
            0
        ),
        marker="o",
        capsize=4,
        linewidth=2,
    )

    axes[0].set_ylim(
        -0.05,
        1.05,
    )

    axes[0].set_xlabel("k")

    axes[0].set_ylabel(
        "ARI to reference clustering"
    )

    axes[0].set_title(
        "Bootstrap ARI Stability"
    )

    axes[1].errorbar(
        summary["k"],
        summary["mean_jaccard"],
        yerr=summary[
            "jaccard_std"
        ].fillna(0),
        marker="o",
        capsize=4,
        linewidth=2,
        color="#ff7f0e",
        label="Mean cluster Jaccard",
    )

    axes[1].plot(
        summary["k"],
        summary[
            "worst_cluster_mean_jaccard"
        ],
        marker="s",
        linestyle="--",
        color="#d62728",
        label="Worst cluster",
    )

    axes[1].set_ylim(
        -0.05,
        1.05,
    )

    axes[1].set_xlabel("k")

    axes[1].set_ylabel(
        "Jaccard stability"
    )

    axes[1].set_title(
        "Bootstrap Cluster-Membership Stability"
    )

    axes[1].legend()

    figure.suptitle(
        "Bootstrap Stability Across Candidate k"
    )

    return save_figure(
        "bootstrap_stability_by_k.png"
    )


def plot_jaccard_distribution(
    jaccard_results: pd.DataFrame,
) -> Path:
    plt.figure(
        figsize=(11, 6)
    )

    sns.boxplot(
        data=jaccard_results,
        x="k",
        y="jaccard",
    )

    sns.stripplot(
        data=jaccard_results,
        x="k",
        y="jaccard",
        color="black",
        alpha=0.25,
        size=2,
    )

    plt.ylim(
        -0.05,
        1.05,
    )

    plt.xlabel("k")

    plt.ylabel(
        "Best matched-cluster Jaccard"
    )

    plt.title(
        "Bootstrap Jaccard Stability Distribution"
    )

    return save_figure(
        "bootstrap_jaccard_distribution.png"
    )


def plot_coclustering_heatmap(
    probability_matrix: np.ndarray,
    reference_labels: np.ndarray,
    selected_k: int,
) -> Path:
    order = np.argsort(
        reference_labels,
        kind="stable",
    )

    ordered_matrix = (
        probability_matrix[
            np.ix_(
                order,
                order,
            )
        ]
    )

    ordered_labels = (
        reference_labels[
            order
        ]
    )

    boundaries = np.flatnonzero(
        np.diff(
            ordered_labels
        )
        != 0
    ) + 1

    plt.figure(
        figsize=(10, 9)
    )

    axis = sns.heatmap(
        ordered_matrix,
        cmap="viridis",
        vmin=0,
        vmax=1,
        xticklabels=False,
        yticklabels=False,
        cbar_kws={
            "label": (
                "Bootstrap co-clustering probability"
            )
        },
    )

    for boundary in boundaries:
        axis.axhline(
            boundary,
            color="white",
            linewidth=1.0,
        )

        axis.axvline(
            boundary,
            color="white",
            linewidth=1.0,
        )

    plt.xlabel(
        "Anchor records ordered by "
        "reference cluster"
    )

    plt.ylabel(
        "Anchor records ordered by "
        "reference cluster"
    )

    plt.title(
        "Bootstrap Co-clustering "
        f"Probability Matrix, k={selected_k}"
    )

    return save_figure(
        "bootstrap_coclustering_heatmap.png"
    )


def calculate_matrix_diagnostics(
    probability_matrix: np.ndarray,
    reference_labels: np.ndarray,
) -> dict[str, float]:
    same_cluster = (
        reference_labels[:, None]
        == reference_labels[None, :]
    )

    different_cluster = (
        ~same_cluster
    )

    diagonal = np.eye(
        len(reference_labels),
        dtype=bool,
    )

    within_mask = (
        same_cluster
        & ~diagonal
    )

    between_mask = (
        different_cluster
    )

    within_values = (
        probability_matrix[
            within_mask
        ]
    )

    between_values = (
        probability_matrix[
            between_mask
        ]
    )

    return {
        "mean_within_cluster_probability": float(
            np.mean(
                within_values
            )
        ),
        "median_within_cluster_probability": float(
            np.median(
                within_values
            )
        ),
        "minimum_within_cluster_probability": float(
            np.min(
                within_values
            )
        ),
        "mean_between_cluster_probability": float(
            np.mean(
                between_values
            )
        ),
        "median_between_cluster_probability": float(
            np.median(
                between_values
            )
        ),
        "maximum_between_cluster_probability": float(
            np.max(
                between_values
            )
        ),
        "probability_separation": float(
            np.mean(
                within_values
            )
            - np.mean(
                between_values
            )
        ),
    }


def write_report(
    selected_k: int,
    summary: pd.DataFrame,
    diagnostics: dict[str, float],
    bootstrap_runs: int,
    anchor_size: int,
) -> None:
    selected_row = summary[
        summary["k"] == selected_k
    ]

    if selected_row.empty:
        raise ValueError(
            "Selected k is missing from "
            "stability summary"
        )

    selected = selected_row.iloc[0]

    best_by_jaccard = summary.sort_values(
        [
            "mean_jaccard",
            "ari_mean",
        ],
        ascending=[
            False,
            False,
        ],
    ).iloc[0]

    lines = [
        "# Bootstrap Stability and Co-clustering Analysis",
        "",
        f"Bootstrap repetitions per k: {bootstrap_runs}",
        "",
        f"Fixed anchor records: {anchor_size}",
        "",
        "A reference K-Means model was fitted for each candidate k. "
        "Each bootstrap repetition resampled the evaluation data with "
        "replacement, fitted a new K-Means model, and predicted cluster "
        "membership for the same fixed anchor records.",
        "",
        "The fraud label was not used in resampling, model fitting, "
        "cluster matching, or stability selection.",
        "",
        "## Selected K-Means k",
        "",
        f"- Selected k from the original silhouette search: {selected_k}",
        f"- Mean bootstrap ARI: {selected['ari_mean']:.6f}",
        f"- Mean cluster Jaccard: {selected['mean_jaccard']:.6f}",
        f"- Worst-cluster mean Jaccard: "
        f"{selected['worst_cluster_mean_jaccard']:.6f}",
        "",
        "## Most Stable Candidate",
        "",
        f"- k: {int(best_by_jaccard['k'])}",
        f"- Mean cluster Jaccard: "
        f"{best_by_jaccard['mean_jaccard']:.6f}",
        f"- Mean ARI: {best_by_jaccard['ari_mean']:.6f}",
        "",
        "## Co-clustering Matrix Diagnostics",
        "",
        f"- Mean within-cluster probability: "
        f"{diagnostics['mean_within_cluster_probability']:.6f}",
        f"- Mean between-cluster probability: "
        f"{diagnostics['mean_between_cluster_probability']:.6f}",
        f"- Probability separation: "
        f"{diagnostics['probability_separation']:.6f}",
        "",
        "High within-cluster probability combined with low "
        "between-cluster probability indicates a stable partition. "
        "Blurred blocks or high between-cluster probability indicate "
        "unstable boundaries.",
        "",
        "Cluster-level Jaccard scores are obtained by matching each "
        "reference cluster to the bootstrap cluster with maximum "
        "membership-set Jaccard similarity.",
    ]

    (
        REPORT_DIR
        / "bootstrap_coclustering_analysis.md"
    ).write_text(
        "\n".join(
            lines
        ),
        encoding="utf-8",
    )


def main() -> None:
    ensure_directories()
    setup_logging()
    setup_plots()

    params = load_yaml(
        PARAMS_PATH
    )

    random_seed = int(
        params["project"][
            "random_seed"
        ]
    )

    k_min = int(
        params["cluster_search"][
            "k_min"
        ]
    )

    k_max = int(
        params["cluster_search"][
            "k_max"
        ]
    )

    k_values = list(
        range(
            k_min,
            k_max + 1,
        )
    )

    selected_k = determine_selected_k(
        params
    )

    if selected_k not in k_values:
        raise ValueError(
            "Selected k is outside the "
            "configured search range"
        )

    bootstrap_runs = int(
        params.get(
            "stability",
            {},
        ).get(
            "bootstrap_runs",
            20,
        )
    )

    bootstrap_runs = max(
        bootstrap_runs,
        20,
    )

    anchor_size = int(
        params["sampling"].get(
            "stability_anchor_size",
            1200,
        )
    )

    n_init = int(
        params["kmeans"][
            "n_init"
        ]
    )

    x, selected_indices = (
        load_analysis_data(
            params
        )
    )

    anchor_size = min(
        anchor_size,
        len(x),
    )

    rng = np.random.default_rng(
        random_seed
    )

    anchor_positions = rng.choice(
        len(x),
        size=anchor_size,
        replace=False,
    )

    anchor_positions = np.sort(
        anchor_positions
    )

    anchor_array_indices = (
        selected_indices[
            anchor_positions
        ]
    )

    pd.DataFrame(
        {
            "anchor_position": (
                anchor_positions
            ),
            "array_index": (
                anchor_array_indices
            ),
        }
    ).to_csv(
        REPORT_DIR
        / "bootstrap_anchor_indices.csv",
        index=False,
    )

    all_run_results = []
    all_jaccard_results = []

    selected_probability_matrix = None
    selected_reference_labels = None

    for k in k_values:
        (
            run_results,
            jaccard_results,
            probability_matrix,
            reference_labels,
        ) = run_bootstrap_for_k(
            x=x,
            anchor_positions=(
                anchor_positions
            ),
            k=k,
            bootstrap_runs=(
                bootstrap_runs
            ),
            n_init=n_init,
            random_seed=random_seed,
            store_matrix=(
                k == selected_k
            ),
        )

        all_run_results.append(
            run_results
        )

        all_jaccard_results.append(
            jaccard_results
        )

        if k == selected_k:
            selected_probability_matrix = (
                probability_matrix
            )

            selected_reference_labels = (
                reference_labels
            )

    combined_runs = pd.concat(
        all_run_results,
        ignore_index=True,
    )

    combined_jaccards = pd.concat(
        all_jaccard_results,
        ignore_index=True,
    )

    summary = create_k_stability_summary(
        run_results=combined_runs,
        jaccard_results=(
            combined_jaccards
        ),
    )

    combined_runs.to_csv(
        REPORT_DIR
        / "bootstrap_run_stability.csv",
        index=False,
    )

    combined_jaccards.to_csv(
        REPORT_DIR
        / "bootstrap_cluster_jaccard.csv",
        index=False,
    )

    summary.to_csv(
        REPORT_DIR
        / "bootstrap_k_stability_summary.csv",
        index=False,
    )

    if (
        selected_probability_matrix
        is None
        or selected_reference_labels
        is None
    ):
        raise RuntimeError(
            "Selected co-clustering matrix "
            "was not generated"
        )

    np.save(
        PROCESSED_DIR
        / "phase2_bootstrap_coclustering.npy",
        selected_probability_matrix.astype(
            np.float32
        ),
    )

    pd.DataFrame(
        {
            "anchor_position": (
                anchor_positions
            ),
            "array_index": (
                anchor_array_indices
            ),
            "reference_cluster": (
                selected_reference_labels
            ),
        }
    ).to_parquet(
        PROCESSED_DIR
        / "phase2_bootstrap_anchor_labels.parquet",
        index=False,
    )

    stability_figure = (
        plot_k_stability(
            summary
        )
    )

    jaccard_figure = (
        plot_jaccard_distribution(
            combined_jaccards
        )
    )

    heatmap_figure = (
        plot_coclustering_heatmap(
            probability_matrix=(
                selected_probability_matrix
            ),
            reference_labels=(
                selected_reference_labels
            ),
            selected_k=selected_k,
        )
    )

    diagnostics = (
        calculate_matrix_diagnostics(
            probability_matrix=(
                selected_probability_matrix
            ),
            reference_labels=(
                selected_reference_labels
            ),
        )
    )

    save_json(
        diagnostics,
        REPORT_DIR
        / "bootstrap_coclustering_diagnostics.json",
    )

    write_report(
        selected_k=selected_k,
        summary=summary,
        diagnostics=diagnostics,
        bootstrap_runs=(
            bootstrap_runs
        ),
        anchor_size=anchor_size,
    )

    selected_summary = summary[
        summary["k"] == selected_k
    ].iloc[0]

    completion_record = {
        "status": "completed",
        "bootstrap_runs_per_k": int(
            bootstrap_runs
        ),
        "candidate_k_values": (
            k_values
        ),
        "selected_k": int(
            selected_k
        ),
        "anchor_size": int(
            anchor_size
        ),
        "selected_k_mean_ari": float(
            selected_summary[
                "ari_mean"
            ]
        ),
        "selected_k_mean_jaccard": float(
            selected_summary[
                "mean_jaccard"
            ]
        ),
        "selected_k_worst_cluster_jaccard": float(
            selected_summary[
                "worst_cluster_mean_jaccard"
            ]
        ),
        "matrix_diagnostics": (
            diagnostics
        ),
        "generated_figures": [
            str(
                path.relative_to(ROOT)
            )
            for path in [
                stability_figure,
                jaccard_figure,
                heatmap_figure,
            ]
        ],
    }

    save_json(
        completion_record,
        REPORT_DIR
        / "phase2_completion_step3_record.json",
    )

    logging.info(
        "Phase 2 completion step 3 succeeded"
    )


if __name__ == "__main__":
    main()