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
from scipy.cluster.hierarchy import (
    cophenet,
    dendrogram,
    fcluster,
    linkage,
)
from scipy.spatial.distance import pdist
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    calinski_harabasz_score,
    completeness_score,
    davies_bouldin_score,
    fowlkes_mallows_score,
    homogeneity_score,
    normalized_mutual_info_score,
    silhouette_score,
    v_measure_score,
)


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

LINKAGE_METHODS = [
    "single",
    "complete",
    "average",
    "ward",
]


def ensure_directories() -> None:
    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    FIGURE_DIR.mkdir(
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
                / "phase2_completion_step2.log",
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


def purity_score(
    y_true: np.ndarray,
    labels: np.ndarray,
) -> float:
    table = pd.crosstab(
        pd.Series(
            y_true,
            name="true",
        ),
        pd.Series(
            labels,
            name="cluster",
        ),
    )

    if table.empty:
        return float("nan")

    return float(
        table.max(axis=0).sum()
        / table.to_numpy().sum()
    )


def load_analysis_data(
    params: dict[str, Any],
) -> tuple[
    np.ndarray,
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
            "y",
            "train_indices",
        }

        missing = (
            required
            - set(arrays.files)
        )

        if missing:
            raise KeyError(
                f"Missing arrays: {sorted(missing)}"
            )

        x_all = arrays[
            "X_pca"
        ].astype(np.float64)

        y_all = arrays[
            "y"
        ].astype(np.int8)

        train_indices = arrays[
            "train_indices"
        ].astype(np.int64)

    if PHASE2_SAMPLE_PATH.exists():
        phase2_sample = pd.read_csv(
            PHASE2_SAMPLE_PATH
        )

        if (
            "array_index"
            not in phase2_sample.columns
        ):
            raise KeyError(
                "array_index is missing from "
                "evaluation_sample_indices.csv"
            )

        candidate_indices = phase2_sample[
            "array_index"
        ].to_numpy(dtype=np.int64)
    else:
        candidate_indices = train_indices

        logging.warning(
            "Phase 2 sample file was not found. "
            "Using Phase 1 training indices."
        )

    candidate_indices = np.unique(
        candidate_indices
    )

    if np.any(candidate_indices < 0):
        raise IndexError(
            "Negative sample indices found"
        )

    if np.any(
        candidate_indices >= len(x_all)
    ):
        raise IndexError(
            "Sample indices exceed X_pca row count"
        )

    hierarchy_size = int(
        params["sampling"][
            "hierarchy_dendrogram_size"
        ]
    )

    hierarchy_size = min(
        hierarchy_size,
        len(candidate_indices),
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
        candidate_indices,
        size=hierarchy_size,
        replace=False,
    )

    selected_indices = np.sort(
        selected_indices
    )

    x = x_all[
        selected_indices
    ]

    y = y_all[
        selected_indices
    ]

    if not np.isfinite(x).all():
        raise ValueError(
            "Hierarchical input contains "
            "NaN or infinite values"
        )

    return (
        x,
        y,
        selected_indices,
    )


def calculate_internal_metrics(
    x: np.ndarray,
    labels: np.ndarray,
    random_seed: int,
    silhouette_sample_size: int,
) -> dict[str, float]:
    unique_labels = np.unique(
        labels
    )

    valid = (
        len(unique_labels) >= 2
        and len(unique_labels) < len(labels)
    )

    if not valid:
        return {
            "silhouette": float("nan"),
            "davies_bouldin": float("nan"),
            "calinski_harabasz": float("nan"),
        }

    sample_size = min(
        silhouette_sample_size,
        len(x),
    )

    return {
        "silhouette": float(
            silhouette_score(
                x,
                labels,
                metric="euclidean",
                sample_size=sample_size,
                random_state=random_seed,
            )
        ),
        "davies_bouldin": float(
            davies_bouldin_score(
                x,
                labels,
            )
        ),
        "calinski_harabasz": float(
            calinski_harabasz_score(
                x,
                labels,
            )
        ),
    }


def calculate_external_metrics(
    y_true: np.ndarray,
    labels: np.ndarray,
) -> dict[str, float]:
    return {
        "ari": float(
            adjusted_rand_score(
                y_true,
                labels,
            )
        ),
        "nmi": float(
            normalized_mutual_info_score(
                y_true,
                labels,
            )
        ),
        "ami": float(
            adjusted_mutual_info_score(
                y_true,
                labels,
            )
        ),
        "fowlkes_mallows": float(
            fowlkes_mallows_score(
                y_true,
                labels,
            )
        ),
        "homogeneity": float(
            homogeneity_score(
                y_true,
                labels,
            )
        ),
        "completeness": float(
            completeness_score(
                y_true,
                labels,
            )
        ),
        "v_measure": float(
            v_measure_score(
                y_true,
                labels,
            )
        ),
        "purity": purity_score(
            y_true,
            labels,
        ),
    }


def evaluate_partition(
    x: np.ndarray,
    y_true: np.ndarray,
    labels: np.ndarray,
    linkage_method: str,
    strategy: str,
    requested_k: int,
    cut_height: float,
    cophenetic_correlation: float,
    runtime_seconds: float,
    random_seed: int,
    silhouette_sample_size: int,
) -> dict[str, Any]:
    internal = calculate_internal_metrics(
        x=x,
        labels=labels,
        random_seed=random_seed,
        silhouette_sample_size=(
            silhouette_sample_size
        ),
    )

    external = calculate_external_metrics(
        y_true=y_true,
        labels=labels,
    )

    unique_labels, cluster_sizes = (
        np.unique(
            labels,
            return_counts=True,
        )
    )

    return {
        "linkage": linkage_method,
        "strategy": strategy,
        "requested_k": (
            int(requested_k)
            if requested_k >= 0
            else None
        ),
        "actual_k": int(
            len(unique_labels)
        ),
        "cut_height": float(
            cut_height
        ),
        "cophenetic_correlation": float(
            cophenetic_correlation
        ),
        "minimum_cluster_size": int(
            cluster_sizes.min()
        ),
        "maximum_cluster_size": int(
            cluster_sizes.max()
        ),
        "runtime_seconds": float(
            runtime_seconds
        ),
        **internal,
        **external,
    }


def calculate_height_for_k(
    linkage_matrix: np.ndarray,
    k: int,
) -> float:
    number_of_records = (
        linkage_matrix.shape[0] + 1
    )

    if k <= 1:
        return float(
            linkage_matrix[-1, 2]
        )

    if k >= number_of_records:
        return 0.0

    next_merge_index = (
        number_of_records - k
    )

    previous_merge_index = (
        next_merge_index - 1
    )

    if previous_merge_index < 0:
        lower_height = 0.0
    else:
        lower_height = float(
            linkage_matrix[
                previous_merge_index,
                2,
            ]
        )

    upper_height = float(
        linkage_matrix[
            next_merge_index,
            2,
        ]
    )

    if upper_height == lower_height:
        return upper_height

    return (
        lower_height
        + upper_height
    ) / 2.0


def select_fixed_height(
    linkage_matrix: np.ndarray,
    configured_ratio: float,
) -> tuple[
    float,
    np.ndarray,
    float,
]:
    if not 0 < configured_ratio < 1:
        raise ValueError(
            "fixed_height_ratio must be "
            "between zero and one"
        )

    maximum_height = float(
        linkage_matrix[-1, 2]
    )

    candidate_ratios = [
        configured_ratio,
        0.80,
        0.75,
        0.70,
        0.65,
        0.60,
        0.55,
        0.50,
        0.45,
        0.40,
        0.35,
        0.30,
    ]

    unique_ratios = []

    for ratio in candidate_ratios:
        if ratio not in unique_ratios:
            unique_ratios.append(
                ratio
            )

    selected_height = None
    selected_labels = None
    selected_ratio = None

    for ratio in unique_ratios:
        height = (
            ratio
            * maximum_height
        )

        labels = (
            fcluster(
                linkage_matrix,
                t=height,
                criterion="distance",
            )
            - 1
        ).astype(np.int32)

        cluster_count = len(
            np.unique(labels)
        )

        if 2 <= cluster_count < len(
            labels
        ):
            selected_height = height
            selected_labels = labels
            selected_ratio = ratio
            break

    if selected_labels is None:
        fallback_k = 2

        selected_height = (
            calculate_height_for_k(
                linkage_matrix,
                fallback_k,
            )
        )

        selected_labels = (
            fcluster(
                linkage_matrix,
                t=fallback_k,
                criterion="maxclust",
            )
            - 1
        ).astype(np.int32)

        selected_ratio = (
            selected_height
            / maximum_height
            if maximum_height > 0
            else 0.0
        )

    return (
        float(selected_height),
        selected_labels,
        float(selected_ratio),
    )


def run_linkage_analysis(
    x: np.ndarray,
    y_true: np.ndarray,
    linkage_method: str,
    k_values: list[int],
    fixed_height_ratio: float,
    random_seed: int,
    silhouette_sample_size: int,
) -> tuple[
    np.ndarray,
    pd.DataFrame,
    dict[str, Any],
    np.ndarray,
    np.ndarray,
]:
    logging.info(
        "Running hierarchical analysis: %s",
        linkage_method,
    )

    pairwise_distances = pdist(
        x,
        metric="euclidean",
    )

    start = time.perf_counter()

    linkage_matrix = linkage(
        x,
        method=linkage_method,
        metric="euclidean",
    )

    runtime = (
        time.perf_counter()
        - start
    )

    cophenetic_value, _ = cophenet(
        linkage_matrix,
        pairwise_distances,
    )

    search_rows = []
    labels_by_k = {}

    for requested_k in k_values:
        labels = (
            fcluster(
                linkage_matrix,
                t=requested_k,
                criterion="maxclust",
            )
            - 1
        ).astype(np.int32)

        labels_by_k[
            requested_k
        ] = labels

        row = evaluate_partition(
            x=x,
            y_true=y_true,
            labels=labels,
            linkage_method=linkage_method,
            strategy=(
                "silhouette_candidate"
            ),
            requested_k=requested_k,
            cut_height=(
                calculate_height_for_k(
                    linkage_matrix,
                    requested_k,
                )
            ),
            cophenetic_correlation=(
                cophenetic_value
            ),
            runtime_seconds=runtime,
            random_seed=random_seed,
            silhouette_sample_size=(
                silhouette_sample_size
            ),
        )

        search_rows.append(row)

    search_results = pd.DataFrame(
        search_rows
    )

    valid_search = search_results.dropna(
        subset=["silhouette"]
    )

    if valid_search.empty:
        raise ValueError(
            f"No valid silhouette cut for "
            f"{linkage_method}"
        )

    best_row = valid_search.sort_values(
        [
            "silhouette",
            "davies_bouldin",
        ],
        ascending=[
            False,
            True,
        ],
    ).iloc[0]

    best_k = int(
        best_row["requested_k"]
    )

    silhouette_labels = labels_by_k[
        best_k
    ]

    silhouette_height = (
        calculate_height_for_k(
            linkage_matrix,
            best_k,
        )
    )

    (
        fixed_height,
        fixed_labels,
        actual_fixed_ratio,
    ) = select_fixed_height(
        linkage_matrix=linkage_matrix,
        configured_ratio=(
            fixed_height_ratio
        ),
    )

    fixed_result = evaluate_partition(
        x=x,
        y_true=y_true,
        labels=fixed_labels,
        linkage_method=linkage_method,
        strategy="fixed_height",
        requested_k=-1,
        cut_height=fixed_height,
        cophenetic_correlation=(
            cophenetic_value
        ),
        runtime_seconds=runtime,
        random_seed=random_seed,
        silhouette_sample_size=(
            silhouette_sample_size
        ),
    )

    fixed_result[
        "fixed_height_ratio"
    ] = actual_fixed_ratio

    silhouette_result = (
        evaluate_partition(
            x=x,
            y_true=y_true,
            labels=silhouette_labels,
            linkage_method=(
                linkage_method
            ),
            strategy=(
                "maximum_silhouette"
            ),
            requested_k=best_k,
            cut_height=(
                silhouette_height
            ),
            cophenetic_correlation=(
                cophenetic_value
            ),
            runtime_seconds=runtime,
            random_seed=random_seed,
            silhouette_sample_size=(
                silhouette_sample_size
            ),
        )
    )

    silhouette_result[
        "fixed_height_ratio"
    ] = None

    strategy_comparison = {
        "linkage": linkage_method,
        "fixed_height": float(
            fixed_height
        ),
        "fixed_height_ratio": float(
            actual_fixed_ratio
        ),
        "fixed_height_k": int(
            len(
                np.unique(
                    fixed_labels
                )
            )
        ),
        "maximum_silhouette_k": (
            best_k
        ),
        "maximum_silhouette_height": float(
            silhouette_height
        ),
        "ari_between_strategies": float(
            adjusted_rand_score(
                fixed_labels,
                silhouette_labels,
            )
        ),
        "cophenetic_correlation": float(
            cophenetic_value
        ),
    }

    final_rows = pd.DataFrame(
        [
            fixed_result,
            silhouette_result,
        ]
    )

    return (
        linkage_matrix,
        final_rows,
        strategy_comparison,
        fixed_labels,
        silhouette_labels,
    )


def plot_dendrogram_comparison(
    linkage_matrices: dict[
        str,
        np.ndarray,
    ],
    strategy_details: dict[
        str,
        dict[str, Any],
    ],
) -> Path:
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(17, 11),
    )

    axes = axes.ravel()

    for axis, linkage_method in zip(
        axes,
        LINKAGE_METHODS,
    ):
        linkage_matrix = (
            linkage_matrices[
                linkage_method
            ]
        )

        details = strategy_details[
            linkage_method
        ]

        dendrogram(
            linkage_matrix,
            truncate_mode="lastp",
            p=40,
            no_labels=True,
            show_contracted=True,
            ax=axis,
            color_threshold=None,
        )

        axis.axhline(
            details["fixed_height"],
            color="red",
            linestyle="--",
            linewidth=2,
            label=(
                "Fixed-height cut "
                f"(k={details['fixed_height_k']})"
            ),
        )

        axis.axhline(
            details[
                "maximum_silhouette_height"
            ],
            color="green",
            linestyle=":",
            linewidth=2,
            label=(
                "Max-silhouette cut "
                f"(k={details['maximum_silhouette_k']})"
            ),
        )

        axis.set_title(
            linkage_method.capitalize()
            + " linkage"
        )

        axis.set_xlabel(
            "Merged groups"
        )

        axis.set_ylabel(
            "Linkage distance"
        )

        axis.legend(
            loc="upper left",
            fontsize=8,
        )

    figure.suptitle(
        "Hierarchical Clustering: "
        "Fixed-Height and Maximum-Silhouette Cuts",
        fontsize=16,
    )

    return save_figure(
        "hierarchical_cut_strategy_dendrograms.png"
    )


def plot_strategy_metrics(
    comparison: pd.DataFrame,
) -> Path:
    internal_metrics = [
        "silhouette",
        "davies_bouldin",
        "calinski_harabasz",
    ]

    melted = comparison.melt(
        id_vars=[
            "linkage",
            "strategy",
        ],
        value_vars=internal_metrics,
        var_name="metric",
        value_name="value",
    )

    graph = sns.catplot(
        data=melted,
        x="linkage",
        y="value",
        hue="strategy",
        col="metric",
        kind="bar",
        sharey=False,
        height=4.5,
        aspect=1.1,
    )

    graph.set_xticklabels(
        rotation=25
    )

    graph.fig.suptitle(
        "Hierarchical Cut Strategy Comparison",
        y=1.04,
    )

    path = (
        FIGURE_DIR
        / "hierarchical_cut_strategy_metrics.png"
    )

    graph.savefig(
        path,
        bbox_inches="tight",
        dpi=200,
    )

    plt.close("all")

    logging.info(
        "Saved figure: %s",
        path,
    )

    return path


def plot_strategy_agreement(
    details_frame: pd.DataFrame,
) -> Path:
    plt.figure(
        figsize=(9, 5)
    )

    sns.barplot(
        data=details_frame,
        x="linkage",
        y="ari_between_strategies",
        hue="linkage",
        legend=False,
    )

    plt.ylim(
        -0.05,
        1.05,
    )

    plt.axhline(
        1.0,
        color="black",
        linestyle=":",
    )

    plt.xlabel(
        "Linkage method"
    )

    plt.ylabel(
        "ARI between fixed-height "
        "and max-silhouette cuts"
    )

    plt.title(
        "Agreement Between Hierarchical "
        "Cutting Strategies"
    )

    return save_figure(
        "hierarchical_cut_strategy_agreement.png"
    )


def save_cluster_assignments(
    selected_indices: np.ndarray,
    y_true: np.ndarray,
    fixed_labels_by_method: dict[
        str,
        np.ndarray,
    ],
    silhouette_labels_by_method: dict[
        str,
        np.ndarray,
    ],
) -> None:
    output = pd.DataFrame(
        {
            "array_index": (
                selected_indices
            ),
            "Class_external_only": (
                y_true
            ),
        }
    )

    for method in LINKAGE_METHODS:
        output[
            f"{method}_fixed_height"
        ] = fixed_labels_by_method[
            method
        ]

        output[
            f"{method}_maximum_silhouette"
        ] = (
            silhouette_labels_by_method[
                method
            ]
        )

    output.to_parquet(
        ROOT
        / "data"
        / "processed"
        / "phase2_hierarchical_cut_assignments.parquet",
        index=False,
    )


def write_report(
    comparison: pd.DataFrame,
    details_frame: pd.DataFrame,
) -> None:
    valid_rows = comparison.dropna(
        subset=["silhouette"]
    )

    if valid_rows.empty:
        raise ValueError(
            "No valid hierarchical strategy results"
        )

    best_row = valid_rows.sort_values(
        [
            "silhouette",
            "davies_bouldin",
        ],
        ascending=[
            False,
            True,
        ],
    ).iloc[0]

    report_lines = [
        "# Hierarchical Cutting Strategy Analysis",
        "",
        "Two flat-clustering strategies were compared "
        "for single, complete, average, and Ward linkage.",
        "",
        "The fixed-height strategy uses a documented "
        "fraction of the maximum dendrogram height. "
        "The same deterministic rule is applied without "
        "consulting the fraud label.",
        "",
        "The maximum-silhouette strategy evaluates the "
        "candidate k range and selects the partition with "
        "the highest internal silhouette coefficient. "
        "External metrics are computed only afterward.",
        "",
        "Best internal result:",
        "",
        f"- Linkage: {best_row['linkage']}",
        f"- Strategy: {best_row['strategy']}",
        f"- Requested k: {best_row['requested_k']}",
        f"- Actual k: {best_row['actual_k']}",
        f"- Silhouette: {best_row['silhouette']:.6f}",
        f"- Davies-Bouldin: "
        f"{best_row['davies_bouldin']:.6f}",
        f"- Calinski-Harabasz: "
        f"{best_row['calinski_harabasz']:.6f}",
        "",
        "Agreement between the two strategies:",
        "",
    ]

    for row in details_frame.itertuples(
        index=False
    ):
        report_lines.extend(
            [
                f"- {row.linkage}: "
                f"ARI={row.ari_between_strategies:.6f}, "
                f"fixed k={row.fixed_height_k}, "
                f"max-silhouette k="
                f"{row.maximum_silhouette_k}",
            ]
        )

    report_lines.extend(
        [
            "",
            "Interpretation:",
            "",
            "ARI values near one indicate that both "
            "cutting strategies recover nearly the same "
            "partition. Low ARI indicates that the flat "
            "clustering is sensitive to how the "
            "dendrogram is cut.",
            "",
            "Single linkage may exhibit chaining, while "
            "Ward linkage typically produces tighter and "
            "more balanced clusters. Cophenetic "
            "correlation measures how faithfully each "
            "dendrogram preserves the original pairwise "
            "distances.",
            "",
            "The fraud label was not used to choose the "
            "linkage, height, or number of clusters.",
        ]
    )

    (
        REPORT_DIR
        / "hierarchical_cut_strategy_analysis.md"
    ).write_text(
        "\n".join(
            report_lines
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

    silhouette_sample_size = int(
        params["sampling"][
            "silhouette_sample_size"
        ]
    )

    hierarchical_params = params.get(
        "hierarchical",
        {},
    )

    fixed_height_ratio = float(
        hierarchical_params.get(
            "fixed_height_ratio",
            0.70,
        )
    )

    x, y_true, selected_indices = (
        load_analysis_data(
            params
        )
    )

    linkage_matrices = {}
    comparison_frames = []
    details = {}
    fixed_labels_by_method = {}
    silhouette_labels_by_method = {}

    for linkage_method in (
        LINKAGE_METHODS
    ):
        (
            linkage_matrix,
            comparison,
            strategy_details,
            fixed_labels,
            silhouette_labels,
        ) = run_linkage_analysis(
            x=x,
            y_true=y_true,
            linkage_method=(
                linkage_method
            ),
            k_values=k_values,
            fixed_height_ratio=(
                fixed_height_ratio
            ),
            random_seed=random_seed,
            silhouette_sample_size=(
                silhouette_sample_size
            ),
        )

        linkage_matrices[
            linkage_method
        ] = linkage_matrix

        comparison_frames.append(
            comparison
        )

        details[
            linkage_method
        ] = strategy_details

        fixed_labels_by_method[
            linkage_method
        ] = fixed_labels

        silhouette_labels_by_method[
            linkage_method
        ] = silhouette_labels

    comparison_results = pd.concat(
        comparison_frames,
        ignore_index=True,
    )

    details_frame = pd.DataFrame(
        list(
            details.values()
        )
    )

    comparison_results.to_csv(
        REPORT_DIR
        / "hierarchical_cut_strategy_comparison.csv",
        index=False,
    )

    details_frame.to_csv(
        REPORT_DIR
        / "hierarchical_cut_strategy_agreement.csv",
        index=False,
    )

    save_cluster_assignments(
        selected_indices=(
            selected_indices
        ),
        y_true=y_true,
        fixed_labels_by_method=(
            fixed_labels_by_method
        ),
        silhouette_labels_by_method=(
            silhouette_labels_by_method
        ),
    )

    dendrogram_figure = (
        plot_dendrogram_comparison(
            linkage_matrices=(
                linkage_matrices
            ),
            strategy_details=details,
        )
    )

    metrics_figure = (
        plot_strategy_metrics(
            comparison_results
        )
    )

    agreement_figure = (
        plot_strategy_agreement(
            details_frame
        )
    )

    write_report(
        comparison=comparison_results,
        details_frame=details_frame,
    )

    best_row = (
        comparison_results
        .dropna(
            subset=["silhouette"]
        )
        .sort_values(
            [
                "silhouette",
                "davies_bouldin",
            ],
            ascending=[
                False,
                True,
            ],
        )
        .iloc[0]
    )

    completion_record = {
        "status": "completed",
        "sample_size": int(
            len(x)
        ),
        "candidate_k_values": (
            k_values
        ),
        "configured_fixed_height_ratio": (
            fixed_height_ratio
        ),
        "linkages": LINKAGE_METHODS,
        "best_linkage": str(
            best_row["linkage"]
        ),
        "best_strategy": str(
            best_row["strategy"]
        ),
        "best_actual_k": int(
            best_row["actual_k"]
        ),
        "best_silhouette": float(
            best_row["silhouette"]
        ),
        "generated_figures": [
            str(
                path.relative_to(ROOT)
            )
            for path in [
                dendrogram_figure,
                metrics_figure,
                agreement_figure,
            ]
        ],
    }

    save_json(
        completion_record,
        REPORT_DIR
        / "phase2_completion_step2_record.json",
    )

    logging.info(
        "Phase 2 completion step 2 succeeded"
    )


if __name__ == "__main__":
    main()