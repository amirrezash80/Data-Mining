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
from sklearn.cluster import AgglomerativeClustering
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

PROCESSED_DIR = (
    ROOT
    / "data"
    / "processed"
)

DISTANCE_METRICS = [
    "euclidean",
    "manhattan",
]


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
                / "phase2_completion_step4.log",
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


def purity_score(
    y_true: np.ndarray,
    labels: np.ndarray,
) -> float:
    contingency = pd.crosstab(
        pd.Series(
            y_true,
            name="true",
        ),
        pd.Series(
            labels,
            name="cluster",
        ),
    )

    if contingency.empty:
        return float("nan")

    return float(
        contingency.max(axis=0).sum()
        / contingency.to_numpy().sum()
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
        required_arrays = {
            "X_pca",
            "y",
            "train_indices",
        }

        missing_arrays = (
            required_arrays
            - set(arrays.files)
        )

        if missing_arrays:
            raise KeyError(
                "Missing Phase 1 arrays: "
                f"{sorted(missing_arrays)}"
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

        candidate_indices = sample_table[
            "array_index"
        ].to_numpy(dtype=np.int64)

        logging.info(
            "Using Phase 2 evaluation sample"
        )
    else:
        candidate_indices = train_indices

        logging.warning(
            "Phase 2 evaluation sample was not found. "
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
            "Sample index exceeds X_pca row count"
        )

    sampling_params = params.get(
        "sampling",
        {},
    )

    comparison_size = int(
        sampling_params.get(
            "metric_comparison_size",
            sampling_params.get(
                "hierarchy_dendrogram_size",
                2500,
            ),
        )
    )

    comparison_size = min(
        comparison_size,
        len(candidate_indices),
    )

    if comparison_size < 10:
        raise ValueError(
            "Distance comparison sample is too small"
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
        size=comparison_size,
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
            "Distance-comparison matrix contains "
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
    distance_metric: str,
    random_seed: int,
    silhouette_sample_size: int,
) -> dict[str, float]:
    unique_labels = np.unique(
        labels
    )

    valid_partition = (
        len(unique_labels) >= 2
        and len(unique_labels) < len(labels)
    )

    if not valid_partition:
        return {
            "silhouette_matching_metric": float(
                "nan"
            ),
            "silhouette_euclidean": float(
                "nan"
            ),
            "silhouette_manhattan": float(
                "nan"
            ),
            "davies_bouldin": float(
                "nan"
            ),
            "calinski_harabasz": float(
                "nan"
            ),
        }

    sample_size = min(
        silhouette_sample_size,
        len(x),
    )

    silhouette_matching = float(
        silhouette_score(
            x,
            labels,
            metric=distance_metric,
            sample_size=sample_size,
            random_state=random_seed,
        )
    )

    silhouette_euclidean = float(
        silhouette_score(
            x,
            labels,
            metric="euclidean",
            sample_size=sample_size,
            random_state=random_seed,
        )
    )

    silhouette_manhattan = float(
        silhouette_score(
            x,
            labels,
            metric="manhattan",
            sample_size=sample_size,
            random_state=random_seed,
        )
    )

    davies_bouldin = float(
        davies_bouldin_score(
            x,
            labels,
        )
    )

    calinski_harabasz = float(
        calinski_harabasz_score(
            x,
            labels,
        )
    )

    return {
        "silhouette_matching_metric": (
            silhouette_matching
        ),
        "silhouette_euclidean": (
            silhouette_euclidean
        ),
        "silhouette_manhattan": (
            silhouette_manhattan
        ),
        "davies_bouldin": (
            davies_bouldin
        ),
        "calinski_harabasz": (
            calinski_harabasz
        ),
    }


def calculate_external_metrics(
    y_true: np.ndarray,
    labels: np.ndarray,
) -> dict[str, float]:
    return {
        "ari_external": float(
            adjusted_rand_score(
                y_true,
                labels,
            )
        ),
        "nmi_external": float(
            normalized_mutual_info_score(
                y_true,
                labels,
            )
        ),
        "ami_external": float(
            adjusted_mutual_info_score(
                y_true,
                labels,
            )
        ),
        "fowlkes_mallows_external": float(
            fowlkes_mallows_score(
                y_true,
                labels,
            )
        ),
        "homogeneity_external": float(
            homogeneity_score(
                y_true,
                labels,
            )
        ),
        "completeness_external": float(
            completeness_score(
                y_true,
                labels,
            )
        ),
        "v_measure_external": float(
            v_measure_score(
                y_true,
                labels,
            )
        ),
        "purity_external": purity_score(
            y_true,
            labels,
        ),
    }


def calculate_cluster_size_statistics(
    labels: np.ndarray,
) -> dict[str, Any]:
    cluster_ids, counts = np.unique(
        labels,
        return_counts=True,
    )

    return {
        "actual_k": int(
            len(cluster_ids)
        ),
        "minimum_cluster_size": int(
            counts.min()
        ),
        "maximum_cluster_size": int(
            counts.max()
        ),
        "median_cluster_size": float(
            np.median(counts)
        ),
        "smallest_cluster_fraction": float(
            counts.min() / len(labels)
        ),
        "largest_cluster_fraction": float(
            counts.max() / len(labels)
        ),
    }


def fit_agglomerative_partition(
    x: np.ndarray,
    y_true: np.ndarray,
    distance_metric: str,
    k: int,
    random_seed: int,
    silhouette_sample_size: int,
) -> tuple[
    np.ndarray,
    dict[str, Any],
]:
    logging.info(
        "Agglomerative average linkage: "
        "metric=%s k=%d",
        distance_metric,
        k,
    )

    model = AgglomerativeClustering(
        n_clusters=k,
        metric=distance_metric,
        linkage="average",
        compute_full_tree=True,
    )

    start = time.perf_counter()

    labels = model.fit_predict(
        x
    ).astype(np.int32)

    runtime = (
        time.perf_counter()
        - start
    )

    internal_metrics = (
        calculate_internal_metrics(
            x=x,
            labels=labels,
            distance_metric=(
                distance_metric
            ),
            random_seed=random_seed,
            silhouette_sample_size=(
                silhouette_sample_size
            ),
        )
    )

    external_metrics = (
        calculate_external_metrics(
            y_true=y_true,
            labels=labels,
        )
    )

    size_statistics = (
        calculate_cluster_size_statistics(
            labels
        )
    )

    result = {
        "algorithm": (
            "Agglomerative-average"
        ),
        "distance_metric": (
            distance_metric
        ),
        "requested_k": int(k),
        "runtime_seconds": float(
            runtime
        ),
        **size_statistics,
        **internal_metrics,
        **external_metrics,
    }

    return labels, result


def run_metric_search(
    x: np.ndarray,
    y_true: np.ndarray,
    k_values: list[int],
    random_seed: int,
    silhouette_sample_size: int,
) -> tuple[
    pd.DataFrame,
    dict[str, dict[int, np.ndarray]],
]:
    rows = []

    labels_by_metric: dict[
        str,
        dict[int, np.ndarray],
    ] = {
        metric: {}
        for metric in DISTANCE_METRICS
    }

    for distance_metric in (
        DISTANCE_METRICS
    ):
        for k in k_values:
            labels, result = (
                fit_agglomerative_partition(
                    x=x,
                    y_true=y_true,
                    distance_metric=(
                        distance_metric
                    ),
                    k=k,
                    random_seed=(
                        random_seed
                    ),
                    silhouette_sample_size=(
                        silhouette_sample_size
                    ),
                )
            )

            labels_by_metric[
                distance_metric
            ][k] = labels

            rows.append(result)

    return (
        pd.DataFrame(rows),
        labels_by_metric,
    )


def calculate_metric_agreement(
    labels_by_metric: dict[
        str,
        dict[int, np.ndarray],
    ],
    k_values: list[int],
) -> pd.DataFrame:
    rows = []

    for k in k_values:
        euclidean_labels = (
            labels_by_metric[
                "euclidean"
            ][k]
        )

        manhattan_labels = (
            labels_by_metric[
                "manhattan"
            ][k]
        )

        agreement = float(
            adjusted_rand_score(
                euclidean_labels,
                manhattan_labels,
            )
        )

        rows.append(
            {
                "k": int(k),
                "euclidean_actual_k": int(
                    len(
                        np.unique(
                            euclidean_labels
                        )
                    )
                ),
                "manhattan_actual_k": int(
                    len(
                        np.unique(
                            manhattan_labels
                        )
                    )
                ),
                "ari_between_metrics": (
                    agreement
                ),
                "identical_partition": bool(
                    np.isclose(
                        agreement,
                        1.0,
                    )
                ),
            }
        )

    return pd.DataFrame(rows)


def rank_metric_candidates(
    search_results: pd.DataFrame,
) -> pd.DataFrame:
    ranked_frames = []

    for distance_metric in (
        DISTANCE_METRICS
    ):
        candidates = search_results[
            search_results[
                "distance_metric"
            ]
            == distance_metric
        ].copy()

        candidates = candidates.dropna(
            subset=[
                "silhouette_matching_metric",
                "davies_bouldin",
                "calinski_harabasz",
            ]
        )

        if candidates.empty:
            raise ValueError(
                "No valid candidates for metric "
                f"{distance_metric}"
            )

        candidates[
            "rank_silhouette"
        ] = candidates[
            "silhouette_matching_metric"
        ].rank(
            ascending=False,
            method="min",
        )

        candidates[
            "rank_davies_bouldin"
        ] = candidates[
            "davies_bouldin"
        ].rank(
            ascending=True,
            method="min",
        )

        candidates[
            "rank_calinski_harabasz"
        ] = candidates[
            "calinski_harabasz"
        ].rank(
            ascending=False,
            method="min",
        )

        candidates[
            "internal_rank_sum"
        ] = (
            candidates[
                "rank_silhouette"
            ]
            + candidates[
                "rank_davies_bouldin"
            ]
            + candidates[
                "rank_calinski_harabasz"
            ]
        )

        candidates[
            "selected_within_metric"
        ] = False

        selected_index = (
            candidates.sort_values(
                [
                    "internal_rank_sum",
                    "silhouette_matching_metric",
                    "requested_k",
                ],
                ascending=[
                    True,
                    False,
                    True,
                ],
            ).index[0]
        )

        candidates.loc[
            selected_index,
            "selected_within_metric",
        ] = True

        ranked_frames.append(
            candidates
        )

    return pd.concat(
        ranked_frames,
        ignore_index=True,
    )


def select_best_for_each_metric(
    ranked_results: pd.DataFrame,
) -> pd.DataFrame:
    selected = ranked_results[
        ranked_results[
            "selected_within_metric"
        ]
    ].copy()

    if len(selected) != len(
        DISTANCE_METRICS
    ):
        raise RuntimeError(
            "A best candidate was not selected "
            "for every distance metric"
        )

    return selected.sort_values(
        "distance_metric"
    ).reset_index(
        drop=True
    )


def calculate_selected_partition_agreement(
    selected_results: pd.DataFrame,
    labels_by_metric: dict[
        str,
        dict[int, np.ndarray],
    ],
) -> dict[str, Any]:
    selected_k: dict[str, int] = {}

    selected_labels: dict[
        str,
        np.ndarray,
    ] = {}

    for row in selected_results.itertuples(
        index=False
    ):
        distance_metric = str(
            row.distance_metric
        )

        k = int(
            row.requested_k
        )

        selected_k[
            distance_metric
        ] = k

        selected_labels[
            distance_metric
        ] = labels_by_metric[
            distance_metric
        ][k]

    agreement = float(
        adjusted_rand_score(
            selected_labels[
                "euclidean"
            ],
            selected_labels[
                "manhattan"
            ],
        )
    )

    if agreement >= 0.90:
        sensitivity_level = "low"
    elif agreement >= 0.70:
        sensitivity_level = (
            "moderate"
        )
    else:
        sensitivity_level = "high"

    return {
        "euclidean_selected_k": int(
            selected_k["euclidean"]
        ),
        "manhattan_selected_k": int(
            selected_k["manhattan"]
        ),
        "ari_between_selected_partitions": (
            agreement
        ),
        "distance_sensitivity_level": (
            sensitivity_level
        ),
    }


def save_selected_assignments(
    selected_indices: np.ndarray,
    y_true: np.ndarray,
    selected_results: pd.DataFrame,
    labels_by_metric: dict[
        str,
        dict[int, np.ndarray],
    ],
) -> pd.DataFrame:
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

    for row in selected_results.itertuples(
        index=False
    ):
        distance_metric = str(
            row.distance_metric
        )

        k = int(
            row.requested_k
        )

        output[
            f"{distance_metric}_cluster"
        ] = labels_by_metric[
            distance_metric
        ][k]

    output.to_parquet(
        PROCESSED_DIR
        / "phase2_distance_metric_assignments.parquet",
        index=False,
    )

    return output


def plot_metric_search(
    search_results: pd.DataFrame,
    selected_results: pd.DataFrame,
) -> Path:
    figure, axes = plt.subplots(
        1,
        3,
        figsize=(17, 5),
    )

    metrics = [
        (
            "silhouette_matching_metric",
            "Matching-Metric Silhouette",
            "Higher is better",
        ),
        (
            "davies_bouldin",
            "Davies-Bouldin",
            "Lower is better",
        ),
        (
            "calinski_harabasz",
            "Calinski-Harabasz",
            "Higher is better",
        ),
    ]

    colours = {
        "euclidean": "#1f77b4",
        "manhattan": "#ff7f0e",
    }

    for axis, (
        metric_column,
        title,
        ylabel,
    ) in zip(
        axes,
        metrics,
    ):
        for distance_metric in (
            DISTANCE_METRICS
        ):
            subset = search_results[
                search_results[
                    "distance_metric"
                ]
                == distance_metric
            ].sort_values(
                "requested_k"
            )

            axis.plot(
                subset["requested_k"],
                subset[metric_column],
                marker="o",
                linewidth=2,
                color=colours[
                    distance_metric
                ],
                label=distance_metric,
            )

            selected = selected_results[
                selected_results[
                    "distance_metric"
                ]
                == distance_metric
            ]

            if not selected.empty:
                axis.scatter(
                    selected[
                        "requested_k"
                    ],
                    selected[
                        metric_column
                    ],
                    color=colours[
                        distance_metric
                    ],
                    edgecolor="black",
                    marker="*",
                    s=220,
                    zorder=5,
                )

        axis.set_xlabel("k")
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.legend()

    figure.suptitle(
        "Agglomerative Average-Linkage "
        "Sensitivity to Distance Metric",
        fontsize=15,
    )

    return save_figure(
        "distance_metric_search_comparison.png"
    )


def plot_metric_agreement(
    agreement_results: pd.DataFrame,
) -> Path:
    plt.figure(
        figsize=(9, 5)
    )

    sns.barplot(
        data=agreement_results,
        x="k",
        y="ari_between_metrics",
        color="#4c72b0",
    )

    plt.axhline(
        0.90,
        color="green",
        linestyle="--",
        label="Low sensitivity threshold",
    )

    plt.axhline(
        0.70,
        color="orange",
        linestyle=":",
        label=(
            "Moderate sensitivity threshold"
        ),
    )

    plt.ylim(
        -0.05,
        1.05,
    )

    plt.xlabel("k")

    plt.ylabel(
        "ARI between Euclidean and "
        "Manhattan partitions"
    )

    plt.title(
        "Partition Agreement Across "
        "Distance Metrics"
    )

    plt.legend(
        loc="lower right"
    )

    return save_figure(
        "distance_metric_partition_agreement.png"
    )


def plot_selected_partitions(
    x: np.ndarray,
    selected_results: pd.DataFrame,
    labels_by_metric: dict[
        str,
        dict[int, np.ndarray],
    ],
) -> Path:
    if x.shape[1] < 2:
        raise ValueError(
            "At least two PCA dimensions are required"
        )

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(15, 6),
        sharex=True,
        sharey=True,
    )

    for axis, distance_metric in zip(
        axes,
        DISTANCE_METRICS,
    ):
        selected = selected_results[
            selected_results[
                "distance_metric"
            ]
            == distance_metric
        ]

        if selected.empty:
            raise RuntimeError(
                "Selected result missing for "
                f"{distance_metric}"
            )

        selected_k = int(
            selected.iloc[0][
                "requested_k"
            ]
        )

        labels = labels_by_metric[
            distance_metric
        ][selected_k]

        scatter = axis.scatter(
            x[:, 0],
            x[:, 1],
            c=labels,
            cmap="tab20",
            s=14,
            alpha=0.65,
            rasterized=True,
        )

        axis.set_title(
            distance_metric.capitalize()
            + f" distance, k={selected_k}"
        )

        axis.set_xlabel("PC1")
        axis.set_ylabel("PC2")

        figure.colorbar(
            scatter,
            ax=axis,
            label="Cluster",
        )

    figure.suptitle(
        "Selected Partitions Under "
        "Alternative Distance Metrics"
    )

    return save_figure(
        "distance_metric_selected_partitions.png"
    )


def plot_cluster_size_comparison(
    assignments: pd.DataFrame,
) -> Path:
    rows = []

    for distance_metric in (
        DISTANCE_METRICS
    ):
        cluster_column = (
            f"{distance_metric}_cluster"
        )

        counts = (
            assignments[
                cluster_column
            ]
            .value_counts()
            .sort_index()
        )

        for cluster_id, count in (
            counts.items()
        ):
            rows.append(
                {
                    "distance_metric": (
                        distance_metric
                    ),
                    "cluster": str(
                        cluster_id
                    ),
                    "size": int(
                        count
                    ),
                    "fraction": float(
                        count
                        / len(assignments)
                    ),
                }
            )

    size_table = pd.DataFrame(
        rows
    )

    size_table.to_csv(
        REPORT_DIR
        / "distance_metric_cluster_sizes.csv",
        index=False,
    )

    plt.figure(
        figsize=(11, 6)
    )

    sns.barplot(
        data=size_table,
        x="cluster",
        y="fraction",
        hue="distance_metric",
    )

    plt.xlabel("Cluster identifier")
    plt.ylabel("Sample fraction")

    plt.title(
        "Cluster-Size Profiles Across "
        "Distance Metrics"
    )

    return save_figure(
        "distance_metric_cluster_sizes.png"
    )


def write_report(
    selected_results: pd.DataFrame,
    same_k_agreement: pd.DataFrame,
    selected_agreement: dict[str, Any],
    sample_size: int,
) -> None:
    euclidean_row = selected_results[
        selected_results[
            "distance_metric"
        ]
        == "euclidean"
    ].iloc[0]

    manhattan_row = selected_results[
        selected_results[
            "distance_metric"
        ]
        == "manhattan"
    ].iloc[0]

    same_k_minimum = float(
        same_k_agreement[
            "ari_between_metrics"
        ].min()
    )

    same_k_maximum = float(
        same_k_agreement[
            "ari_between_metrics"
        ].max()
    )

    lines = [
        "# Distance-Metric Sensitivity Analysis",
        "",
        "Agglomerative clustering with average linkage was "
        "held fixed while only the distance metric was changed.",
        "",
        "The experiment used the same records, PCA representation, "
        "candidate k values, linkage criterion, and random sample for "
        "both metrics.",
        "",
        f"Analysis sample size: {sample_size}",
        "",
        "Euclidean distance is natural for standardised continuous "
        "features and emphasises larger coordinate differences.",
        "",
        "Manhattan distance is less dominated by a small number of "
        "large coordinate deviations and can be more robust to "
        "heavy-tailed observations.",
        "",
        "The fraud label was not used to select the distance metric "
        "or k. It was consulted only for post-hoc external metrics.",
        "",
        "## Euclidean Selection",
        "",
        f"- Selected k: {int(euclidean_row['requested_k'])}",
        f"- Matching-metric silhouette: "
        f"{euclidean_row['silhouette_matching_metric']:.6f}",
        f"- Davies-Bouldin: "
        f"{euclidean_row['davies_bouldin']:.6f}",
        f"- Calinski-Harabasz: "
        f"{euclidean_row['calinski_harabasz']:.6f}",
        f"- External ARI: "
        f"{euclidean_row['ari_external']:.6f}",
        "",
        "## Manhattan Selection",
        "",
        f"- Selected k: {int(manhattan_row['requested_k'])}",
        f"- Matching-metric silhouette: "
        f"{manhattan_row['silhouette_matching_metric']:.6f}",
        f"- Davies-Bouldin: "
        f"{manhattan_row['davies_bouldin']:.6f}",
        f"- Calinski-Harabasz: "
        f"{manhattan_row['calinski_harabasz']:.6f}",
        f"- External ARI: "
        f"{manhattan_row['ari_external']:.6f}",
        "",
        "## Partition Agreement",
        "",
        f"- ARI between independently selected partitions: "
        f"{selected_agreement['ari_between_selected_partitions']:.6f}",
        f"- Sensitivity level: "
        f"{selected_agreement['distance_sensitivity_level']}",
        f"- Minimum same-k ARI: {same_k_minimum:.6f}",
        f"- Maximum same-k ARI: {same_k_maximum:.6f}",
        "",
        "## Interpretation",
        "",
    ]

    sensitivity_level = (
        selected_agreement[
            "distance_sensitivity_level"
        ]
    )

    if sensitivity_level == "low":
        lines.extend(
            [
                "The two metrics recover highly similar partitions. "
                "The cluster structure is therefore relatively robust "
                "to replacing Euclidean distance with Manhattan "
                "distance.",
            ]
        )
    elif sensitivity_level == "moderate":
        lines.extend(
            [
                "The two metrics recover related but not identical "
                "partitions. Cluster interpretation should acknowledge "
                "moderate dependence on the geometry imposed by the "
                "distance metric.",
            ]
        )
    else:
        lines.extend(
            [
                "The partitions disagree substantially. The discovered "
                "cluster structure is highly sensitive to the distance "
                "metric and should not be presented as a unique "
                "intrinsic partition of the transactions.",
            ]
        )

    lines.extend(
        [
            "",
            "Davies-Bouldin and Calinski-Harabasz are based on "
            "Euclidean geometry in their standard scikit-learn "
            "implementations. They are retained as common-space "
            "comparators, while matching-metric silhouette is the "
            "primary metric-specific criterion.",
            "",
            "External fraud metrics are descriptive only. They did "
            "not participate in metric or hyperparameter selection.",
        ]
    )

    (
        REPORT_DIR
        / "distance_metric_sensitivity_analysis.md"
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

    silhouette_sample_size = int(
        params["sampling"][
            "silhouette_sample_size"
        ]
    )

    (
        x,
        y_true,
        selected_indices,
    ) = load_analysis_data(
        params
    )

    (
        search_results,
        labels_by_metric,
    ) = run_metric_search(
        x=x,
        y_true=y_true,
        k_values=k_values,
        random_seed=random_seed,
        silhouette_sample_size=(
            silhouette_sample_size
        ),
    )

    ranked_results = (
        rank_metric_candidates(
            search_results
        )
    )

    selected_results = (
        select_best_for_each_metric(
            ranked_results
        )
    )

    same_k_agreement = (
        calculate_metric_agreement(
            labels_by_metric=(
                labels_by_metric
            ),
            k_values=k_values,
        )
    )

    selected_agreement = (
        calculate_selected_partition_agreement(
            selected_results=(
                selected_results
            ),
            labels_by_metric=(
                labels_by_metric
            ),
        )
    )

    assignments = save_selected_assignments(
        selected_indices=(
            selected_indices
        ),
        y_true=y_true,
        selected_results=(
            selected_results
        ),
        labels_by_metric=(
            labels_by_metric
        ),
    )

    ranked_results.to_csv(
        REPORT_DIR
        / "distance_metric_search.csv",
        index=False,
    )

    selected_results.to_csv(
        REPORT_DIR
        / "distance_metric_selected_results.csv",
        index=False,
    )

    same_k_agreement.to_csv(
        REPORT_DIR
        / "distance_metric_same_k_agreement.csv",
        index=False,
    )

    save_json(
        selected_agreement,
        REPORT_DIR
        / "distance_metric_selected_agreement.json",
    )

    metric_figure = plot_metric_search(
        search_results=(
            ranked_results
        ),
        selected_results=(
            selected_results
        ),
    )

    agreement_figure = (
        plot_metric_agreement(
            same_k_agreement
        )
    )

    partition_figure = (
        plot_selected_partitions(
            x=x,
            selected_results=(
                selected_results
            ),
            labels_by_metric=(
                labels_by_metric
            ),
        )
    )

    cluster_size_figure = (
        plot_cluster_size_comparison(
            assignments
        )
    )

    write_report(
        selected_results=(
            selected_results
        ),
        same_k_agreement=(
            same_k_agreement
        ),
        selected_agreement=(
            selected_agreement
        ),
        sample_size=len(x),
    )

    completion_record = {
        "status": "completed",
        "algorithm": (
            "Agglomerative-average"
        ),
        "distance_metrics": (
            DISTANCE_METRICS
        ),
        "sample_size": int(
            len(x)
        ),
        "feature_count": int(
            x.shape[1]
        ),
        "candidate_k_values": (
            k_values
        ),
        **selected_agreement,
        "generated_figures": [
            str(
                path.relative_to(ROOT)
            )
            for path in [
                metric_figure,
                agreement_figure,
                partition_figure,
                cluster_size_figure,
            ]
        ],
    }

    save_json(
        completion_record,
        REPORT_DIR
        / "phase2_completion_step4_record.json",
    )

    logging.info(
        "Phase 2 distance-metric "
        "sensitivity analysis completed"
    )


if __name__ == "__main__":
    main()