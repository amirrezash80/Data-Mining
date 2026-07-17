import json
import logging
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from sklearn.metrics import silhouette_samples


ROOT = Path(__file__).resolve().parents[1]
PARAMS_PATH = ROOT / "params_phase2.yaml"

ARRAY_PATH = (
    ROOT
    / "data"
    / "processed"
    / "phase1_arrays.npz"
)

ASSIGNMENT_PATH = (
    ROOT
    / "data"
    / "processed"
    / "phase2_cluster_assignments.parquet"
)

METADATA_PATH = (
    ROOT
    / "data"
    / "processed"
    / "metadata_and_labels.parquet"
)

FINAL_COMPARISON_PATH = (
    ROOT
    / "reports"
    / "phase2"
    / "final_comparison.csv"
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


ALGORITHM_COLUMNS = {
    "KMeans": "KMeans_cluster",
    "Hierarchical": "Hierarchical_cluster",
    "DBSCAN": "DBSCAN_cluster",
    "GMM": "GMM_cluster",
}


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
                / "phase2_completion_step1.log",
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


def normalise_algorithm_name(
    value: str,
) -> str:
    cleaned = str(value).strip().lower()

    if "kmeans" in cleaned or "k-means" in cleaned:
        return "KMeans"

    if "hierarchical" in cleaned or "agglomerative" in cleaned:
        return "Hierarchical"

    if "dbscan" in cleaned:
        return "DBSCAN"

    if "gmm" in cleaned or "gaussian" in cleaned:
        return "GMM"

    raise ValueError(
        f"Unsupported algorithm name: {value}"
    )


def load_inputs() -> tuple[
    np.ndarray,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    required_paths = [
        ARRAY_PATH,
        ASSIGNMENT_PATH,
        FINAL_COMPARISON_PATH,
    ]

    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(
                f"Required Phase 2 output not found: {path}"
            )

    with np.load(
        ARRAY_PATH,
        allow_pickle=False,
    ) as arrays:
        if "X_pca" not in arrays.files:
            raise KeyError(
                "X_pca is missing from phase1_arrays.npz"
            )

        x_pca_all = arrays[
            "X_pca"
        ].astype(np.float64)

    assignments = pd.read_parquet(
        ASSIGNMENT_PATH
    )

    final_comparison = pd.read_csv(
        FINAL_COMPARISON_PATH
    )

    if METADATA_PATH.exists():
        metadata = pd.read_parquet(
            METADATA_PATH
        )
    else:
        metadata = pd.DataFrame()

    required_assignment_columns = {
        "array_index",
        *ALGORITHM_COLUMNS.values(),
    }

    missing_columns = (
        required_assignment_columns
        - set(assignments.columns)
    )

    if missing_columns:
        raise KeyError(
            "Missing assignment columns: "
            f"{sorted(missing_columns)}"
        )

    if assignments[
        "array_index"
    ].duplicated().any():
        raise ValueError(
            "Duplicate array_index values found"
        )

    array_indices = assignments[
        "array_index"
    ].to_numpy(dtype=np.int64)

    if np.any(array_indices < 0):
        raise IndexError(
            "Negative array indices found"
        )

    if np.any(
        array_indices >= len(x_pca_all)
    ):
        raise IndexError(
            "An assignment index exceeds X_pca"
        )

    x = x_pca_all[array_indices]

    if len(x) != len(assignments):
        raise ValueError(
            "Feature and assignment row counts differ"
        )

    return (
        x,
        assignments.reset_index(drop=True),
        final_comparison,
        metadata,
    )


def calculate_silhouette_values(
    x: np.ndarray,
    labels: np.ndarray,
    exclude_noise: bool,
) -> tuple[
    np.ndarray,
    np.ndarray,
]:
    labels = np.asarray(labels)

    if exclude_noise:
        valid_mask = labels != -1
    else:
        valid_mask = np.ones(
            len(labels),
            dtype=bool,
        )

    valid_x = x[valid_mask]
    valid_labels = labels[valid_mask]

    unique_labels = np.unique(
        valid_labels
    )

    if len(valid_x) < 3:
        raise ValueError(
            "Not enough records for silhouette analysis"
        )

    if len(unique_labels) < 2:
        raise ValueError(
            "At least two clusters are required"
        )

    if len(unique_labels) >= len(valid_x):
        raise ValueError(
            "Every valid record has a unique cluster"
        )

    values = silhouette_samples(
        valid_x,
        valid_labels,
        metric="euclidean",
    )

    return values, valid_mask


def create_silhouette_plot(
    algorithm: str,
    x: np.ndarray,
    labels: np.ndarray,
) -> tuple[
    Path,
    pd.DataFrame,
]:
    exclude_noise = algorithm == "DBSCAN"

    values, valid_mask = (
        calculate_silhouette_values(
            x=x,
            labels=labels,
            exclude_noise=exclude_noise,
        )
    )

    valid_labels = labels[valid_mask]
    unique_labels = np.unique(
        valid_labels
    )

    average_silhouette = float(
        values.mean()
    )

    figure, axis = plt.subplots(
        figsize=(10, 7)
    )

    lower_position = 10
    colour_map = plt.get_cmap(
        "tab20"
    )

    summary_rows = []

    for position, cluster_id in enumerate(
        unique_labels
    ):
        cluster_values = np.sort(
            values[
                valid_labels == cluster_id
            ]
        )

        cluster_size = len(
            cluster_values
        )

        upper_position = (
            lower_position
            + cluster_size
        )

        colour = colour_map(
            position
            / max(
                1,
                len(unique_labels) - 1,
            )
        )

        axis.fill_betweenx(
            np.arange(
                lower_position,
                upper_position,
            ),
            0,
            cluster_values,
            facecolor=colour,
            edgecolor=colour,
            alpha=0.75,
        )

        axis.text(
            -0.05,
            lower_position
            + 0.5 * cluster_size,
            str(cluster_id),
        )

        summary_rows.append(
            {
                "algorithm": algorithm,
                "cluster": int(cluster_id),
                "cluster_size": int(
                    cluster_size
                ),
                "silhouette_mean": float(
                    cluster_values.mean()
                ),
                "silhouette_median": float(
                    np.median(
                        cluster_values
                    )
                ),
                "silhouette_minimum": float(
                    cluster_values.min()
                ),
                "silhouette_maximum": float(
                    cluster_values.max()
                ),
                "negative_fraction": float(
                    np.mean(
                        cluster_values < 0
                    )
                ),
                "boundary_fraction": float(
                    np.mean(
                        np.abs(
                            cluster_values
                        ) <= 0.10
                    )
                ),
            }
        )

        lower_position = (
            upper_position + 10
        )

    axis.axvline(
        average_silhouette,
        color="red",
        linestyle="--",
        linewidth=2,
        label=(
            "Mean silhouette = "
            f"{average_silhouette:.4f}"
        ),
    )

    axis.axvline(
        0,
        color="black",
        linestyle=":",
        linewidth=1,
    )

    axis.set_xlim(
        -1.0,
        1.0,
    )

    axis.set_ylim(
        0,
        lower_position,
    )

    axis.set_xlabel(
        "Silhouette coefficient"
    )

    axis.set_ylabel(
        "Records grouped by cluster"
    )

    axis.set_yticks([])

    axis.set_title(
        f"Per-point Silhouette Plot: {algorithm}"
    )

    axis.legend(
        loc="lower right"
    )

    filename = (
        "silhouette_plot_"
        + algorithm.lower()
        + ".png"
    )

    path = save_figure(
        filename
    )

    summary = pd.DataFrame(
        summary_rows
    )

    summary[
        "overall_mean_silhouette"
    ] = average_silhouette

    summary[
        "noise_fraction"
    ] = float(
        np.mean(labels == -1)
    )

    return path, summary


def create_all_silhouette_outputs(
    x: np.ndarray,
    assignments: pd.DataFrame,
) -> tuple[
    dict[str, np.ndarray],
    pd.DataFrame,
    list[Path],
]:
    silhouette_values: dict[
        str,
        np.ndarray,
    ] = {}

    summary_frames = []
    figure_paths = []

    for algorithm, column in (
        ALGORITHM_COLUMNS.items()
    ):
        labels = assignments[
            column
        ].to_numpy(dtype=np.int32)

        try:
            path, summary = (
                create_silhouette_plot(
                    algorithm=algorithm,
                    x=x,
                    labels=labels,
                )
            )

            values, valid_mask = (
                calculate_silhouette_values(
                    x=x,
                    labels=labels,
                    exclude_noise=(
                        algorithm == "DBSCAN"
                    ),
                )
            )

            full_values = np.full(
                len(labels),
                np.nan,
                dtype=float,
            )

            full_values[
                valid_mask
            ] = values

            silhouette_values[
                algorithm
            ] = full_values

            summary_frames.append(
                summary
            )

            figure_paths.append(
                path
            )

        except ValueError as error:
            logging.warning(
                "Silhouette skipped for %s: %s",
                algorithm,
                error,
            )

    if not summary_frames:
        raise RuntimeError(
            "No silhouette outputs were generated"
        )

    combined_summary = pd.concat(
        summary_frames,
        ignore_index=True,
    )

    combined_summary.to_csv(
        REPORT_DIR
        / "silhouette_cluster_summary.csv",
        index=False,
    )

    return (
        silhouette_values,
        combined_summary,
        figure_paths,
    )


def choose_preferred_algorithm(
    final_comparison: pd.DataFrame,
) -> tuple[
    str,
    pd.Series,
]:
    required_columns = {
        "algorithm",
        "silhouette",
        "davies_bouldin",
        "calinski_harabasz",
    }

    missing = (
        required_columns
        - set(final_comparison.columns)
    )

    if missing:
        raise KeyError(
            "Missing final comparison columns: "
            f"{sorted(missing)}"
        )

    candidates = (
        final_comparison
        .dropna(
            subset=[
                "silhouette",
                "davies_bouldin",
                "calinski_harabasz",
            ]
        )
        .copy()
    )

    if candidates.empty:
        raise ValueError(
            "No valid preferred algorithm candidate"
        )

    candidates["rank_silhouette"] = (
        candidates[
            "silhouette"
        ].rank(
            ascending=False,
            method="min",
        )
    )

    candidates["rank_db"] = (
        candidates[
            "davies_bouldin"
        ].rank(
            ascending=True,
            method="min",
        )
    )

    candidates["rank_ch"] = (
        candidates[
            "calinski_harabasz"
        ].rank(
            ascending=False,
            method="min",
        )
    )

    candidates[
        "internal_rank_sum"
    ] = (
        candidates[
            "rank_silhouette"
        ]
        + candidates["rank_db"]
        + candidates["rank_ch"]
    )

    selected_row = candidates.sort_values(
        [
            "internal_rank_sum",
            "silhouette",
        ],
        ascending=[
            True,
            False,
        ],
    ).iloc[0]

    algorithm = normalise_algorithm_name(
        selected_row["algorithm"]
    )

    return algorithm, selected_row


def add_metadata(
    analysis: pd.DataFrame,
    metadata: pd.DataFrame,
) -> pd.DataFrame:
    if metadata.empty:
        return analysis

    indices = analysis[
        "array_index"
    ].to_numpy(dtype=np.int64)

    if np.any(indices >= len(metadata)):
        logging.warning(
            "Metadata merge skipped because "
            "indices exceed metadata length"
        )

        return analysis

    selected_metadata = (
        metadata.iloc[indices]
        .reset_index(drop=True)
        .copy()
    )

    duplicate_columns = [
        column
        for column in selected_metadata.columns
        if column in analysis.columns
    ]

    selected_metadata = (
        selected_metadata.drop(
            columns=duplicate_columns,
            errors="ignore",
        )
    )

    return pd.concat(
        [
            analysis.reset_index(
                drop=True
            ),
            selected_metadata,
        ],
        axis=1,
    )


def calculate_centroid_diagnostics(
    x: np.ndarray,
    labels: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    cluster_ids = np.unique(labels)

    centroids = np.vstack(
        [
            x[
                labels == cluster_id
            ].mean(axis=0)
            for cluster_id in cluster_ids
        ]
    )

    distances = np.linalg.norm(
        x[:, None, :]
        - centroids[None, :, :],
        axis=2,
    )

    assigned_positions = np.asarray(
        [
            int(
                np.where(
                    cluster_ids
                    == cluster_id
                )[0][0]
            )
            for cluster_id in labels
        ],
        dtype=np.int64,
    )

    row_positions = np.arange(
        len(x)
    )

    assigned_distance = distances[
        row_positions,
        assigned_positions,
    ]

    masked_distances = distances.copy()

    masked_distances[
        row_positions,
        assigned_positions,
    ] = np.inf

    nearest_other_position = (
        np.argmin(
            masked_distances,
            axis=1,
        )
    )

    nearest_other_cluster = cluster_ids[
        nearest_other_position
    ]

    nearest_other_distance = (
        masked_distances[
            row_positions,
            nearest_other_position,
        ]
    )

    return (
        assigned_distance,
        nearest_other_cluster,
        nearest_other_distance,
    )


def create_error_analysis(
    preferred_algorithm: str,
    x: np.ndarray,
    assignments: pd.DataFrame,
    metadata: pd.DataFrame,
    silhouette_values: np.ndarray,
    lowest_count: int = 100,
) -> tuple[
    pd.DataFrame,
    dict[str, Any],
]:
    label_column = ALGORITHM_COLUMNS[
        preferred_algorithm
    ]

    labels = assignments[
        label_column
    ].to_numpy(dtype=np.int32)

    valid_mask = ~np.isnan(
        silhouette_values
    )

    if not np.any(valid_mask):
        raise ValueError(
            "No valid silhouette values "
            "for preferred algorithm"
        )

    valid_positions = np.flatnonzero(
        valid_mask
    )

    sorted_positions = valid_positions[
        np.argsort(
            silhouette_values[
                valid_positions
            ]
        )
    ]

    selected_positions = sorted_positions[
        :min(
            lowest_count,
            len(sorted_positions),
        )
    ]

    (
        assigned_distance,
        nearest_other_cluster,
        nearest_other_distance,
    ) = calculate_centroid_diagnostics(
        x=x,
        labels=labels,
    )

    analysis = pd.DataFrame(
        {
            "sample_position": (
                selected_positions
            ),
            "array_index": assignments.iloc[
                selected_positions
            ]["array_index"].to_numpy(
                dtype=np.int64
            ),
            "row_id": (
                assignments.iloc[
                    selected_positions
                ]["row_id"].to_numpy(
                    dtype=np.int64
                )
                if "row_id"
                in assignments.columns
                else selected_positions
            ),
            "preferred_algorithm": (
                preferred_algorithm
            ),
            "assigned_cluster": labels[
                selected_positions
            ],
            "silhouette": (
                silhouette_values[
                    selected_positions
                ]
            ),
            "distance_to_assigned_centroid": (
                assigned_distance[
                    selected_positions
                ]
            ),
            "nearest_other_cluster": (
                nearest_other_cluster[
                    selected_positions
                ]
            ),
            "distance_to_nearest_other_centroid": (
                nearest_other_distance[
                    selected_positions
                ]
            ),
        }
    )

    analysis[
        "centroid_distance_margin"
    ] = (
        analysis[
            "distance_to_nearest_other_centroid"
        ]
        - analysis[
            "distance_to_assigned_centroid"
        ]
    )

    analysis[
        "diagnostic_category"
    ] = np.select(
        [
            analysis["silhouette"] < 0,
            analysis["silhouette"] <= 0.10,
        ],
        [
            "probable_misassignment",
            "boundary_case",
        ],
        default="weakly_separated",
    )

    analysis = add_metadata(
        analysis=analysis,
        metadata=metadata,
    )

    class_column = None

    for candidate in [
        "Class_external_only",
        "Class",
    ]:
        if candidate in analysis.columns:
            class_column = candidate
            break

    summary: dict[str, Any] = {
        "preferred_algorithm": (
            preferred_algorithm
        ),
        "analysed_record_count": int(
            len(analysis)
        ),
        "minimum_silhouette": float(
            analysis["silhouette"].min()
        ),
        "mean_silhouette": float(
            analysis["silhouette"].mean()
        ),
        "negative_silhouette_count": int(
            np.sum(
                analysis["silhouette"] < 0
            )
        ),
        "negative_silhouette_fraction": float(
            np.mean(
                analysis["silhouette"] < 0
            )
        ),
        "boundary_count": int(
            np.sum(
                np.abs(
                    analysis["silhouette"]
                ) <= 0.10
            )
        ),
        "boundary_fraction": float(
            np.mean(
                np.abs(
                    analysis["silhouette"]
                ) <= 0.10
            )
        ),
        "diagnostic_category_counts": (
            analysis[
                "diagnostic_category"
            ]
            .value_counts()
            .to_dict()
        ),
    }

    if class_column is not None:
        summary[
            "fraud_rate_lowest_silhouette"
        ] = float(
            analysis[
                class_column
            ].mean()
        )

    if "Amount" in analysis.columns:
        summary[
            "amount_median_lowest_silhouette"
        ] = float(
            analysis[
                "Amount"
            ].median()
        )

        summary[
            "amount_mean_lowest_silhouette"
        ] = float(
            analysis[
                "Amount"
            ].mean()
        )

    analysis.to_csv(
        REPORT_DIR
        / "lowest_silhouette_records.csv",
        index=False,
    )

    save_json(
        summary,
        REPORT_DIR
        / "preferred_model_error_summary.json",
    )

    return analysis, summary


def plot_error_analysis(
    analysis: pd.DataFrame,
    preferred_algorithm: str,
) -> Path:
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(14, 5),
    )

    sns.histplot(
        data=analysis,
        x="silhouette",
        hue="diagnostic_category",
        bins=30,
        multiple="stack",
        ax=axes[0],
    )

    axes[0].axvline(
        0,
        color="black",
        linestyle=":",
    )

    axes[0].set_title(
        "Lowest Silhouette Distribution"
    )

    axes[0].set_xlabel(
        "Silhouette coefficient"
    )

    sns.scatterplot(
        data=analysis,
        x="silhouette",
        y="centroid_distance_margin",
        hue="diagnostic_category",
        style="assigned_cluster",
        s=70,
        ax=axes[1],
    )

    axes[1].axvline(
        0,
        color="black",
        linestyle=":",
    )

    axes[1].axhline(
        0,
        color="black",
        linestyle=":",
    )

    axes[1].set_title(
        "Boundary and Misassignment Diagnostics"
    )

    axes[1].set_xlabel(
        "Silhouette coefficient"
    )

    axes[1].set_ylabel(
        "Nearest-other minus assigned "
        "centroid distance"
    )

    figure.suptitle(
        f"Preferred Model Error Analysis: "
        f"{preferred_algorithm}"
    )

    return save_figure(
        "preferred_model_error_analysis.png"
    )


def write_error_analysis_report(
    summary: dict[str, Any],
    selected_row: pd.Series,
) -> None:
    parameters = selected_row.get(
        "parameters",
        "not available",
    )

    report = (
        "# Phase 2 Preferred-Clustering Error Analysis\n\n"
        f"Preferred algorithm: "
        f"{summary['preferred_algorithm']}\n\n"
        f"Selected parameters: {parameters}\n\n"
        "The preferred algorithm was selected using "
        "internal metrics only. Fraud labels were not "
        "used for algorithm selection.\n\n"
        "Records with the lowest silhouette coefficients "
        "were inspected as potential boundary cases, "
        "probable misassignments, or weakly separated "
        "observations.\n\n"
        f"Analysed records: "
        f"{summary['analysed_record_count']}\n\n"
        f"Minimum silhouette: "
        f"{summary['minimum_silhouette']:.6f}\n\n"
        f"Mean silhouette among inspected records: "
        f"{summary['mean_silhouette']:.6f}\n\n"
        f"Negative-silhouette fraction: "
        f"{summary['negative_silhouette_fraction']:.6f}\n\n"
        f"Boundary fraction: "
        f"{summary['boundary_fraction']:.6f}\n\n"
        "Diagnostic categories:\n\n"
        f"{json.dumps(summary['diagnostic_category_counts'], indent=2)}\n\n"
        "Interpretation guidance:\n\n"
        "- Negative silhouette indicates that a record "
        "is, on average, closer to another cluster than "
        "to its assigned cluster.\n"
        "- Values near zero indicate boundary cases.\n"
        "- A small centroid-distance margin indicates "
        "ambiguity between the assigned and nearest "
        "alternative cluster.\n"
        "- Fraud composition is evaluated only after "
        "the unsupervised model has been selected.\n"
    )

    if (
        "fraud_rate_lowest_silhouette"
        in summary
    ):
        report += (
            "\nPost-hoc fraud rate among inspected "
            "records: "
            f"{summary['fraud_rate_lowest_silhouette']:.6f}\n"
        )

    (
        REPORT_DIR
        / "preferred_model_error_analysis.md"
    ).write_text(
        report,
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

    (
        x,
        assignments,
        final_comparison,
        metadata,
    ) = load_inputs()

    (
        silhouette_values,
        silhouette_summary,
        silhouette_figures,
    ) = create_all_silhouette_outputs(
        x=x,
        assignments=assignments,
    )

    (
        preferred_algorithm,
        selected_row,
    ) = choose_preferred_algorithm(
        final_comparison
    )

    if (
        preferred_algorithm
        not in silhouette_values
    ):
        raise RuntimeError(
            "Preferred algorithm has no "
            "valid silhouette values"
        )

    (
        error_analysis,
        error_summary,
    ) = create_error_analysis(
        preferred_algorithm=(
            preferred_algorithm
        ),
        x=x,
        assignments=assignments,
        metadata=metadata,
        silhouette_values=(
            silhouette_values[
                preferred_algorithm
            ]
        ),
        lowest_count=100,
    )

    error_figure = plot_error_analysis(
        analysis=error_analysis,
        preferred_algorithm=(
            preferred_algorithm
        ),
    )

    write_error_analysis_report(
        summary=error_summary,
        selected_row=selected_row,
    )

    completion_record = {
        "status": "completed",
        "preferred_algorithm": (
            preferred_algorithm
        ),
        "preferred_parameters": (
            selected_row.get(
                "parameters",
                None,
            )
        ),
        "silhouette_algorithms": (
            list(
                silhouette_values.keys()
            )
        ),
        "silhouette_summary_rows": int(
            len(silhouette_summary)
        ),
        "lowest_silhouette_record_count": int(
            len(error_analysis)
        ),
        "generated_figures": [
            str(
                path.relative_to(ROOT)
            )
            for path in [
                *silhouette_figures,
                error_figure,
            ]
        ],
    }

    save_json(
        completion_record,
        REPORT_DIR
        / "phase2_completion_step1_record.json",
    )

    logging.info(
        "Phase 2 completion step 1 succeeded"
    )


if __name__ == "__main__":
    main()