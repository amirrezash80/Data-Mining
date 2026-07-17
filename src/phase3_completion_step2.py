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
from scipy.spatial.distance import cdist
from sklearn.metrics import silhouette_samples


ROOT = Path(__file__).resolve().parents[1]
PARAMS_PATH = ROOT / "params_phase3.yaml"

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
    / "phase3_consensus_assignments.parquet"
)

FEATURE_PATH = (
    ROOT
    / "data"
    / "processed"
    / "features_unscaled.parquet"
)

METADATA_PATH = (
    ROOT
    / "data"
    / "processed"
    / "metadata_and_labels.parquet"
)

DOMAIN_LABEL_PATH = (
    ROOT
    / "reports"
    / "phase3"
    / "cluster_domain_labels.csv"
)

REPORT_DIR = (
    ROOT
    / "reports"
    / "phase3"
)

FIGURE_DIR = (
    ROOT
    / "reports"
    / "figures"
    / "phase3"
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
                / "phase3_completion_step2.log",
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
            f"Invalid configuration file: {path}"
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


def load_inputs() -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    required_paths = [
        ARRAY_PATH,
        ASSIGNMENT_PATH,
        FEATURE_PATH,
        METADATA_PATH,
    ]

    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(
                f"Required input was not found: {path}"
            )

    with np.load(
        ARRAY_PATH,
        allow_pickle=False,
    ) as arrays:
        required_arrays = {
            "X_pca",
            "row_id",
            "y",
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

        x_pca_all = arrays[
            "X_pca"
        ].astype(np.float64)

        row_id_all = arrays[
            "row_id"
        ].astype(np.int64)

        y_all = arrays[
            "y"
        ].astype(np.int8)

    assignments = pd.read_parquet(
        ASSIGNMENT_PATH
    )

    features_all = pd.read_parquet(
        FEATURE_PATH
    )

    metadata_all = pd.read_parquet(
        METADATA_PATH
    )

    required_assignment_columns = {
        "array_index",
        "Consensus_cluster",
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
            "Duplicate array_index values "
            "exist in consensus assignments"
        )

    array_indices = assignments[
        "array_index"
    ].to_numpy(dtype=np.int64)

    if np.any(array_indices < 0):
        raise IndexError(
            "Negative array indices were found"
        )

    if np.any(
        array_indices >= len(x_pca_all)
    ):
        raise IndexError(
            "A consensus assignment index "
            "exceeds the Phase 1 arrays"
        )

    expected_rows = len(x_pca_all)

    if len(features_all) != expected_rows:
        raise ValueError(
            "Feature table and Phase 1 arrays "
            "have different row counts"
        )

    if len(metadata_all) != expected_rows:
        raise ValueError(
            "Metadata table and Phase 1 arrays "
            "have different row counts"
        )

    x = x_pca_all[
        array_indices
    ]

    labels = assignments[
        "Consensus_cluster"
    ].to_numpy(dtype=np.int32)

    row_ids = row_id_all[
        array_indices
    ]

    y_true = y_all[
        array_indices
    ]

    features = (
        features_all.iloc[
            array_indices
        ]
        .reset_index(drop=True)
        .copy()
    )

    metadata = (
        metadata_all.iloc[
            array_indices
        ]
        .reset_index(drop=True)
        .copy()
    )

    assignments = (
        assignments
        .reset_index(drop=True)
        .copy()
    )

    if not np.isfinite(x).all():
        raise ValueError(
            "PCA matrix contains NaN or "
            "infinite values"
        )

    if len(np.unique(labels)) < 2:
        raise ValueError(
            "At least two consensus clusters "
            "are required"
        )

    return (
        x,
        labels,
        array_indices,
        row_ids,
        features,
        metadata,
        assignments,
    )


def calculate_exact_medoid(
    cluster_x: np.ndarray,
    chunk_size: int,
) -> tuple[
    int,
    np.ndarray,
]:
    cluster_size = len(cluster_x)

    if cluster_size == 0:
        raise ValueError(
            "Cannot calculate a medoid "
            "for an empty cluster"
        )

    if cluster_size == 1:
        return (
            0,
            np.zeros(
                1,
                dtype=np.float64,
            ),
        )

    distance_sums = np.zeros(
        cluster_size,
        dtype=np.float64,
    )

    for start in range(
        0,
        cluster_size,
        chunk_size,
    ):
        end = min(
            start + chunk_size,
            cluster_size,
        )

        distances = cdist(
            cluster_x[
                start:end
            ],
            cluster_x,
            metric="euclidean",
        )

        distance_sums[
            start:end
        ] = distances.sum(
            axis=1
        )

    medoid_local_index = int(
        np.argmin(
            distance_sums
        )
    )

    return (
        medoid_local_index,
        distance_sums,
    )


def calculate_silhouette(
    x: np.ndarray,
    labels: np.ndarray,
) -> np.ndarray:
    unique_labels = np.unique(
        labels
    )

    if len(unique_labels) < 2:
        return np.full(
            len(labels),
            np.nan,
            dtype=np.float64,
        )

    if len(unique_labels) >= len(labels):
        return np.full(
            len(labels),
            np.nan,
            dtype=np.float64,
        )

    return silhouette_samples(
        x,
        labels,
        metric="euclidean",
    )


def select_cluster_prototypes(
    x: np.ndarray,
    labels: np.ndarray,
    array_indices: np.ndarray,
    row_ids: np.ndarray,
    y_true: np.ndarray,
    chunk_size: int,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    silhouette_values = (
        calculate_silhouette(
            x,
            labels,
        )
    )

    prototype_rows = []
    diagnostic_rows = []

    for cluster_id in np.unique(
        labels
    ):
        cluster_positions = np.flatnonzero(
            labels == cluster_id
        )

        cluster_x = x[
            cluster_positions
        ]

        cluster_silhouettes = (
            silhouette_values[
                cluster_positions
            ]
        )

        started = time.perf_counter()

        (
            medoid_local_index,
            distance_sums,
        ) = calculate_exact_medoid(
            cluster_x=cluster_x,
            chunk_size=chunk_size,
        )

        medoid_runtime = (
            time.perf_counter()
            - started
        )

        centroid = cluster_x.mean(
            axis=0
        )

        centroid_distances = (
            np.linalg.norm(
                cluster_x - centroid,
                axis=1,
            )
        )

        centroid_exemplar_local = int(
            np.argmin(
                centroid_distances
            )
        )

        if np.all(
            np.isnan(
                cluster_silhouettes
            )
        ):
            highest_silhouette_local = (
                medoid_local_index
            )

            boundary_local = int(
                np.argmax(
                    centroid_distances
                )
            )
        else:
            highest_silhouette_local = int(
                np.nanargmax(
                    cluster_silhouettes
                )
            )

            boundary_local = int(
                np.nanargmin(
                    cluster_silhouettes
                )
            )

        selected_prototypes = {
            "medoid": medoid_local_index,
            "highest_silhouette": (
                highest_silhouette_local
            ),
            "boundary_point": (
                boundary_local
            ),
            "centroid_exemplar": (
                centroid_exemplar_local
            ),
        }

        medoid_global_position = int(
            cluster_positions[
                medoid_local_index
            ]
        )

        centroid_global_position = int(
            cluster_positions[
                centroid_exemplar_local
            ]
        )

        medoid_centroid_same_record = bool(
            medoid_global_position
            == centroid_global_position
        )

        for (
            prototype_type,
            local_index,
        ) in selected_prototypes.items():
            sample_position = int(
                cluster_positions[
                    local_index
                ]
            )

            prototype_rows.append(
                {
                    "cluster": int(
                        cluster_id
                    ),
                    "cluster_size": int(
                        len(
                            cluster_positions
                        )
                    ),
                    "prototype_type": (
                        prototype_type
                    ),
                    "sample_position": (
                        sample_position
                    ),
                    "array_index": int(
                        array_indices[
                            sample_position
                        ]
                    ),
                    "row_id": int(
                        row_ids[
                            sample_position
                        ]
                    ),
                    "Class_external_only": int(
                        y_true[
                            sample_position
                        ]
                    ),
                    "silhouette": float(
                        silhouette_values[
                            sample_position
                        ]
                    ),
                    "distance_to_centroid": float(
                        centroid_distances[
                            local_index
                        ]
                    ),
                    "sum_distance_to_cluster": float(
                        distance_sums[
                            local_index
                        ]
                    ),
                    "mean_distance_to_cluster": float(
                        distance_sums[
                            local_index
                        ]
                        / max(
                            1,
                            len(
                                cluster_positions
                            )
                            - 1,
                        )
                    ),
                    "medoid_centroid_same_record": (
                        medoid_centroid_same_record
                    ),
                }
            )

        diagnostic_rows.append(
            {
                "cluster": int(
                    cluster_id
                ),
                "cluster_size": int(
                    len(cluster_positions)
                ),
                "medoid_sample_position": (
                    medoid_global_position
                ),
                "centroid_exemplar_position": (
                    centroid_global_position
                ),
                "medoid_centroid_same_record": (
                    medoid_centroid_same_record
                ),
                "medoid_objective": float(
                    distance_sums[
                        medoid_local_index
                    ]
                ),
                "centroid_exemplar_medoid_objective": float(
                    distance_sums[
                        centroid_exemplar_local
                    ]
                ),
                "objective_improvement_from_true_medoid": float(
                    distance_sums[
                        centroid_exemplar_local
                    ]
                    - distance_sums[
                        medoid_local_index
                    ]
                ),
                "minimum_silhouette": float(
                    np.nanmin(
                        cluster_silhouettes
                    )
                ),
                "maximum_silhouette": float(
                    np.nanmax(
                        cluster_silhouettes
                    )
                ),
                "mean_silhouette": float(
                    np.nanmean(
                        cluster_silhouettes
                    )
                ),
                "negative_silhouette_fraction": float(
                    np.nanmean(
                        cluster_silhouettes
                        < 0
                    )
                ),
                "medoid_runtime_seconds": float(
                    medoid_runtime
                ),
                "medoid_calculation": (
                    "Exact minimum total "
                    "Euclidean distance"
                ),
            }
        )

        logging.info(
            "Cluster %s | size=%d | "
            "medoid runtime=%.3f seconds",
            cluster_id,
            len(cluster_positions),
            medoid_runtime,
        )

    return (
        pd.DataFrame(
            prototype_rows
        ),
        pd.DataFrame(
            diagnostic_rows
        ),
    )


def enrich_prototypes(
    prototypes: pd.DataFrame,
    features: pd.DataFrame,
    metadata: pd.DataFrame,
) -> pd.DataFrame:
    positions = prototypes[
        "sample_position"
    ].to_numpy(dtype=np.int64)

    selected_features = (
        features.iloc[
            positions
        ]
        .reset_index(drop=True)
        .copy()
    )

    selected_metadata = (
        metadata.iloc[
            positions
        ]
        .reset_index(drop=True)
        .copy()
    )

    duplicate_metadata_columns = [
        column
        for column in selected_metadata.columns
        if (
            column
            in prototypes.columns
            or column
            in selected_features.columns
        )
    ]

    selected_metadata = (
        selected_metadata.drop(
            columns=(
                duplicate_metadata_columns
            ),
            errors="ignore",
        )
    )

    enriched = pd.concat(
        [
            prototypes.reset_index(
                drop=True
            ),
            selected_metadata,
            selected_features,
        ],
        axis=1,
    )

    if DOMAIN_LABEL_PATH.exists():
        domain_labels = pd.read_csv(
            DOMAIN_LABEL_PATH
        )

        if "cluster" in domain_labels.columns:
            enriched = enriched.merge(
                domain_labels,
                on="cluster",
                how="left",
            )

    return enriched


def validate_prototype_selection(
    prototypes: pd.DataFrame,
    labels: np.ndarray,
) -> dict[str, Any]:
    required_types = {
        "medoid",
        "highest_silhouette",
        "boundary_point",
        "centroid_exemplar",
    }

    cluster_ids = set(
        np.unique(labels).tolist()
    )

    selected_cluster_ids = set(
        prototypes[
            "cluster"
        ].unique().tolist()
    )

    if cluster_ids != selected_cluster_ids:
        raise ValueError(
            "Prototype output does not cover "
            "all consensus clusters"
        )

    validation_rows = []

    for cluster_id in sorted(
        cluster_ids
    ):
        cluster_prototypes = prototypes[
            prototypes["cluster"]
            == cluster_id
        ]

        present_types = set(
            cluster_prototypes[
                "prototype_type"
            ].tolist()
        )

        missing_types = (
            required_types
            - present_types
        )

        if missing_types:
            raise ValueError(
                f"Cluster {cluster_id} is missing "
                f"prototype types: "
                f"{sorted(missing_types)}"
            )

        duplicate_types = bool(
            cluster_prototypes[
                "prototype_type"
            ].duplicated().any()
        )

        if duplicate_types:
            raise ValueError(
                f"Cluster {cluster_id} has "
                "duplicate prototype types"
            )

        medoid_row = cluster_prototypes[
            cluster_prototypes[
                "prototype_type"
            ]
            == "medoid"
        ].iloc[0]

        minimum_objective = (
            cluster_prototypes[
                "sum_distance_to_cluster"
            ].min()
        )

        validation_rows.append(
            {
                "cluster": int(
                    cluster_id
                ),
                "required_types_present": True,
                "medoid_is_actual_record": True,
                "prototype_count": int(
                    len(
                        cluster_prototypes
                    )
                ),
                "medoid_objective_in_selected_set": float(
                    medoid_row[
                        "sum_distance_to_cluster"
                    ]
                ),
                "minimum_selected_objective": float(
                    minimum_objective
                ),
            }
        )

    validation_frame = pd.DataFrame(
        validation_rows
    )

    validation_frame.to_csv(
        REPORT_DIR
        / "cluster_prototype_validation.csv",
        index=False,
    )

    return {
        "status": "valid",
        "cluster_count": int(
            len(cluster_ids)
        ),
        "prototype_count": int(
            len(prototypes)
        ),
        "required_types": sorted(
            required_types
        ),
        "all_clusters_covered": True,
    }


def plot_prototypes_in_pca(
    x: np.ndarray,
    labels: np.ndarray,
    prototypes: pd.DataFrame,
) -> Path:
    if x.shape[1] < 2:
        raise ValueError(
            "At least two PCA dimensions "
            "are required for plotting"
        )

    plt.figure(
        figsize=(11, 8)
    )

    scatter = plt.scatter(
        x[:, 0],
        x[:, 1],
        c=labels,
        cmap="tab20",
        s=13,
        alpha=0.35,
        rasterized=True,
    )

    marker_styles = {
        "medoid": {
            "marker": "*",
            "size": 280,
            "edgecolor": "black",
            "linewidth": 1.3,
        },
        "highest_silhouette": {
            "marker": "P",
            "size": 150,
            "edgecolor": "black",
            "linewidth": 1.0,
        },
        "boundary_point": {
            "marker": "X",
            "size": 150,
            "edgecolor": "red",
            "linewidth": 1.2,
        },
        "centroid_exemplar": {
            "marker": "D",
            "size": 100,
            "edgecolor": "white",
            "linewidth": 1.0,
        },
    }

    for prototype_type, style in (
        marker_styles.items()
    ):
        selected = prototypes[
            prototypes[
                "prototype_type"
            ]
            == prototype_type
        ]

        positions = selected[
            "sample_position"
        ].to_numpy(dtype=np.int64)

        plt.scatter(
            x[
                positions,
                0,
            ],
            x[
                positions,
                1,
            ],
            c=selected[
                "cluster"
            ],
            cmap="tab20",
            marker=style[
                "marker"
            ],
            s=style["size"],
            edgecolor=style[
                "edgecolor"
            ],
            linewidth=style[
                "linewidth"
            ],
            label=prototype_type,
        )

        for row in selected.itertuples(
            index=False
        ):
            position = int(
                row.sample_position
            )

            plt.annotate(
                f"C{int(row.cluster)}",
                (
                    x[position, 0],
                    x[position, 1],
                ),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
                weight="bold",
            )

    plt.colorbar(
        scatter,
        label="Consensus cluster",
    )

    plt.xlabel("PC1")
    plt.ylabel("PC2")

    plt.title(
        "Consensus Cluster Prototypes "
        "and Exemplars"
    )

    plt.legend(
        loc="best",
        frameon=True,
    )

    return save_figure(
        "cluster_prototypes_pca.png"
    )


def plot_prototype_silhouettes(
    prototypes: pd.DataFrame,
) -> Path:
    display_order = [
        "medoid",
        "centroid_exemplar",
        "highest_silhouette",
        "boundary_point",
    ]

    plt.figure(
        figsize=(12, 6)
    )

    sns.barplot(
        data=prototypes,
        x="cluster",
        y="silhouette",
        hue="prototype_type",
        hue_order=display_order,
    )

    plt.axhline(
        0,
        color="black",
        linestyle=":",
    )

    plt.xlabel(
        "Consensus cluster"
    )

    plt.ylabel(
        "Silhouette coefficient"
    )

    plt.title(
        "Silhouette Values of Cluster "
        "Prototypes and Boundary Points"
    )

    plt.legend(
        title="Prototype type",
        bbox_to_anchor=(
            1.02,
            1,
        ),
        loc="upper left",
    )

    return save_figure(
        "cluster_prototype_silhouettes.png"
    )


def plot_medoid_objective_comparison(
    prototypes: pd.DataFrame,
) -> Path:
    selected = prototypes[
        prototypes[
            "prototype_type"
        ].isin(
            [
                "medoid",
                "centroid_exemplar",
            ]
        )
    ].copy()

    plt.figure(
        figsize=(11, 6)
    )

    sns.barplot(
        data=selected,
        x="cluster",
        y="mean_distance_to_cluster",
        hue="prototype_type",
    )

    plt.xlabel(
        "Consensus cluster"
    )

    plt.ylabel(
        "Mean distance to all "
        "cluster members"
    )

    plt.title(
        "True Medoid versus "
        "Centroid-Nearest Exemplar"
    )

    return save_figure(
        "medoid_vs_centroid_exemplar.png"
    )


def write_report(
    diagnostics: pd.DataFrame,
    prototypes: pd.DataFrame,
) -> None:
    total_clusters = int(
        diagnostics["cluster"].nunique()
    )

    same_count = int(
        diagnostics[
            "medoid_centroid_same_record"
        ].sum()
    )

    different_count = (
        total_clusters - same_count
    )

    mean_improvement = float(
        diagnostics[
            "objective_improvement_from_true_medoid"
        ].mean()
    )

    maximum_improvement = float(
        diagnostics[
            "objective_improvement_from_true_medoid"
        ].max()
    )

    negative_boundary_count = int(
        np.sum(
            prototypes[
                "prototype_type"
            ].eq(
                "boundary_point"
            )
            & (
                prototypes[
                    "silhouette"
                ] < 0
            )
        )
    )

    lines = [
        "# Cluster Prototypes and Exemplars",
        "",
        "The final consensus clusters were "
        "interpreted using four representative "
        "records per cluster:",
        "",
        "- the exact medoid",
        "- the point with the highest silhouette",
        "- the most marginal boundary point",
        "- the record nearest to the arithmetic centroid",
        "",
        "The medoid is an actual observed record "
        "whose total Euclidean distance to all other "
        "records in its cluster is minimal.",
        "",
        "This differs from a centroid exemplar. "
        "A centroid exemplar is the observed record "
        "nearest to the arithmetic mean, but it does "
        "not necessarily minimise total pairwise "
        "distance.",
        "",
        f"Number of clusters: {total_clusters}",
        "",
        f"Clusters where medoid and centroid exemplar "
        f"are the same record: {same_count}",
        "",
        f"Clusters where they differ: {different_count}",
        "",
        f"Mean reduction in total within-cluster "
        f"distance from using the true medoid: "
        f"{mean_improvement:.6f}",
        "",
        f"Maximum reduction in total within-cluster "
        f"distance: {maximum_improvement:.6f}",
        "",
        f"Boundary points with negative silhouette: "
        f"{negative_boundary_count}",
        "",
        "A negative boundary-point silhouette "
        "indicates that the selected record is, on "
        "average, closer to another cluster than to "
        "its assigned cluster.",
        "",
        "The fraud label was not used to select any "
        "prototype. It is attached only for post-hoc "
        "inspection.",
        "",
        "Because V1 through V28 are anonymised PCA "
        "variables, prototypes support statistical "
        "interpretation but cannot reveal direct "
        "business semantics for those variables.",
    ]

    (
        REPORT_DIR
        / "cluster_prototype_analysis.md"
    ).write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main() -> None:
    ensure_directories()
    setup_logging()
    setup_plots()

    params = load_yaml(
        PARAMS_PATH
    )

    medoid_chunk_size = int(
        params.get(
            "interpretation",
            {},
        ).get(
            "medoid_chunk_size",
            256,
        )
    )

    medoid_chunk_size = max(
        1,
        medoid_chunk_size,
    )

    (
        x,
        labels,
        array_indices,
        row_ids,
        features,
        metadata,
        assignments,
    ) = load_inputs()

    with np.load(
        ARRAY_PATH,
        allow_pickle=False,
    ) as arrays:
        y_true = arrays[
            "y"
        ][
            array_indices
        ].astype(np.int8)

    (
        prototypes,
        diagnostics,
    ) = select_cluster_prototypes(
        x=x,
        labels=labels,
        array_indices=array_indices,
        row_ids=row_ids,
        y_true=y_true,
        chunk_size=medoid_chunk_size,
    )

    enriched_prototypes = (
        enrich_prototypes(
            prototypes=prototypes,
            features=features,
            metadata=metadata,
        )
    )

    validation = (
        validate_prototype_selection(
            prototypes=prototypes,
            labels=labels,
        )
    )

    prototypes.to_csv(
        REPORT_DIR
        / "cluster_prototypes.csv",
        index=False,
    )

    diagnostics.to_csv(
        REPORT_DIR
        / "cluster_medoid_diagnostics.csv",
        index=False,
    )

    enriched_prototypes.to_csv(
        REPORT_DIR
        / "cluster_prototype_features.csv",
        index=False,
    )

    enriched_prototypes.to_parquet(
        PROCESSED_DIR
        / "phase3_cluster_prototypes.parquet",
        index=False,
    )

    pca_figure = (
        plot_prototypes_in_pca(
            x=x,
            labels=labels,
            prototypes=prototypes,
        )
    )

    silhouette_figure = (
        plot_prototype_silhouettes(
            prototypes
        )
    )

    medoid_figure = (
        plot_medoid_objective_comparison(
            prototypes
        )
    )

    write_report(
        diagnostics=diagnostics,
        prototypes=prototypes,
    )

    completion_record = {
        "status": "completed",
        "cluster_count": int(
            len(
                np.unique(
                    labels
                )
            )
        ),
        "prototype_count": int(
            len(prototypes)
        ),
        "prototype_types": sorted(
            prototypes[
                "prototype_type"
            ].unique().tolist()
        ),
        "medoid_method": (
            "Exact minimum total Euclidean "
            "distance with chunked pairwise "
            "distance computation"
        ),
        "medoid_chunk_size": int(
            medoid_chunk_size
        ),
        "validation": validation,
        "medoid_centroid_same_cluster_count": int(
            diagnostics[
                "medoid_centroid_same_record"
            ].sum()
        ),
        "medoid_centroid_different_cluster_count": int(
            np.sum(
                ~diagnostics[
                    "medoid_centroid_same_record"
                ]
            )
        ),
        "generated_figures": [
            str(
                path.relative_to(ROOT)
            )
            for path in [
                pca_figure,
                silhouette_figure,
                medoid_figure,
            ]
        ],
    }

    save_json(
        completion_record,
        REPORT_DIR
        / "phase3_completion_step2_record.json",
    )

    logging.info(
        "Phase 3 completion step 2 succeeded"
    )


if __name__ == "__main__":
    main()