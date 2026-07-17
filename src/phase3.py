import json
import logging
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    calinski_harabasz_score,
    completeness_score,
    davies_bouldin_score,
    fowlkes_mallows_score,
    homogeneity_score,
    normalized_mutual_info_score,
    silhouette_samples,
    silhouette_score,
    v_measure_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree


ROOT = Path(__file__).resolve().parents[1]
PARAMS_PATH = ROOT / "params_phase3.yaml"
REPORT_DIR = ROOT / "reports" / "phase3"
FIGURE_DIR = ROOT / "reports" / "figures" / "phase3"
MODEL_DIR = ROOT / "models" / "phase3"
PROCESSED_DIR = ROOT / "data" / "processed"


def ensure_directories() -> None:
    for directory in (
        REPORT_DIR,
        FIGURE_DIR,
        MODEL_DIR,
        PROCESSED_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(
                REPORT_DIR / "phase3_execution.log",
                mode="w",
                encoding="utf-8",
            ),
        ],
        force=True,
    )


def setup_plots(dpi: int) -> None:
    sns.set_theme(
        style="whitegrid",
        context="notebook",
        palette="colorblind",
    )
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": dpi,
            "figure.autolayout": True,
        }
    )


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {path}"
        )

    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)

    if not isinstance(data, dict):
        raise ValueError(
            f"Invalid configuration file: {path}"
        )

    return data


def save_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(
            data,
            file,
            ensure_ascii=False,
            indent=2,
            default=str,
        )


def save_figure(filename: str) -> None:
    path = FIGURE_DIR / filename
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    logging.info("Saved figure: %s", path)


def load_phase1_arrays(
    params: dict[str, Any],
) -> dict[str, np.ndarray]:
    path = ROOT / params["data"]["phase1_arrays"]

    if not path.exists():
        raise FileNotFoundError(
            f"Phase 1 arrays not found: {path}"
        )

    with np.load(path, allow_pickle=False) as loaded:
        required = {
            "X_standard",
            "X_robust",
            "X_pca",
            "y",
            "row_id",
            "train_indices",
        }

        missing = required.difference(loaded.files)

        if missing:
            raise KeyError(
                f"Missing Phase 1 arrays: {sorted(missing)}"
            )

        result = {
            name: loaded[name]
            for name in loaded.files
        }

    row_count = len(result["y"])

    for name in (
        "X_standard",
        "X_robust",
        "X_pca",
        "row_id",
    ):
        if len(result[name]) != row_count:
            raise ValueError(
                f"Inconsistent row count in {name}"
            )

    return result


def load_supporting_tables(
    params: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_path = (
        ROOT / params["data"]["unscaled_features"]
    )
    metadata_path = (
        ROOT / params["data"]["metadata"]
    )

    if not feature_path.exists():
        raise FileNotFoundError(
            f"Feature table not found: {feature_path}"
        )

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Metadata table not found: {metadata_path}"
        )

    features = pd.read_parquet(feature_path)
    metadata = pd.read_parquet(metadata_path)

    if len(features) != len(metadata):
        raise ValueError(
            "Features and metadata have different row counts"
        )

    if features.empty:
        raise ValueError("Feature table is empty")

    invalid_columns = [
        column
        for column in features.columns
        if not pd.api.types.is_numeric_dtype(
            features[column]
        )
    ]

    if invalid_columns:
        raise TypeError(
            f"Non-numeric features found: {invalid_columns}"
        )

    if features.isna().any().any():
        raise ValueError(
            "Feature table contains missing values"
        )

    return features, metadata


def select_consensus_indices(
    phase1: dict[str, np.ndarray],
    params: dict[str, Any],
    random_seed: int,
) -> np.ndarray:
    sample_size = int(
        params["consensus"]["sample_size"]
    )

    if sample_size < 3:
        raise ValueError(
            "consensus.sample_size must be at least 3"
        )

    phase2_sample_path = (
        ROOT / params["data"]["phase2_sample"]
    )

    if phase2_sample_path.exists():
        sample_table = pd.read_csv(
            phase2_sample_path
        )

        if "array_index" not in sample_table.columns:
            raise KeyError(
                "array_index is missing from Phase 2 sample"
            )

        candidates = sample_table[
            "array_index"
        ].to_numpy(dtype=np.int64)
    else:
        candidates = phase1[
            "train_indices"
        ].astype(np.int64)

    candidates = np.unique(candidates)
    total_rows = len(phase1["y"])

    if np.any(candidates < 0):
        raise IndexError("Negative indices found")

    if np.any(candidates >= total_rows):
        raise IndexError(
            "Sample indices exceed Phase 1 row count"
        )

    sample_size = min(sample_size, len(candidates))

    if sample_size < 3:
        raise ValueError(
            "Not enough candidate records"
        )

    rng = np.random.default_rng(random_seed)

    selected = rng.choice(
        candidates,
        size=sample_size,
        replace=False,
    )

    return np.sort(selected)


def purity_score(
    y_true: np.ndarray,
    labels: np.ndarray,
) -> float:
    table = pd.crosstab(
        pd.Series(y_true, name="true"),
        pd.Series(labels, name="cluster"),
    )

    if table.empty:
        return float("nan")

    return float(
        table.max(axis=0).sum()
        / table.to_numpy().sum()
    )


def evaluate_labels(
    algorithm: str,
    parameters: str,
    x: np.ndarray,
    labels: np.ndarray,
    y_true: np.ndarray,
    silhouette_sample_size: int,
    random_seed: int,
    requested_k: int,
) -> dict[str, Any]:
    labels = np.asarray(labels)
    y_true = np.asarray(y_true)

    if len(x) != len(labels):
        raise ValueError(
            "Feature and label lengths differ"
        )

    if len(y_true) != len(labels):
        raise ValueError(
            "External label length differs"
        )

    unique_labels = np.unique(labels)

    valid = (
        2 <= len(unique_labels) < len(labels)
    )

    if valid:
        sample_size = min(
            silhouette_sample_size,
            len(x),
        )

        silhouette = float(
            silhouette_score(
                x,
                labels,
                metric="euclidean",
                sample_size=sample_size,
                random_state=random_seed,
            )
        )

        davies_bouldin = float(
            davies_bouldin_score(x, labels)
        )

        calinski_harabasz = float(
            calinski_harabasz_score(x, labels)
        )
    else:
        silhouette = float("nan")
        davies_bouldin = float("nan")
        calinski_harabasz = float("nan")

    return {
        "algorithm": algorithm,
        "parameters": parameters,
        "requested_k": requested_k,
        "n_records": int(len(x)),
        "n_clusters": int(len(unique_labels)),
        "silhouette": silhouette,
        "davies_bouldin": davies_bouldin,
        "calinski_harabasz": calinski_harabasz,
        "ari": float(
            adjusted_rand_score(y_true, labels)
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
            homogeneity_score(y_true, labels)
        ),
        "completeness": float(
            completeness_score(y_true, labels)
        ),
        "v_measure": float(
            v_measure_score(y_true, labels)
        ),
        "purity": purity_score(
            y_true,
            labels,
        ),
    }


def generate_base_clusterings(
    x_pca: np.ndarray,
    params: dict[str, Any],
) -> tuple[list[np.ndarray], pd.DataFrame]:
    k_min = int(params["consensus"]["k_min"])
    k_max = int(params["consensus"]["k_max"])

    if k_min < 2 or k_max < k_min:
        raise ValueError("Invalid k range")

    if k_max >= len(x_pca):
        raise ValueError(
            "k_max must be smaller than sample size"
        )

    k_values = range(k_min, k_max + 1)

    feature_dimensions = sorted(
        {
            max(
                1,
                min(
                    int(value),
                    x_pca.shape[1],
                ),
            )
            for value in params["consensus"][
                "feature_dimensions"
            ]
        }
    )

    kmeans_seeds = [
        int(value)
        for value in params["consensus"][
            "kmeans_seeds"
        ]
    ]

    gmm_seeds = [
        int(value)
        for value in params["consensus"][
            "gmm_seeds"
        ]
    ]

    covariance_types = [
        str(value)
        for value in params["consensus"][
            "gmm_covariance_types"
        ]
    ]

    valid_covariances = {
        "spherical",
        "diag",
        "tied",
        "full",
    }

    invalid_covariances = (
        set(covariance_types)
        - valid_covariances
    )

    if invalid_covariances:
        raise ValueError(
            f"Invalid covariance types: {invalid_covariances}"
        )

    label_collection: list[np.ndarray] = []
    records: list[dict[str, Any]] = []

    for dimension in feature_dimensions:
        matrix = x_pca[:, :dimension]

        for k in k_values:
            for seed in kmeans_seeds:
                logging.info(
                    "KMeans dim=%d k=%d seed=%d",
                    dimension,
                    k,
                    seed,
                )

                model = KMeans(
                    n_clusters=k,
                    init="k-means++",
                    n_init=10,
                    max_iter=300,
                    random_state=seed,
                )

                started = time.perf_counter()
                labels = model.fit_predict(matrix)
                runtime = (
                    time.perf_counter() - started
                )

                label_collection.append(
                    labels.astype(np.int32)
                )

                records.append(
                    {
                        "base_id": len(label_collection) - 1,
                        "algorithm": "KMeans",
                        "dimension": dimension,
                        "k": k,
                        "seed": seed,
                        "covariance": None,
                        "runtime_seconds": runtime,
                        "inertia": float(model.inertia_),
                        "bic": None,
                        "converged": True,
                    }
                )

    for dimension in feature_dimensions:
        matrix = x_pca[:, :dimension]

        for covariance_type in covariance_types:
            for k in k_values:
                for seed in gmm_seeds:
                    logging.info(
                        "GMM dim=%d k=%d covariance=%s seed=%d",
                        dimension,
                        k,
                        covariance_type,
                        seed,
                    )

                    model = GaussianMixture(
                        n_components=k,
                        covariance_type=covariance_type,
                        n_init=1,
                        max_iter=200,
                        reg_covar=1e-6,
                        random_state=seed,
                    )

                    started = time.perf_counter()

                    with warnings.catch_warnings():
                        warnings.simplefilter(
                            "ignore",
                            ConvergenceWarning,
                        )
                        labels = model.fit_predict(
                            matrix
                        )

                    runtime = (
                        time.perf_counter()
                        - started
                    )

                    label_collection.append(
                        labels.astype(np.int32)
                    )

                    records.append(
                        {
                            "base_id": len(label_collection) - 1,
                            "algorithm": "GMM",
                            "dimension": dimension,
                            "k": k,
                            "seed": seed,
                            "covariance": covariance_type,
                            "runtime_seconds": runtime,
                            "inertia": None,
                            "bic": float(
                                model.bic(matrix)
                            ),
                            "converged": bool(
                                model.converged_
                            ),
                        }
                    )

    hierarchy_dimensions = feature_dimensions[:2]

    for dimension in hierarchy_dimensions:
        matrix = x_pca[:, :dimension]

        for k in k_values:
            logging.info(
                "Hierarchical dim=%d k=%d",
                dimension,
                k,
            )

            model = AgglomerativeClustering(
                n_clusters=k,
                linkage="ward",
            )

            started = time.perf_counter()
            labels = model.fit_predict(matrix)
            runtime = (
                time.perf_counter() - started
            )

            label_collection.append(
                labels.astype(np.int32)
            )

            records.append(
                {
                    "base_id": len(label_collection) - 1,
                    "algorithm": "Hierarchical-Ward",
                    "dimension": dimension,
                    "k": k,
                    "seed": None,
                    "covariance": None,
                    "runtime_seconds": runtime,
                    "inertia": None,
                    "bic": None,
                    "converged": True,
                }
            )

    if not label_collection:
        raise RuntimeError(
            "No base clusterings generated"
        )

    return (
        label_collection,
        pd.DataFrame(records),
    )


def evaluate_base_clusterings(
    x: np.ndarray,
    y_true: np.ndarray,
    labels_collection: list[np.ndarray],
    manifest: pd.DataFrame,
    params: dict[str, Any],
    random_seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for record in manifest.itertuples(
        index=False
    ):
        base_id = int(record.base_id)
        labels = labels_collection[base_id]

        parameters = (
            f"base_id={base_id},"
            f"k={int(record.k)},"
            f"dimension={int(record.dimension)},"
            f"seed={record.seed},"
            f"covariance={record.covariance}"
        )

        metrics = evaluate_labels(
            algorithm=str(record.algorithm),
            parameters=parameters,
            x=x,
            labels=labels,
            y_true=y_true,
            silhouette_sample_size=int(
                params["evaluation"][
                    "silhouette_sample_size"
                ]
            ),
            random_seed=random_seed,
            requested_k=int(record.k),
        )

        metrics["base_id"] = base_id
        metrics["dimension"] = int(
            record.dimension
        )
        metrics["seed"] = record.seed
        metrics["covariance"] = (
            record.covariance
        )

        rows.append(metrics)

    return pd.DataFrame(rows)


def build_coassociation_matrix(
    labels_collection: list[np.ndarray],
) -> np.ndarray:
    if not labels_collection:
        raise ValueError(
            "No base labels supplied"
        )

    run_count = len(labels_collection)
    record_count = len(labels_collection[0])

    count_dtype = (
        np.uint16
        if run_count <= np.iinfo(np.uint16).max
        else np.uint32
    )

    counts = np.zeros(
        (record_count, record_count),
        dtype=count_dtype,
    )

    for run_number, labels in enumerate(
        labels_collection,
        start=1,
    ):
        if len(labels) != record_count:
            raise ValueError(
                "Base label lengths differ"
            )

        logging.info(
            "Co-association %d/%d",
            run_number,
            run_count,
        )

        for cluster_id in np.unique(labels):
            members = np.flatnonzero(
                labels == cluster_id
            )
            counts[
                np.ix_(members, members)
            ] += 1

    matrix = counts.astype(np.float32)
    matrix /= float(run_count)
    matrix = np.clip(matrix, 0.0, 1.0)
    np.fill_diagonal(matrix, 1.0)

    return matrix


def consensus_k_search(
    x: np.ndarray,
    coassociation: np.ndarray,
    y_true: np.ndarray,
    params: dict[str, Any],
    random_seed: int,
) -> tuple[
    pd.DataFrame,
    dict[int, np.ndarray],
    np.ndarray,
]:
    expected_shape = (len(x), len(x))

    if coassociation.shape != expected_shape:
        raise ValueError(
            "Invalid co-association shape"
        )

    distance = np.clip(
        1.0 - coassociation,
        0.0,
        1.0,
    )

    distance = (
        distance + distance.T
    ) / 2.0

    np.fill_diagonal(distance, 0.0)

    condensed = squareform(
        distance,
        checks=True,
    )

    consensus_tree = linkage(
        condensed,
        method="average",
    )

    rows: list[dict[str, Any]] = []
    labels_by_k: dict[int, np.ndarray] = {}

    k_min = int(params["consensus"]["k_min"])
    k_max = int(params["consensus"]["k_max"])

    for requested_k in range(
        k_min,
        k_max + 1,
    ):
        labels = (
            fcluster(
                consensus_tree,
                t=requested_k,
                criterion="maxclust",
            )
            - 1
        ).astype(np.int32)

        labels_by_k[requested_k] = labels

        rows.append(
            evaluate_labels(
                algorithm="Consensus",
                parameters=(
                    f"k={requested_k},"
                    "linkage=average"
                ),
                x=x,
                labels=labels,
                y_true=y_true,
                silhouette_sample_size=int(
                    params["evaluation"][
                        "silhouette_sample_size"
                    ]
                ),
                random_seed=random_seed,
                requested_k=requested_k,
            )
        )

    return (
        pd.DataFrame(rows),
        labels_by_k,
        consensus_tree,
    )


def select_consensus_k(
    results: pd.DataFrame,
) -> int:
    candidates = results.dropna(
        subset=[
            "silhouette",
            "davies_bouldin",
            "calinski_harabasz",
        ]
    ).copy()

    if candidates.empty:
        raise ValueError(
            "No valid consensus result"
        )

    candidates["rank_silhouette"] = (
        candidates["silhouette"].rank(
            ascending=False,
            method="min",
        )
    )

    candidates["rank_db"] = (
        candidates["davies_bouldin"].rank(
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

    candidates["rank_sum"] = (
        candidates["rank_silhouette"]
        + candidates["rank_db"]
        + candidates["rank_ch"]
    )

    selected = candidates.sort_values(
        [
            "rank_sum",
            "silhouette",
            "requested_k",
        ],
        ascending=[
            True,
            False,
            True,
        ],
    ).iloc[0]

    return int(selected["requested_k"])


def compare_consensus_with_best_base(
    consensus_row: pd.Series,
    base_results: pd.DataFrame,
) -> pd.DataFrame:
    candidates = base_results.dropna(
        subset=[
            "silhouette",
            "davies_bouldin",
            "calinski_harabasz",
        ]
    ).copy()

    if candidates.empty:
        raise ValueError(
            "No valid base result"
        )

    candidates["rank_silhouette"] = (
        candidates["silhouette"].rank(
            ascending=False,
            method="min",
        )
    )

    candidates["rank_db"] = (
        candidates["davies_bouldin"].rank(
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

    candidates["rank_sum"] = (
        candidates["rank_silhouette"]
        + candidates["rank_db"]
        + candidates["rank_ch"]
    )

    best_base = candidates.sort_values(
        ["rank_sum", "silhouette"],
        ascending=[True, False],
    ).iloc[0]

    fields = [
        "algorithm",
        "parameters",
        "requested_k",
        "n_records",
        "n_clusters",
        "silhouette",
        "davies_bouldin",
        "calinski_harabasz",
        "ari",
        "nmi",
        "ami",
        "fowlkes_mallows",
        "homogeneity",
        "completeness",
        "v_measure",
        "purity",
    ]

    base_record = {
        field: best_base.get(field)
        for field in fields
    }

    consensus_record = {
        field: consensus_row.get(field)
        for field in fields
    }

    base_record["algorithm"] = (
        "Best base: "
        + str(base_record["algorithm"])
    )

    consensus_record["algorithm"] = (
        "Consensus"
    )

    return pd.DataFrame(
        [base_record, consensus_record]
    )


def plot_coassociation(
    matrix: np.ndarray,
    labels: np.ndarray,
) -> None:
    order = np.argsort(
        labels,
        kind="stable",
    )

    ordered = matrix[
        np.ix_(order, order)
    ]

    plt.figure(figsize=(9, 8))

    sns.heatmap(
        ordered,
        cmap="viridis",
        vmin=0,
        vmax=1,
        xticklabels=False,
        yticklabels=False,
        cbar_kws={
            "label": "Co-assignment probability"
        },
    )

    plt.title(
        "Consensus co-association matrix"
    )

    plt.xlabel(
        "Observations ordered by cluster"
    )

    plt.ylabel(
        "Observations ordered by cluster"
    )

    save_figure(
        "coassociation_heatmap.png"
    )


def plot_consensus_selection(
    results: pd.DataFrame,
    selected_k: int,
) -> None:
    figure, axes = plt.subplots(
        1,
        3,
        figsize=(16, 4.5),
    )

    axes[0].plot(
        results["requested_k"],
        results["silhouette"],
        marker="o",
    )
    axes[0].axvline(
        selected_k,
        color="red",
        linestyle="--",
    )
    axes[0].set_xlabel("k")
    axes[0].set_ylabel("Silhouette")
    axes[0].set_title("Silhouette")

    axes[1].plot(
        results["requested_k"],
        results["davies_bouldin"],
        marker="o",
        color="#ff7f0e",
    )
    axes[1].axvline(
        selected_k,
        color="red",
        linestyle="--",
    )
    axes[1].set_xlabel("k")
    axes[1].set_ylabel("Davies-Bouldin")
    axes[1].set_title("Davies-Bouldin")

    axes[2].plot(
        results["requested_k"],
        results["calinski_harabasz"],
        marker="o",
        color="#2ca02c",
    )
    axes[2].axvline(
        selected_k,
        color="red",
        linestyle="--",
    )
    axes[2].set_xlabel("k")
    axes[2].set_ylabel(
        "Calinski-Harabasz"
    )
    axes[2].set_title(
        "Calinski-Harabasz"
    )

    figure.suptitle(
        f"Consensus cluster selection: k={selected_k}"
    )

    save_figure(
        "consensus_k_selection.png"
    )


def plot_consensus_embedding(
    x_pca: np.ndarray,
    labels: np.ndarray,
) -> None:
    if x_pca.shape[1] < 2:
        raise ValueError(
            "At least two PCA features required"
        )

    plt.figure(figsize=(9, 7))

    scatter = plt.scatter(
        x_pca[:, 0],
        x_pca[:, 1],
        c=labels,
        cmap="tab20",
        s=14,
        alpha=0.65,
        rasterized=True,
    )

    plt.colorbar(
        scatter,
        label="Consensus cluster",
    )

    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("Consensus PCA projection")

    save_figure(
        "consensus_pca_projection.png"
    )


def plot_advanced_comparison(
    comparison: pd.DataFrame,
) -> None:
    metrics = [
        "silhouette",
        "davies_bouldin",
        "calinski_harabasz",
        "ari",
        "nmi",
        "ami",
    ]

    melted = comparison.melt(
        id_vars=["algorithm"],
        value_vars=metrics,
        var_name="metric",
        value_name="value",
    )

    graph = sns.catplot(
        data=melted,
        x="algorithm",
        y="value",
        col="metric",
        col_wrap=3,
        kind="bar",
        sharey=False,
        height=4,
        aspect=1.15,
    )

    graph.set_xticklabels(rotation=20)

    graph.fig.suptitle(
        "Consensus versus best base clustering",
        y=1.02,
    )

    path = (
        FIGURE_DIR
        / "consensus_vs_best_base.png"
    )

    graph.savefig(
        path,
        bbox_inches="tight",
    )

    plt.close("all")


def create_cluster_profiles(
    features: pd.DataFrame,
    labels: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if len(features) != len(labels):
        raise ValueError(
            "Feature and label lengths differ"
        )

    data = features.copy()
    data["cluster"] = labels

    means = data.groupby(
        "cluster"
    ).mean(numeric_only=True)

    medians = data.groupby(
        "cluster"
    ).median(numeric_only=True)

    global_mean = features.mean(
        numeric_only=True
    )

    global_std = (
        features.std(numeric_only=True)
        .replace(0, 1.0)
        .fillna(1.0)
    )

    z_profiles = (
        means - global_mean
    ) / global_std

    rows: list[dict[str, Any]] = []

    for cluster_id in means.index:
        mask = labels == cluster_id

        row: dict[str, Any] = {
            "cluster": int(cluster_id),
            "size": int(mask.sum()),
            "fraction": float(mask.mean()),
        }

        for feature in means.columns:
            row[f"{feature}_mean"] = float(
                means.loc[
                    cluster_id,
                    feature,
                ]
            )

            row[f"{feature}_median"] = float(
                medians.loc[
                    cluster_id,
                    feature,
                ]
            )

            row[f"{feature}_z"] = float(
                z_profiles.loc[
                    cluster_id,
                    feature,
                ]
            )

        rows.append(row)

    return pd.DataFrame(rows), z_profiles


def create_domain_labels(
    z_profiles: pd.DataFrame,
    top_n: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    top_n = max(1, int(top_n))

    for cluster_id, values in (
        z_profiles.iterrows()
    ):
        features = (
            values.abs()
            .sort_values(ascending=False)
            .head(top_n)
            .index
        )

        descriptions = []

        for feature in features:
            direction = (
                "high"
                if values[feature] >= 0
                else "low"
            )
            descriptions.append(
                f"{direction} {feature}"
            )

        rows.append(
            {
                "cluster": int(cluster_id),
                "proposed_domain_label": (
                    f"Cluster {cluster_id}: "
                    + ", ".join(descriptions)
                ),
            }
        )

    return pd.DataFrame(rows)


def plot_cluster_profiles(
    z_profiles: pd.DataFrame,
) -> None:
    strengths = (
        z_profiles.abs()
        .max(axis=0)
        .sort_values(ascending=False)
    )

    selected = strengths.head(15).index

    height = max(
        5.0,
        float(len(z_profiles)) * 0.8,
    )

    plt.figure(
        figsize=(14, height)
    )

    sns.heatmap(
        z_profiles[selected],
        cmap="vlag",
        center=0,
        annot=True,
        fmt=".2f",
    )

    plt.xlabel("Feature")
    plt.ylabel("Cluster")
    plt.title("Cluster profile heatmap")

    save_figure(
        "cluster_profile_heatmap.png"
    )


def fraud_composition(
    labels: np.ndarray,
    y_true: np.ndarray,
) -> pd.DataFrame:
    data = pd.DataFrame(
        {
            "cluster": labels,
            "Class": y_true,
        }
    )

    result = (
        data.groupby("cluster")
        .agg(
            size=("Class", "size"),
            fraud_count=("Class", "sum"),
            fraud_rate=("Class", "mean"),
        )
        .reset_index()
    )

    global_rate = float(y_true.mean())

    result["global_fraud_rate"] = (
        global_rate
    )

    result["fraud_lift"] = np.where(
        global_rate > 0,
        result["fraud_rate"] / global_rate,
        np.nan,
    )

    return result


def plot_fraud_composition(
    composition: pd.DataFrame,
) -> None:
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(13, 5),
    )

    sns.barplot(
        data=composition,
        x="cluster",
        y="fraud_rate",
        ax=axes[0],
    )

    axes[0].axhline(
        composition[
            "global_fraud_rate"
        ].iloc[0],
        color="red",
        linestyle="--",
    )

    axes[0].set_title(
        "Fraud rate by cluster"
    )

    sns.barplot(
        data=composition,
        x="cluster",
        y="fraud_lift",
        ax=axes[1],
    )

    axes[1].axhline(
        1.0,
        color="red",
        linestyle="--",
    )

    axes[1].set_title(
        "Fraud lift by cluster"
    )

    figure.suptitle(
        "Post-hoc fraud composition"
    )

    save_figure(
        "fraud_composition_by_cluster.png"
    )


def extract_exemplars(
    x: np.ndarray,
    labels: np.ndarray,
    array_indices: np.ndarray,
    row_ids: np.ndarray,
    y_true: np.ndarray,
) -> pd.DataFrame:
    unique_labels = np.unique(labels)

    if 2 <= len(unique_labels) < len(labels):
        silhouette_values = silhouette_samples(
            x,
            labels,
            metric="euclidean",
        )
    else:
        silhouette_values = np.full(
            len(x),
            np.nan,
        )

    rows: list[dict[str, Any]] = []

    for cluster_id in unique_labels:
        positions = np.flatnonzero(
            labels == cluster_id
        )

        cluster_x = x[positions]
        centroid = cluster_x.mean(axis=0)

        distances = np.linalg.norm(
            cluster_x - centroid,
            axis=1,
        )

        cluster_silhouettes = (
            silhouette_values[positions]
        )

        central_local = int(
            np.argmin(distances)
        )

        if np.all(
            np.isnan(cluster_silhouettes)
        ):
            representative_local = central_local
            boundary_local = int(
                np.argmax(distances)
            )
        else:
            representative_local = int(
                np.nanargmax(
                    cluster_silhouettes
                )
            )
            boundary_local = int(
                np.nanargmin(
                    cluster_silhouettes
                )
            )

        selections = {
            "centroid_exemplar": central_local,
            "highest_silhouette": representative_local,
            "boundary_point": boundary_local,
        }

        for exemplar_type, local_index in (
            selections.items()
        ):
            position = int(
                positions[local_index]
            )

            rows.append(
                {
                    "cluster": int(cluster_id),
                    "exemplar_type": exemplar_type,
                    "sample_position": position,
                    "array_index": int(
                        array_indices[position]
                    ),
                    "row_id": int(
                        row_ids[position]
                    ),
                    "Class_external_only": int(
                        y_true[position]
                    ),
                    "silhouette": float(
                        silhouette_values[position]
                    ),
                    "distance_to_centroid": float(
                        distances[local_index]
                    ),
                }
            )

    return pd.DataFrame(rows)


def fit_rule_extractor(
    features: pd.DataFrame,
    labels: np.ndarray,
    params: dict[str, Any],
    random_seed: int,
) -> tuple[
    DecisionTreeClassifier,
    dict[str, Any],
]:
    unique_labels, counts = np.unique(
        labels,
        return_counts=True,
    )

    if len(unique_labels) < 2:
        raise ValueError(
            "At least two clusters required"
        )

    stratify = (
        labels
        if np.all(counts >= 2)
        else None
    )

    (
        x_train,
        x_test,
        y_train,
        y_test,
    ) = train_test_split(
        features,
        labels,
        test_size=0.25,
        random_state=random_seed,
        stratify=stratify,
    )

    model = DecisionTreeClassifier(
        max_depth=int(
            params["interpretation"][
                "tree_max_depth"
            ]
        ),
        min_samples_leaf=int(
            params["interpretation"][
                "tree_min_samples_leaf"
            ]
        ),
        class_weight="balanced",
        random_state=random_seed,
    )

    model.fit(x_train, y_train)

    metrics = {
        "train_accuracy": float(
            model.score(x_train, y_train)
        ),
        "test_accuracy": float(
            model.score(x_test, y_test)
        ),
        "tree_depth": int(
            model.get_depth()
        ),
        "leaf_count": int(
            model.get_n_leaves()
        ),
    }

    rules = export_text(
        model,
        feature_names=list(
            features.columns
        ),
    )

    (
        REPORT_DIR / "cluster_rules.txt"
    ).write_text(
        rules,
        encoding="utf-8",
    )

    joblib.dump(
        model,
        MODEL_DIR
        / "cluster_rule_tree.joblib",
    )

    importance = pd.DataFrame(
        {
            "feature": features.columns,
            "tree_importance": (
                model.feature_importances_
            ),
        }
    ).sort_values(
        "tree_importance",
        ascending=False,
    )

    importance.to_csv(
        REPORT_DIR
        / "tree_feature_importance.csv",
        index=False,
    )

    plt.figure(figsize=(22, 11))

    plot_tree(
        model,
        feature_names=list(
            features.columns
        ),
        class_names=[
            str(value)
            for value in model.classes_
        ],
        filled=True,
        rounded=True,
        fontsize=7,
    )

    plt.title(
        "Cluster rule extraction tree"
    )

    save_figure(
        "cluster_rule_tree.png"
    )

    return model, metrics


def run_optional_shap(
    model: DecisionTreeClassifier,
    features: pd.DataFrame,
    random_seed: int,
) -> dict[str, Any]:
    try:
        import shap
    except ImportError:
        return {
            "status": "skipped",
            "reason": "shap is not installed",
        }

    sample = features.sample(
        n=min(1000, len(features)),
        random_state=random_seed,
    )

    try:
        explainer = shap.TreeExplainer(model)
        raw_values = explainer.shap_values(
            sample
        )

        if isinstance(raw_values, list):
            values = np.stack(
                [
                    np.abs(
                        np.asarray(item)
                    ).mean(axis=0)
                    for item in raw_values
                ],
                axis=0,
            ).mean(axis=0)
        else:
            array = np.asarray(raw_values)

            if array.ndim == 2:
                values = np.abs(array).mean(
                    axis=0
                )
            elif array.ndim == 3:
                values = np.abs(array).mean(
                    axis=(0, 2)
                )
            else:
                raise ValueError(
                    f"Unexpected SHAP shape: {array.shape}"
                )

        values = np.asarray(values).reshape(-1)

        if len(values) != len(
            sample.columns
        ):
            raise ValueError(
                "SHAP feature count mismatch"
            )

        result = pd.DataFrame(
            {
                "feature": sample.columns,
                "mean_absolute_shap": values,
            }
        ).sort_values(
            "mean_absolute_shap",
            ascending=False,
        )

        result.to_csv(
            REPORT_DIR
            / "shap_feature_importance.csv",
            index=False,
        )

        plt.figure(figsize=(9, 6))

        sns.barplot(
            data=result.head(15),
            x="mean_absolute_shap",
            y="feature",
        )

        plt.title(
            "SHAP feature importance"
        )

        save_figure(
            "shap_feature_importance.png"
        )

        return {
            "status": "completed",
            "sample_size": int(len(sample)),
        }

    except Exception as error:
        logging.exception("SHAP failed")

        return {
            "status": "failed",
            "error_type": type(error).__name__,
            "error_message": str(error),
        }


def cluster_anomaly_analysis(
    x: np.ndarray,
    labels: np.ndarray,
    array_indices: np.ndarray,
    row_ids: np.ndarray,
    y_true: np.ndarray,
    params: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    distances = np.zeros(
        len(x),
        dtype=float,
    )

    scores = np.zeros(
        len(x),
        dtype=float,
    )

    for cluster_id in np.unique(labels):
        positions = np.flatnonzero(
            labels == cluster_id
        )

        cluster_x = x[positions]
        centroid = cluster_x.mean(axis=0)

        cluster_distances = np.linalg.norm(
            cluster_x - centroid,
            axis=1,
        )

        median = float(
            np.median(cluster_distances)
        )

        mad = float(
            np.median(
                np.abs(
                    cluster_distances - median
                )
            )
        )

        scale = max(
            1.4826 * mad,
            np.finfo(float).eps,
        )

        distances[positions] = (
            cluster_distances
        )

        scores[positions] = (
            cluster_distances - median
        ) / scale

    result = pd.DataFrame(
        {
            "array_index": array_indices,
            "row_id": row_ids,
            "cluster": labels,
            "Class_external_only": y_true,
            "distance_to_cluster_centroid": distances,
            "robust_cluster_anomaly_score": scores,
        }
    )

    result = result.sort_values(
        "robust_cluster_anomaly_score",
        ascending=False,
    ).reset_index(drop=True)

    result["anomaly_rank"] = (
        np.arange(len(result)) + 1
    )

    quantile = float(
        params["anomaly"][
            "contamination_quantile"
        ]
    )

    if not 0.0 < quantile < 1.0:
        raise ValueError(
            "Invalid contamination quantile"
        )

    threshold = float(
        np.quantile(scores, quantile)
    )

    top_n = min(
        int(params["anomaly"]["top_n"]),
        len(result),
    )

    flagged = result[
        "robust_cluster_anomaly_score"
    ] >= threshold

    top = result.head(top_n)

    global_fraud_rate = float(
        y_true.mean()
    )

    flagged_fraud_rate = (
        float(
            result.loc[
                flagged,
                "Class_external_only",
            ].mean()
        )
        if flagged.any()
        else None
    )

    top_fraud_rate = (
        float(
            top[
                "Class_external_only"
            ].mean()
        )
        if len(top) > 0
        else None
    )

    summary = {
        "threshold": threshold,
        "quantile": quantile,
        "top_n": top_n,
        "flagged_count": int(
            flagged.sum()
        ),
        "flagged_fraction": float(
            flagged.mean()
        ),
        "fraud_rate_all": global_fraud_rate,
        "fraud_rate_top_n": top_fraud_rate,
        "fraud_rate_flagged": (
            flagged_fraud_rate
        ),
    }

    return result, summary


def plot_anomaly_scores(
    anomaly_data: pd.DataFrame,
) -> None:
    data = anomaly_data.copy()

    data["DisplayClass"] = data[
        "Class_external_only"
    ].map(
        {
            0: "Legitimate",
            1: "Fraud",
        }
    )

    plt.figure(figsize=(10, 6))

    sns.histplot(
        data=data,
        x="robust_cluster_anomaly_score",
        hue="DisplayClass",
        bins=80,
        element="step",
        stat="density",
        common_norm=False,
    )

    plt.yscale("log")
    plt.title("Cluster anomaly scores")

    save_figure(
        "cluster_anomaly_scores.png"
    )


def preprocessing_sensitivity(
    x_standard: np.ndarray,
    x_robust: np.ndarray,
    x_pca: np.ndarray,
    selected_k: int,
    random_seed: int,
    params: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    representations = {
        "StandardScaler": x_standard,
        "RobustScaler": x_robust,
        "PCA_of_Standard": x_pca,
    }

    n_init = int(
        params["sensitivity"][
            "kmeans_n_init"
        ]
    )

    label_sets: dict[str, np.ndarray] = {}
    metric_rows: list[dict[str, Any]] = []

    for name, matrix in (
        representations.items()
    ):
        model = KMeans(
            n_clusters=selected_k,
            init="k-means++",
            n_init=n_init,
            max_iter=300,
            random_state=random_seed,
        )

        labels = model.fit_predict(matrix)
        label_sets[name] = labels

        metric_rows.append(
            {
                "representation": name,
                "n_features": int(
                    matrix.shape[1]
                ),
                "silhouette": float(
                    silhouette_score(
                        matrix,
                        labels,
                        sample_size=min(
                            2000,
                            len(matrix),
                        ),
                        random_state=random_seed,
                    )
                ),
                "inertia": float(
                    model.inertia_
                ),
            }
        )

    names = list(label_sets)
    agreement = np.zeros(
        (len(names), len(names)),
        dtype=float,
    )

    for first_index, first_name in (
        enumerate(names)
    ):
        for second_index, second_name in (
            enumerate(names)
        ):
            agreement[
                first_index,
                second_index,
            ] = adjusted_rand_score(
                label_sets[first_name],
                label_sets[second_name],
            )

    agreement_frame = pd.DataFrame(
        agreement,
        index=names,
        columns=names,
    )

    plt.figure(figsize=(7, 6))

    sns.heatmap(
        agreement_frame,
        annot=True,
        fmt=".3f",
        cmap="viridis",
        vmin=0,
        vmax=1,
        square=True,
    )

    plt.title(
        "Preprocessing sensitivity"
    )

    save_figure(
        "preprocessing_sensitivity_heatmap.png"
    )

    return (
        pd.DataFrame(metric_rows),
        agreement_frame,
    )


def run_phase3() -> None:
    ensure_directories()
    setup_logging()

    started = time.perf_counter()

    try:
        params = load_yaml(PARAMS_PATH)
        seed = int(
            params["project"]["random_seed"]
        )

        np.random.seed(seed)

        setup_plots(
            int(params["plots"]["dpi"])
        )

        logging.info("=" * 70)
        logging.info("Phase 3 started")
        logging.info("=" * 70)

        phase1 = load_phase1_arrays(params)

        (
            all_features,
            all_metadata,
        ) = load_supporting_tables(params)

        if len(all_features) != len(
            phase1["y"]
        ):
            raise ValueError(
                "Phase 1 outputs have inconsistent row counts"
            )

        sample_indices = (
            select_consensus_indices(
                phase1,
                params,
                seed,
            )
        )

        x_pca = phase1["X_pca"][
            sample_indices
        ].astype(np.float64)

        x_standard = phase1[
            "X_standard"
        ][sample_indices].astype(
            np.float64
        )

        x_robust = phase1["X_robust"][
            sample_indices
        ].astype(np.float64)

        y_true = phase1["y"][
            sample_indices
        ].astype(np.int8)

        row_ids = phase1["row_id"][
            sample_indices
        ].astype(np.int64)

        features = (
            all_features.iloc[
                sample_indices
            ]
            .reset_index(drop=True)
            .copy()
        )

        metadata = (
            all_metadata.iloc[
                sample_indices
            ]
            .reset_index(drop=True)
            .copy()
        )

        save_json(
            {
                "sample_size": len(
                    sample_indices
                ),
                "pca_dimensions": int(
                    x_pca.shape[1]
                ),
                "random_seed": seed,
                "class_used_for_sampling": False,
            },
            REPORT_DIR
            / "consensus_sample_manifest.json",
        )

        pd.DataFrame(
            {
                "array_index": sample_indices,
                "row_id": row_ids,
            }
        ).to_csv(
            REPORT_DIR
            / "consensus_sample_indices.csv",
            index=False,
        )

        (
            labels_collection,
            base_manifest,
        ) = generate_base_clusterings(
            x_pca,
            params,
        )

        base_manifest.to_csv(
            REPORT_DIR
            / "base_clustering_manifest.csv",
            index=False,
        )

        base_metrics = (
            evaluate_base_clusterings(
                x_pca,
                y_true,
                labels_collection,
                base_manifest,
                params,
                seed,
            )
        )

        base_metrics.to_csv(
            REPORT_DIR
            / "base_clustering_metrics.csv",
            index=False,
        )

        coassociation = (
            build_coassociation_matrix(
                labels_collection
            )
        )

        np.save(
            PROCESSED_DIR
            / "phase3_coassociation_matrix.npy",
            coassociation,
        )

        (
            consensus_results,
            labels_by_k,
            consensus_tree,
        ) = consensus_k_search(
            x_pca,
            coassociation,
            y_true,
            params,
            seed,
        )

        consensus_results.to_csv(
            REPORT_DIR
            / "consensus_k_search.csv",
            index=False,
        )

        selected_k = select_consensus_k(
            consensus_results
        )

        final_labels = labels_by_k[
            selected_k
        ]

        actual_clusters = int(
            len(np.unique(final_labels))
        )

        np.save(
            PROCESSED_DIR
            / "phase3_consensus_linkage.npy",
            consensus_tree,
        )

        consensus_row = (
            consensus_results[
                consensus_results[
                    "requested_k"
                ]
                == selected_k
            ]
            .iloc[0]
        )

        comparison = (
            compare_consensus_with_best_base(
                consensus_row,
                base_metrics,
            )
        )

        comparison.to_csv(
            REPORT_DIR
            / "advanced_method_comparison.csv",
            index=False,
        )

        plot_coassociation(
            coassociation,
            final_labels,
        )

        plot_consensus_selection(
            consensus_results,
            selected_k,
        )

        plot_consensus_embedding(
            x_pca,
            final_labels,
        )

        plot_advanced_comparison(
            comparison
        )

        assignments = metadata.copy()

        if "array_index" in assignments.columns:
            assignments = assignments.drop(
                columns=["array_index"]
            )

        assignments.insert(
            0,
            "array_index",
            sample_indices,
        )

        assignments[
            "Consensus_cluster"
        ] = final_labels

        assignments.to_parquet(
            PROCESSED_DIR
            / "phase3_consensus_assignments.parquet",
            index=False,
        )

        profiles, z_profiles = (
            create_cluster_profiles(
                features,
                final_labels,
            )
        )

        profiles.to_csv(
            REPORT_DIR
            / "cluster_profiles.csv",
            index=False,
        )

        z_profiles.to_csv(
            REPORT_DIR
            / "cluster_profile_zscores.csv",
            index=True,
            index_label="cluster",
        )

        domain_labels = (
            create_domain_labels(
                z_profiles,
                int(
                    params["interpretation"].get(
                        "top_features_per_cluster",
                        5,
                    )
                ),
            )
        )

        domain_labels.to_csv(
            REPORT_DIR
            / "cluster_domain_labels.csv",
            index=False,
        )

        plot_cluster_profiles(
            z_profiles
        )

        composition = fraud_composition(
            final_labels,
            y_true,
        )

        composition.to_csv(
            REPORT_DIR
            / "fraud_composition.csv",
            index=False,
        )

        plot_fraud_composition(
            composition
        )

        exemplars = extract_exemplars(
            x_pca,
            final_labels,
            sample_indices,
            row_ids,
            y_true,
        )

        exemplars.to_csv(
            REPORT_DIR
            / "cluster_exemplars.csv",
            index=False,
        )

        tree_model, tree_metrics = (
            fit_rule_extractor(
                features,
                final_labels,
                params,
                seed,
            )
        )

        save_json(
            tree_metrics,
            REPORT_DIR
            / "rule_extractor_metrics.json",
        )

        shap_status = run_optional_shap(
            tree_model,
            features,
            seed,
        )

        save_json(
            shap_status,
            REPORT_DIR
            / "shap_execution_status.json",
        )

        (
            anomaly_ranking,
            anomaly_summary,
        ) = cluster_anomaly_analysis(
            x_pca,
            final_labels,
            sample_indices,
            row_ids,
            y_true,
            params,
        )

        anomaly_ranking.to_csv(
            REPORT_DIR
            / "cluster_anomaly_ranking.csv",
            index=False,
        )

        save_json(
            anomaly_summary,
            REPORT_DIR
            / "cluster_anomaly_summary.json",
        )

        plot_anomaly_scores(
            anomaly_ranking
        )

        (
            sensitivity_metrics,
            sensitivity_agreement,
        ) = preprocessing_sensitivity(
            x_standard,
            x_robust,
            x_pca,
            selected_k,
            seed,
            params,
        )

        sensitivity_metrics.to_csv(
            REPORT_DIR
            / "preprocessing_sensitivity_metrics.csv",
            index=False,
        )

        sensitivity_agreement.to_csv(
            REPORT_DIR
            / "preprocessing_sensitivity_ari.csv",
            index=True,
            index_label="representation",
        )

        save_json(
            {
                "model_type": (
                    "Ensemble consensus clustering"
                ),
                "selected_k": selected_k,
                "actual_cluster_count": (
                    actual_clusters
                ),
                "base_clustering_count": len(
                    labels_collection
                ),
                "sample_size": len(
                    sample_indices
                ),
                "random_seed": seed,
                "fit_date_utc": datetime.now(
                    timezone.utc
                ).isoformat(),
                "class_used_for_tuning": False,
            },
            MODEL_DIR
            / "consensus_model_metadata.json",
        )

        elapsed = (
            time.perf_counter() - started
        )

        save_json(
            {
                "status": "completed",
                "elapsed_seconds": elapsed,
                "elapsed_minutes": (
                    elapsed / 60.0
                ),
                "random_seed": seed,
                "sample_size": len(
                    sample_indices
                ),
                "base_clustering_count": len(
                    labels_collection
                ),
                "selected_consensus_k": (
                    selected_k
                ),
                "actual_cluster_count": (
                    actual_clusters
                ),
            },
            REPORT_DIR
            / "execution_record.json",
        )

        failure_path = (
            REPORT_DIR
            / "execution_failed.json"
        )

        if failure_path.exists():
            failure_path.unlink()

        logging.info(
            "Phase 3 completed in %.2f minutes",
            elapsed / 60.0,
        )

    except Exception as error:
        logging.exception(
            "Phase 3 failed"
        )

        save_json(
            {
                "status": "failed",
                "error_type": (
                    type(error).__name__
                ),
                "error_message": str(error),
                "created_at_utc": (
                    datetime.now(
                        timezone.utc
                    ).isoformat()
                ),
            },
            REPORT_DIR
            / "execution_failed.json",
        )

        raise


run_phase3()