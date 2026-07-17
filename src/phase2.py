from __future__ import annotations

import json
import logging
import sys
import time
import warnings
from itertools import combinations
from pathlib import Path
from typing import Any, Callable

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from kneed import KneeLocator
from scipy.cluster.hierarchy import (
    cophenet,
    dendrogram,
    fcluster,
    linkage,
)
from scipy.spatial.distance import pdist
from sklearn.cluster import (
    AgglomerativeClustering,
    DBSCAN,
    KMeans,
)
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
    silhouette_score,
    v_measure_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors


# ---------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

PARAMS_PATH = ROOT / "params_phase2.yaml"

REPORT_DIR = ROOT / "reports" / "phase2"
FIGURE_DIR = ROOT / "reports" / "figures" / "phase2"
MODEL_DIR = ROOT / "models" / "phase2"
PROCESSED_DIR = ROOT / "data" / "processed"


# ---------------------------------------------------------------------
# General utilities
# ---------------------------------------------------------------------

def ensure_directories() -> None:
    for path in [
        REPORT_DIR,
        FIGURE_DIR,
        MODEL_DIR,
        PROCESSED_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)


def setup_logging() -> None:
    log_path = REPORT_DIR / "phase2_execution.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(
                log_path,
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
        return yaml.safe_load(file)


def save_json(data: Any, path: Path) -> None:
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
    logging.info("Figure saved: %s", path)


# ---------------------------------------------------------------------
# Loading Phase 1 output
# ---------------------------------------------------------------------

def load_phase1_data(
    params: dict[str, Any],
) -> dict[str, np.ndarray]:
    relative_path = params["data"]["input_file"]
    path = ROOT / relative_path

    if not path.exists():
        raise FileNotFoundError(
            f"""
Phase 1 output was not found:

{path}

Run Phase 1 first:

python src/phase1.py
"""
        )

    logging.info("Loading Phase 1 arrays from: %s", path)

    data = np.load(path, allow_pickle=False)

    required_arrays = {
        "X_standard",
        "X_robust",
        "X_pca",
        "y",
        "row_id",
        "train_indices",
        "test_indices",
    }

    missing = required_arrays - set(data.files)

    if missing:
        raise KeyError(
            f"Missing arrays in Phase 1 output: {sorted(missing)}"
        )

    representation = params["data"]["representation"]

    if representation not in data.files:
        raise KeyError(
            f"Requested representation '{representation}' "
            f"does not exist in {path}."
        )

    result = {
        key: data[key]
        for key in data.files
    }

    logging.info(
        "Loaded representation %s with shape %s",
        representation,
        result[representation].shape,
    )

    return result


def create_evaluation_sample(
    train_indices: np.ndarray,
    sample_size: int,
    random_seed: int,
) -> np.ndarray:
    """
    Select records without using Class labels.

    This prevents the fraud label from affecting algorithm tuning.
    """
    rng = np.random.default_rng(random_seed)

    sample_size = min(
        sample_size,
        len(train_indices),
    )

    selected = rng.choice(
        train_indices,
        size=sample_size,
        replace=False,
    )

    return np.sort(selected)


# ---------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------

def purity_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    table = pd.crosstab(
        pd.Series(y_true, name="true"),
        pd.Series(y_pred, name="cluster"),
    )

    if table.size == 0:
        return float("nan")

    return float(
        table.max(axis=0).sum() / table.to_numpy().sum()
    )


def internal_metrics(
    x: np.ndarray,
    labels: np.ndarray,
    silhouette_sample_size: int,
    random_seed: int,
) -> dict[str, float]:
    """
    Internal metrics exclude DBSCAN noise records labelled -1.
    """
    mask = labels != -1

    x_valid = x[mask]
    labels_valid = labels[mask]

    unique_clusters = np.unique(labels_valid)

    if (
        len(x_valid) < 3
        or len(unique_clusters) < 2
        or len(unique_clusters) >= len(x_valid)
    ):
        return {
            "silhouette": float("nan"),
            "davies_bouldin": float("nan"),
            "calinski_harabasz": float("nan"),
        }

    sample_size = min(
        silhouette_sample_size,
        len(x_valid),
    )

    silhouette = silhouette_score(
        x_valid,
        labels_valid,
        metric="euclidean",
        sample_size=sample_size,
        random_state=random_seed,
    )

    db = davies_bouldin_score(
        x_valid,
        labels_valid,
    )

    ch = calinski_harabasz_score(
        x_valid,
        labels_valid,
    )

    return {
        "silhouette": float(silhouette),
        "davies_bouldin": float(db),
        "calinski_harabasz": float(ch),
    }


def external_metrics(
    y_true: np.ndarray,
    labels: np.ndarray,
) -> dict[str, float]:
    """
    These metrics are calculated only after clustering.

    They are not used for selecting k or hyperparameters.
    """
    return {
        "ari": float(
            adjusted_rand_score(y_true, labels)
        ),
        "nmi": float(
            normalized_mutual_info_score(y_true, labels)
        ),
        "ami": float(
            adjusted_mutual_info_score(y_true, labels)
        ),
        "fowlkes_mallows": float(
            fowlkes_mallows_score(y_true, labels)
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
        "purity": purity_score(y_true, labels),
    }


def evaluate_clustering(
    algorithm: str,
    parameters: str,
    x: np.ndarray,
    labels: np.ndarray,
    y_true: np.ndarray,
    runtime_seconds: float,
    silhouette_sample_size: int,
    random_seed: int,
) -> dict[str, Any]:
    internal = internal_metrics(
        x=x,
        labels=labels,
        silhouette_sample_size=silhouette_sample_size,
        random_seed=random_seed,
    )

    external = external_metrics(
        y_true=y_true,
        labels=labels,
    )

    non_noise = labels[labels != -1]

    number_of_clusters = (
        len(np.unique(non_noise))
        if len(non_noise) > 0
        else 0
    )

    noise_fraction = float(
        np.mean(labels == -1)
    )

    return {
        "algorithm": algorithm,
        "parameters": parameters,
        "n_records": int(len(x)),
        "n_clusters": int(number_of_clusters),
        "noise_fraction": noise_fraction,
        "runtime_seconds": float(runtime_seconds),
        **internal,
        **external,
    }


# ---------------------------------------------------------------------
# K-Means
# ---------------------------------------------------------------------

def run_kmeans_search(
    x: np.ndarray,
    y: np.ndarray,
    k_values: list[int],
    params: dict[str, Any],
    random_seed: int,
) -> tuple[pd.DataFrame, dict[int, np.ndarray]]:
    rows = []
    labels_by_k: dict[int, np.ndarray] = {}

    for k in k_values:
        logging.info("K-Means search: k=%d", k)

        model = KMeans(
            n_clusters=k,
            init="k-means++",
            n_init=int(params["kmeans"]["n_init"]),
            max_iter=int(params["kmeans"]["max_iter"]),
            random_state=random_seed,
        )

        start = time.perf_counter()
        labels = model.fit_predict(x)
        runtime = time.perf_counter() - start

        labels_by_k[k] = labels

        row = evaluate_clustering(
            algorithm="KMeans",
            parameters=f"k={k}",
            x=x,
            labels=labels,
            y_true=y,
            runtime_seconds=runtime,
            silhouette_sample_size=int(
                params["sampling"]["silhouette_sample_size"]
            ),
            random_seed=random_seed,
        )

        row["k"] = k
        row["inertia"] = float(model.inertia_)
        row["iterations"] = int(model.n_iter_)

        rows.append(row)

    return pd.DataFrame(rows), labels_by_k


def determine_elbow(
    kmeans_results: pd.DataFrame,
) -> int:
    x = kmeans_results["k"].to_numpy()
    y = kmeans_results["inertia"].to_numpy()

    locator = KneeLocator(
        x,
        y,
        curve="convex",
        direction="decreasing",
    )

    if locator.knee is not None:
        return int(locator.knee)

    if len(y) >= 3:
        second_difference = np.diff(y, n=2)
        return int(
            x[np.argmax(np.abs(second_difference)) + 1]
        )

    return int(x[0])


def plot_kmeans_selection(
    results: pd.DataFrame,
    elbow_k: int,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    axes[0].plot(
        results["k"],
        results["inertia"],
        marker="o",
    )
    axes[0].axvline(
        elbow_k,
        color="red",
        linestyle="--",
        label=f"Kneedle k={elbow_k}",
    )
    axes[0].set_title("K-Means elbow method")
    axes[0].set_xlabel("k")
    axes[0].set_ylabel("Inertia")
    axes[0].legend()

    axes[1].plot(
        results["k"],
        results["silhouette"],
        marker="o",
        color="#2ca02c",
    )
    axes[1].set_title("Average silhouette")
    axes[1].set_xlabel("k")
    axes[1].set_ylabel("Silhouette")

    axes[2].plot(
        results["k"],
        results["davies_bouldin"],
        marker="o",
        color="#ff7f0e",
    )
    axes[2].set_title("Davies–Bouldin index")
    axes[2].set_xlabel("k")
    axes[2].set_ylabel("Lower is better")

    save_figure("kmeans_k_selection.png")


def kmeans_iteration_curve(
    x: np.ndarray,
    selected_k: int,
    random_seed: int,
) -> pd.DataFrame:
    rows = []

    budgets = list(range(1, 21)) + [30, 50, 100]

    for max_iter in budgets:
        model = KMeans(
            n_clusters=selected_k,
            init="k-means++",
            n_init=1,
            max_iter=max_iter,
            random_state=random_seed,
        )

        model.fit(x)

        rows.append(
            {
                "iteration_budget": max_iter,
                "actual_iterations": int(model.n_iter_),
                "inertia": float(model.inertia_),
            }
        )

    result = pd.DataFrame(rows)

    plt.figure(figsize=(8, 5))
    plt.plot(
        result["iteration_budget"],
        result["inertia"],
        marker="o",
    )
    plt.xlabel("Maximum iteration budget")
    plt.ylabel("Final inertia")
    plt.title("K-Means convergence diagnostic")
    save_figure("kmeans_convergence_curve.png")

    return result


# ---------------------------------------------------------------------
# Gap statistic
# ---------------------------------------------------------------------

def calculate_gap_statistic(
    x: np.ndarray,
    k_values: list[int],
    reference_count: int,
    random_seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_seed)

    minimum = x.min(axis=0)
    maximum = x.max(axis=0)

    rows = []

    for k in k_values:
        logging.info("Gap statistic: k=%d", k)

        real_model = KMeans(
            n_clusters=k,
            n_init=10,
            random_state=random_seed,
        )
        real_model.fit(x)

        real_log_dispersion = np.log(
            max(real_model.inertia_, np.finfo(float).tiny)
        )

        reference_logs = []

        for reference_id in range(reference_count):
            reference = rng.uniform(
                minimum,
                maximum,
                size=x.shape,
            )

            reference_model = KMeans(
                n_clusters=k,
                n_init=5,
                random_state=(
                    random_seed + reference_id
                ),
            )
            reference_model.fit(reference)

            reference_logs.append(
                np.log(
                    max(
                        reference_model.inertia_,
                        np.finfo(float).tiny,
                    )
                )
            )

        reference_logs = np.asarray(reference_logs)

        gap = float(
            reference_logs.mean() - real_log_dispersion
        )

        standard_error = float(
            reference_logs.std(ddof=1)
            * np.sqrt(1 + 1 / reference_count)
        )

        rows.append(
            {
                "k": k,
                "gap": gap,
                "standard_error": standard_error,
                "real_log_dispersion": (
                    real_log_dispersion
                ),
            }
        )

    return pd.DataFrame(rows)


def select_gap_k(
    gap_results: pd.DataFrame,
) -> int:
    """
    Tibshirani rule:
    choose smallest k where Gap(k) >= Gap(k+1) - s(k+1).
    """
    for index in range(len(gap_results) - 1):
        current = gap_results.iloc[index]
        following = gap_results.iloc[index + 1]

        if (
            current["gap"]
            >= following["gap"] - following["standard_error"]
        ):
            return int(current["k"])

    best_index = gap_results["gap"].idxmax()
    return int(gap_results.loc[best_index, "k"])


def plot_gap_statistic(
    results: pd.DataFrame,
    selected_k: int,
) -> None:
    plt.figure(figsize=(8, 5))

    plt.errorbar(
        results["k"],
        results["gap"],
        yerr=results["standard_error"],
        marker="o",
        capsize=4,
    )

    plt.axvline(
        selected_k,
        color="red",
        linestyle="--",
        label=f"Selected k={selected_k}",
    )

    plt.xlabel("k")
    plt.ylabel("Gap statistic")
    plt.title("Gap-statistic cluster selection")
    plt.legend()

    save_figure("gap_statistic.png")


# ---------------------------------------------------------------------
# Hierarchical clustering
# ---------------------------------------------------------------------

def hierarchical_diagnostics(
    x: np.ndarray,
    k_values: list[int],
    linkages: list[str],
    params: dict[str, Any],
    random_seed: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows = []
    linkage_matrices: dict[str, Any] = {}

    pairwise_distances = pdist(
        x,
        metric="euclidean",
    )

    for method in linkages:
        logging.info(
            "Hierarchical diagnostics: linkage=%s",
            method,
        )

        start = time.perf_counter()

        matrix = linkage(
            x,
            method=method,
            metric="euclidean",
        )

        runtime = time.perf_counter() - start

        linkage_matrices[method] = matrix

        coefficient, _ = cophenet(
            matrix,
            pairwise_distances,
        )

        for k in k_values:
            labels = fcluster(
                matrix,
                t=k,
                criterion="maxclust",
            ) - 1

            metrics = internal_metrics(
                x=x,
                labels=labels,
                silhouette_sample_size=int(
                    params["sampling"][
                        "silhouette_sample_size"
                    ]
                ),
                random_seed=random_seed,
            )

            rows.append(
                {
                    "linkage": method,
                    "k": k,
                    "cophenetic_correlation": float(
                        coefficient
                    ),
                    "runtime_seconds": float(runtime),
                    **metrics,
                }
            )

    return pd.DataFrame(rows), linkage_matrices


def plot_dendrograms(
    linkage_matrices: dict[str, Any],
) -> None:
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(16, 10),
    )

    axes = axes.ravel()

    for axis, (method, matrix) in zip(
        axes,
        linkage_matrices.items(),
    ):
        dendrogram(
            matrix,
            truncate_mode="lastp",
            p=30,
            no_labels=True,
            ax=axis,
            color_threshold=None,
        )

        axis.set_title(
            f"{method.capitalize()} linkage"
        )
        axis.set_xlabel("Merged groups")
        axis.set_ylabel("Linkage distance")

    save_figure("hierarchical_dendrograms.png")


def plot_hierarchical_metrics(
    results: pd.DataFrame,
) -> None:
    plt.figure(figsize=(9, 6))

    sns.lineplot(
        data=results,
        x="k",
        y="silhouette",
        hue="linkage",
        marker="o",
    )

    plt.title(
        "Hierarchical clustering: linkage comparison"
    )
    plt.ylabel("Silhouette")
    save_figure("hierarchical_linkage_comparison.png")


# ---------------------------------------------------------------------
# DBSCAN
# ---------------------------------------------------------------------

def dbscan_parameter_search(
    x: np.ndarray,
    y: np.ndarray,
    params: dict[str, Any],
    random_seed: int,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    rows = []
    labels_by_configuration = {}

    min_samples_values = [
        int(value)
        for value in params["dbscan"]["min_samples"]
    ]

    quantiles = [
        float(value)
        for value in params["dbscan"]["eps_quantiles"]
    ]

    for min_samples in min_samples_values:
        logging.info(
            "Computing k-distance for min_samples=%d",
            min_samples,
        )

        neighbours = NearestNeighbors(
            n_neighbors=min_samples,
            metric="euclidean",
            n_jobs=-1,
        )
        neighbours.fit(x)

        distances, _ = neighbours.kneighbors(x)

        k_distances = np.sort(
            distances[:, -1]
        )

        k_distance_table = pd.DataFrame(
            {
                "sorted_position": np.arange(
                    len(k_distances)
                ),
                "k_distance": k_distances,
                "min_samples": min_samples,
            }
        )

        k_distance_table.to_csv(
            REPORT_DIR
            / f"dbscan_k_distance_{min_samples}.csv",
            index=False,
        )

        plt.figure(figsize=(8, 5))
        plt.plot(k_distances)
        plt.xlabel("Sorted observations")
        plt.ylabel(
            f"Distance to neighbour {min_samples}"
        )
        plt.title(
            f"DBSCAN k-distance plot: "
            f"min_samples={min_samples}"
        )
        save_figure(
            f"dbscan_k_distance_{min_samples}.png"
        )

        eps_values = np.unique(
            np.quantile(k_distances, quantiles)
        )

        for eps in eps_values:
            configuration = (
                f"eps={eps:.6f},min_samples={min_samples}"
            )

            logging.info(
                "DBSCAN: %s",
                configuration,
            )

            model = DBSCAN(
                eps=float(eps),
                min_samples=min_samples,
                metric="euclidean",
                n_jobs=-1,
            )

            start = time.perf_counter()
            labels = model.fit_predict(x)
            runtime = time.perf_counter() - start

            labels_by_configuration[
                configuration
            ] = labels

            row = evaluate_clustering(
                algorithm="DBSCAN",
                parameters=configuration,
                x=x,
                labels=labels,
                y_true=y,
                runtime_seconds=runtime,
                silhouette_sample_size=int(
                    params["sampling"][
                        "silhouette_sample_size"
                    ]
                ),
                random_seed=random_seed,
            )

            row["eps"] = float(eps)
            row["min_samples"] = min_samples

            rows.append(row)

    return (
        pd.DataFrame(rows),
        labels_by_configuration,
    )


def choose_dbscan_configuration(
    results: pd.DataFrame,
) -> pd.Series:
    valid = results[
        (results["n_clusters"] >= 2)
        & (results["noise_fraction"] < 0.95)
        & results["silhouette"].notna()
    ].copy()

    if len(valid) > 0:
        valid = valid.sort_values(
            ["silhouette", "noise_fraction"],
            ascending=[False, True],
        )
        return valid.iloc[0]

    logging.warning(
        "No DBSCAN configuration produced at least "
        "two valid non-noise clusters."
    )

    fallback = results.sort_values(
        ["n_clusters", "noise_fraction"],
        ascending=[False, True],
    )

    return fallback.iloc[0]


# ---------------------------------------------------------------------
# Gaussian Mixture Model
# ---------------------------------------------------------------------

def run_gmm_search(
    x: np.ndarray,
    y: np.ndarray,
    k_values: list[int],
    params: dict[str, Any],
    random_seed: int,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    rows = []
    labels_by_configuration = {}

    covariance_types = params["gmm"][
        "covariance_types"
    ]

    for covariance_type in covariance_types:
        for k in k_values:
            configuration = (
                f"k={k},covariance={covariance_type}"
            )

            logging.info("GMM: %s", configuration)

            model = GaussianMixture(
                n_components=k,
                covariance_type=covariance_type,
                n_init=int(params["gmm"]["n_init"]),
                max_iter=int(params["gmm"]["max_iter"]),
                reg_covar=1e-6,
                random_state=random_seed,
            )

            start = time.perf_counter()

            with warnings.catch_warnings():
                warnings.simplefilter(
                    "ignore",
                    ConvergenceWarning,
                )
                labels = model.fit_predict(x)

            runtime = time.perf_counter() - start

            labels_by_configuration[
                configuration
            ] = labels

            row = evaluate_clustering(
                algorithm="GMM",
                parameters=configuration,
                x=x,
                labels=labels,
                y_true=y,
                runtime_seconds=runtime,
                silhouette_sample_size=int(
                    params["sampling"][
                        "silhouette_sample_size"
                    ]
                ),
                random_seed=random_seed,
            )

            row["k"] = k
            row["covariance_type"] = covariance_type
            row["bic"] = float(model.bic(x))
            row["aic"] = float(model.aic(x))
            row["lower_bound"] = float(
                model.lower_bound_
            )
            row["converged"] = bool(model.converged_)
            row["iterations"] = int(model.n_iter_)

            rows.append(row)

    return (
        pd.DataFrame(rows),
        labels_by_configuration,
    )


def plot_gmm_selection(
    results: pd.DataFrame,
) -> None:
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(14, 5),
    )

    sns.lineplot(
        data=results,
        x="k",
        y="bic",
        hue="covariance_type",
        marker="o",
        ax=axes[0],
    )
    axes[0].set_title("GMM model selection by BIC")
    axes[0].set_ylabel("BIC — lower is better")

    sns.lineplot(
        data=results,
        x="k",
        y="aic",
        hue="covariance_type",
        marker="o",
        ax=axes[1],
    )
    axes[1].set_title("GMM model selection by AIC")
    axes[1].set_ylabel("AIC — lower is better")

    save_figure("gmm_bic_aic.png")


def gmm_convergence_curve(
    x: np.ndarray,
    selected_k: int,
    covariance_type: str,
    random_seed: int,
    iterations: int = 30,
) -> pd.DataFrame:
    model = GaussianMixture(
        n_components=selected_k,
        covariance_type=covariance_type,
        n_init=1,
        max_iter=1,
        warm_start=True,
        reg_covar=1e-6,
        random_state=random_seed,
    )

    rows = []

    with warnings.catch_warnings():
        warnings.simplefilter(
            "ignore",
            ConvergenceWarning,
        )

        for iteration in range(1, iterations + 1):
            model.fit(x)

            rows.append(
                {
                    "em_iteration": iteration,
                    "lower_bound": float(
                        model.lower_bound_
                    ),
                }
            )

    result = pd.DataFrame(rows)

    plt.figure(figsize=(8, 5))
    plt.plot(
        result["em_iteration"],
        result["lower_bound"],
        marker="o",
    )
    plt.xlabel("EM iteration")
    plt.ylabel("Variational lower bound")
    plt.title("GMM log-likelihood convergence")
    save_figure("gmm_convergence_curve.png")

    return result


# ---------------------------------------------------------------------
# Seed and bootstrap stability
# ---------------------------------------------------------------------

def pairwise_ari_distribution(
    label_runs: list[np.ndarray],
) -> np.ndarray:
    scores = []

    for first, second in combinations(
        range(len(label_runs)),
        2,
    ):
        scores.append(
            adjusted_rand_score(
                label_runs[first],
                label_runs[second],
            )
        )

    return np.asarray(scores, dtype=float)


def seed_stability(
    x: np.ndarray,
    algorithm_name: str,
    model_factory: Callable[[int], Any],
    number_of_runs: int,
    base_seed: int,
) -> tuple[pd.DataFrame, list[np.ndarray]]:
    labels_runs = []

    for run in range(number_of_runs):
        seed = base_seed + run

        logging.info(
            "%s seed-stability run %d/%d",
            algorithm_name,
            run + 1,
            number_of_runs,
        )

        model = model_factory(seed)

        if hasattr(model, "fit_predict"):
            labels = model.fit_predict(x)
        else:
            model.fit(x)
            labels = model.predict(x)

        labels_runs.append(labels)

    scores = pairwise_ari_distribution(
        labels_runs
    )

    result = pd.DataFrame(
        {
            "algorithm": algorithm_name,
            "pair_id": np.arange(len(scores)),
            "pairwise_ari": scores,
        }
    )

    return result, labels_runs


def bootstrap_stability(
    x: np.ndarray,
    anchor_x: np.ndarray,
    algorithm_name: str,
    model_factory: Callable[[int], Any],
    number_of_runs: int,
    base_seed: int,
) -> pd.DataFrame:
    """
    Fit each model on a bootstrap sample and predict cluster labels
    for the same fixed anchor records. Pairwise ARI then measures
    bootstrap stability.
    """
    predictions = []
    n = len(x)

    for run in range(number_of_runs):
        seed = base_seed + 1000 + run
        rng = np.random.default_rng(seed)

        bootstrap_indices = rng.choice(
            n,
            size=n,
            replace=True,
        )

        model = model_factory(seed)
        model.fit(x[bootstrap_indices])

        if not hasattr(model, "predict"):
            raise TypeError(
                f"{algorithm_name} does not support prediction."
            )

        predictions.append(
            model.predict(anchor_x)
        )

    scores = pairwise_ari_distribution(
        predictions
    )

    return pd.DataFrame(
        {
            "algorithm": algorithm_name,
            "pair_id": np.arange(len(scores)),
            "pairwise_ari": scores,
        }
    )


def plot_stability(
    seed_results: pd.DataFrame,
    bootstrap_results: pd.DataFrame,
) -> None:
    seed_plot = seed_results.copy()
    seed_plot["stability_type"] = "seed"

    bootstrap_plot = bootstrap_results.copy()
    bootstrap_plot["stability_type"] = "bootstrap"

    combined = pd.concat(
        [seed_plot, bootstrap_plot],
        ignore_index=True,
    )

    plt.figure(figsize=(10, 6))

    sns.boxplot(
        data=combined,
        x="algorithm",
        y="pairwise_ari",
        hue="stability_type",
    )

    plt.ylim(-0.05, 1.05)
    plt.title(
        "Seed and bootstrap stability"
    )
    plt.ylabel("Pairwise ARI between runs")

    save_figure("stability_comparison.png")


# ---------------------------------------------------------------------
# Agreement between final algorithms
# ---------------------------------------------------------------------

def algorithm_agreement_matrix(
    final_labels: dict[str, np.ndarray],
) -> pd.DataFrame:
    names = list(final_labels.keys())

    matrix = np.zeros(
        (len(names), len(names)),
        dtype=float,
    )

    for i, first_name in enumerate(names):
        for j, second_name in enumerate(names):
            matrix[i, j] = adjusted_rand_score(
                final_labels[first_name],
                final_labels[second_name],
            )

    return pd.DataFrame(
        matrix,
        index=names,
        columns=names,
    )


def plot_agreement_matrix(
    matrix: pd.DataFrame,
) -> None:
    plt.figure(figsize=(8, 7))

    sns.heatmap(
        matrix,
        annot=True,
        fmt=".3f",
        cmap="viridis",
        vmin=0,
        vmax=1,
        square=True,
    )

    plt.title(
        "Pairwise algorithm agreement—Adjusted Rand Index"
    )

    save_figure("algorithm_agreement_heatmap.png")


# ---------------------------------------------------------------------
# Final comparison and summary
# ---------------------------------------------------------------------

def plot_final_metric_comparison(
    final_results: pd.DataFrame,
) -> None:
    metrics = [
        "silhouette",
        "davies_bouldin",
        "calinski_harabasz",
        "ari",
        "nmi",
        "ami",
    ]

    melted = final_results.melt(
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
        kind="bar",
        col_wrap=3,
        sharey=False,
        height=4,
        aspect=1.2,
    )

    graph.set_xticklabels(rotation=25)
    graph.fig.suptitle(
        "Final algorithm comparison",
        y=1.02,
    )

    path = FIGURE_DIR / "final_metric_comparison.png"
    graph.savefig(path, bbox_inches="tight")
    plt.close("all")


def write_phase2_summary(
    elbow_k: int,
    silhouette_k: int,
    gap_k: int,
    hierarchical_method: str,
    hierarchical_k: int,
    dbscan_parameters: str,
    gmm_k: int,
    gmm_covariance: str,
    final_results: pd.DataFrame,
) -> None:
    display_columns = [
        "algorithm",
        "parameters",
        "n_clusters",
        "noise_fraction",
        "silhouette",
        "davies_bouldin",
        "calinski_harabasz",
        "ari",
        "nmi",
        "ami",
        "purity",
        "runtime_seconds",
    ]

    table = final_results[
        display_columns
    ].round(5).to_markdown(index=False)

    summary = f"""# Phase 2 Execution Summary

## Experimental policy

All hyperparameter selection was performed using internal metrics
only. The `Class` fraud label was used after fitting solely to compute
external evaluation metrics.

The evaluation subset was selected randomly from the Phase 1 training
indices without consulting `Class`.

## Determining the number of clusters

### K-Means

- Elbow/Kneedle recommendation: `{elbow_k}`
- Maximum-silhouette recommendation: `{silhouette_k}`
- Gap-statistic recommendation: `{gap_k}`

Disagreement between these methods is not automatically an error.
Each method measures a different aspect of cluster structure.

### Hierarchical clustering

- Selected linkage: `{hierarchical_method}`
- Selected k: `{hierarchical_k}`

Single, complete, average, and Ward linkages were compared using
silhouette and cophenetic correlation.

### DBSCAN

Selected configuration:

`{dbscan_parameters}`

DBSCAN does not require a predefined k. Its `eps` candidates were
derived from k-nearest-neighbour distance quantiles.

### Gaussian Mixture Model

- BIC-selected components: `{gmm_k}`
- BIC-selected covariance structure: `{gmm_covariance}`

## Final algorithm comparison

{table}

## Interpretation cautions

1. Fraud labels are highly imbalanced, so purity can appear high even
   for uninformative clusterings.
2. ARI, AMI, NMI, homogeneity, completeness, and fraud composition
   must be interpreted together.
3. DBSCAN noise is represented by label `-1`; noise is excluded from
   internal geometric metrics but retained in external metrics.
4. The algorithm recommendation must combine internal quality,
   external agreement, stability, runtime, and domain interpretation.
5. A low external score does not necessarily invalidate transaction
   segmentation because fraud may not form a single isolated cluster.

## Generated reports

- `kmeans_search.csv`
- `gap_statistic.csv`
- `hierarchical_search.csv`
- `dbscan_search.csv`
- `gmm_search.csv`
- `seed_stability.csv`
- `bootstrap_stability.csv`
- `algorithm_agreement.csv`
- `final_comparison.csv`
- `final_cluster_assignments.parquet`

## Generated figures

See `reports/figures/phase2/`.
"""

    path = REPORT_DIR / "phase2_execution_summary.md"
    path.write_text(
        summary,
        encoding="utf-8",
    )


# ---------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------

def run_phase2() -> None:
    ensure_directories()
    setup_logging()

    start_time = time.perf_counter()

    try:
        params = load_yaml(PARAMS_PATH)

        seed = int(
            params["project"]["random_seed"]
        )

        setup_plots(
            int(params["plots"]["dpi"])
        )

        np.random.seed(seed)

        logging.info("=" * 65)
        logging.info("Phase 2 pipeline started")
        logging.info("Project root: %s", ROOT)
        logging.info("=" * 65)

        # -------------------------------------------------------------
        # Load Phase 1 data
        # -------------------------------------------------------------
        phase1 = load_phase1_data(params)

        representation = params["data"][
            "representation"
        ]

        x_all = phase1[representation].astype(
            np.float64,
            copy=False,
        )
        y_all = phase1["y"].astype(np.int8)
        row_id_all = phase1["row_id"]
        train_indices = phase1["train_indices"]

        evaluation_indices = create_evaluation_sample(
            train_indices=train_indices,
            sample_size=int(
                params["sampling"]["evaluation_size"]
            ),
            random_seed=seed,
        )

        x = x_all[evaluation_indices]
        y = y_all[evaluation_indices]
        row_ids = row_id_all[evaluation_indices]

        pd.DataFrame(
            {
                "array_index": evaluation_indices,
                "row_id": row_ids,
            }
        ).to_csv(
            REPORT_DIR / "evaluation_sample_indices.csv",
            index=False,
        )

        sample_manifest = {
            "selection_policy": (
                "Random selection from Phase 1 training indices "
                "without using Class labels."
            ),
            "representation": representation,
            "sample_size": int(len(x)),
            "feature_count": int(x.shape[1]),
            "seed": seed,
            "class_used_for_sampling": False,
        }

        save_json(
            sample_manifest,
            REPORT_DIR / "sample_manifest.json",
        )

        k_min = int(
            params["cluster_search"]["k_min"]
        )
        k_max = int(
            params["cluster_search"]["k_max"]
        )
        k_values = list(
            range(k_min, k_max + 1)
        )

        silhouette_sample_size = int(
            params["sampling"][
                "silhouette_sample_size"
            ]
        )

        # -------------------------------------------------------------
        # K-Means search
        # -------------------------------------------------------------
        kmeans_results, kmeans_labels = (
            run_kmeans_search(
                x=x,
                y=y,
                k_values=k_values,
                params=params,
                random_seed=seed,
            )
        )

        kmeans_results.to_csv(
            REPORT_DIR / "kmeans_search.csv",
            index=False,
        )

        elbow_k = determine_elbow(
            kmeans_results
        )

        silhouette_k = int(
            kmeans_results.loc[
                kmeans_results[
                    "silhouette"
                ].idxmax(),
                "k",
            ]
        )

        plot_kmeans_selection(
            kmeans_results,
            elbow_k,
        )

        convergence = kmeans_iteration_curve(
            x=x,
            selected_k=silhouette_k,
            random_seed=seed,
        )

        convergence.to_csv(
            REPORT_DIR
            / "kmeans_convergence_curve.csv",
            index=False,
        )

        # -------------------------------------------------------------
        # Gap statistic
        # -------------------------------------------------------------
        gap_results = calculate_gap_statistic(
            x=x,
            k_values=k_values,
            reference_count=int(
                params["cluster_search"][
                    "gap_reference_datasets"
                ]
            ),
            random_seed=seed,
        )

        gap_results.to_csv(
            REPORT_DIR / "gap_statistic.csv",
            index=False,
        )

        gap_k = select_gap_k(gap_results)

        plot_gap_statistic(
            gap_results,
            selected_k=gap_k,
        )

        # -------------------------------------------------------------
        # Hierarchical diagnostics
        # -------------------------------------------------------------
        rng = np.random.default_rng(seed)

        hierarchy_size = min(
            int(
                params["sampling"][
                    "hierarchy_dendrogram_size"
                ]
            ),
            len(x),
        )

        hierarchy_positions = rng.choice(
            len(x),
            size=hierarchy_size,
            replace=False,
        )

        x_hierarchy = x[hierarchy_positions]

        hierarchy_results, linkage_matrices = (
            hierarchical_diagnostics(
                x=x_hierarchy,
                k_values=k_values,
                linkages=params["hierarchical"][
                    "linkages"
                ],
                params=params,
                random_seed=seed,
            )
        )

        hierarchy_results.to_csv(
            REPORT_DIR / "hierarchical_search.csv",
            index=False,
        )

        plot_dendrograms(linkage_matrices)
        plot_hierarchical_metrics(
            hierarchy_results
        )

        best_hierarchy_row = hierarchy_results.loc[
            hierarchy_results[
                "silhouette"
            ].idxmax()
        ]

        selected_linkage = str(
            best_hierarchy_row["linkage"]
        )
        selected_hierarchy_k = int(
            best_hierarchy_row["k"]
        )

        hierarchy_model = AgglomerativeClustering(
            n_clusters=selected_hierarchy_k,
            linkage=selected_linkage,
            metric=(
                "euclidean"
                if selected_linkage != "ward"
                else "euclidean"
            ),
        )

        hierarchy_start = time.perf_counter()
        hierarchy_final_labels = (
            hierarchy_model.fit_predict(x)
        )
        hierarchy_runtime = (
            time.perf_counter()
            - hierarchy_start
        )

        # -------------------------------------------------------------
        # DBSCAN
        # -------------------------------------------------------------
        dbscan_results, dbscan_labels = (
            dbscan_parameter_search(
                x=x,
                y=y,
                params=params,
                random_seed=seed,
            )
        )

        dbscan_results.to_csv(
            REPORT_DIR / "dbscan_search.csv",
            index=False,
        )

        selected_dbscan_row = (
            choose_dbscan_configuration(
                dbscan_results
            )
        )

        selected_dbscan_parameters = str(
            selected_dbscan_row["parameters"]
        )

        dbscan_final_labels = dbscan_labels[
            selected_dbscan_parameters
        ]

        # -------------------------------------------------------------
        # Gaussian Mixture Model
        # -------------------------------------------------------------
        gmm_results, gmm_labels = run_gmm_search(
            x=x,
            y=y,
            k_values=k_values,
            params=params,
            random_seed=seed,
        )

        gmm_results.to_csv(
            REPORT_DIR / "gmm_search.csv",
            index=False,
        )

        plot_gmm_selection(gmm_results)

        selected_gmm_row = gmm_results.loc[
            gmm_results["bic"].idxmin()
        ]

        selected_gmm_k = int(
            selected_gmm_row["k"]
        )
        selected_covariance = str(
            selected_gmm_row["covariance_type"]
        )

        selected_gmm_configuration = (
            f"k={selected_gmm_k},"
            f"covariance={selected_covariance}"
        )

        gmm_final_labels = gmm_labels[
            selected_gmm_configuration
        ]

        gmm_curve = gmm_convergence_curve(
            x=x,
            selected_k=selected_gmm_k,
            covariance_type=selected_covariance,
            random_seed=seed,
        )

        gmm_curve.to_csv(
            REPORT_DIR
            / "gmm_convergence_curve.csv",
            index=False,
        )

        # -------------------------------------------------------------
        # Final K-Means and GMM models
        # -------------------------------------------------------------
        final_kmeans = KMeans(
            n_clusters=silhouette_k,
            init="k-means++",
            n_init=int(params["kmeans"]["n_init"]),
            max_iter=int(
                params["kmeans"]["max_iter"]
            ),
            random_state=seed,
        )

        start = time.perf_counter()
        kmeans_final_labels = (
            final_kmeans.fit_predict(x)
        )
        kmeans_runtime = (
            time.perf_counter() - start
        )

        final_gmm = GaussianMixture(
            n_components=selected_gmm_k,
            covariance_type=selected_covariance,
            n_init=int(params["gmm"]["n_init"]),
            max_iter=int(params["gmm"]["max_iter"]),
            reg_covar=1e-6,
            random_state=seed,
        )

        start = time.perf_counter()

        with warnings.catch_warnings():
            warnings.simplefilter(
                "ignore",
                ConvergenceWarning,
            )
            final_gmm.fit(x)
            gmm_final_labels = final_gmm.predict(x)

        gmm_runtime = time.perf_counter() - start

        joblib.dump(
            final_kmeans,
            MODEL_DIR / "final_kmeans.joblib",
        )

        joblib.dump(
            final_gmm,
            MODEL_DIR / "final_gmm.joblib",
        )

        # -------------------------------------------------------------
        # Seed stability
        # -------------------------------------------------------------
        seed_runs = int(
            params["stability"]["seed_runs"]
        )

        kmeans_factory = lambda model_seed: KMeans(
            n_clusters=silhouette_k,
            n_init=1,
            max_iter=int(
                params["kmeans"]["max_iter"]
            ),
            random_state=model_seed,
        )

        gmm_factory = lambda model_seed: GaussianMixture(
            n_components=selected_gmm_k,
            covariance_type=selected_covariance,
            n_init=1,
            max_iter=int(
                params["gmm"]["max_iter"]
            ),
            reg_covar=1e-6,
            random_state=model_seed,
        )

        kmeans_seed_stability, _ = seed_stability(
            x=x,
            algorithm_name="KMeans",
            model_factory=kmeans_factory,
            number_of_runs=seed_runs,
            base_seed=seed,
        )

        gmm_seed_stability, _ = seed_stability(
            x=x,
            algorithm_name="GMM",
            model_factory=gmm_factory,
            number_of_runs=seed_runs,
            base_seed=seed,
        )

        seed_stability_results = pd.concat(
            [
                kmeans_seed_stability,
                gmm_seed_stability,
            ],
            ignore_index=True,
        )

        seed_stability_results.to_csv(
            REPORT_DIR / "seed_stability.csv",
            index=False,
        )

        # -------------------------------------------------------------
        # Bootstrap stability
        # -------------------------------------------------------------
        anchor_size = min(
            int(
                params["sampling"][
                    "stability_anchor_size"
                ]
            ),
            len(x),
        )

        anchor_positions = rng.choice(
            len(x),
            size=anchor_size,
            replace=False,
        )

        anchor_x = x[anchor_positions]

        bootstrap_runs = int(
            params["stability"]["bootstrap_runs"]
        )

        kmeans_bootstrap = bootstrap_stability(
            x=x,
            anchor_x=anchor_x,
            algorithm_name="KMeans",
            model_factory=kmeans_factory,
            number_of_runs=bootstrap_runs,
            base_seed=seed,
        )

        gmm_bootstrap = bootstrap_stability(
            x=x,
            anchor_x=anchor_x,
            algorithm_name="GMM",
            model_factory=gmm_factory,
            number_of_runs=bootstrap_runs,
            base_seed=seed,
        )

        bootstrap_results = pd.concat(
            [
                kmeans_bootstrap,
                gmm_bootstrap,
            ],
            ignore_index=True,
        )

        bootstrap_results.to_csv(
            REPORT_DIR / "bootstrap_stability.csv",
            index=False,
        )

        plot_stability(
            seed_results=seed_stability_results,
            bootstrap_results=bootstrap_results,
        )

        # -------------------------------------------------------------
        # Final labels and agreement
        # -------------------------------------------------------------
        final_labels = {
            "KMeans": kmeans_final_labels,
            "Hierarchical": hierarchy_final_labels,
            "DBSCAN": dbscan_final_labels,
            "GMM": gmm_final_labels,
        }

        agreement = algorithm_agreement_matrix(
            final_labels
        )

        agreement.to_csv(
            REPORT_DIR / "algorithm_agreement.csv"
        )

        plot_agreement_matrix(agreement)

        assignments = pd.DataFrame(
            {
                "array_index": evaluation_indices,
                "row_id": row_ids,
                "Class_external_only": y,
                "KMeans_cluster": (
                    kmeans_final_labels
                ),
                "Hierarchical_cluster": (
                    hierarchy_final_labels
                ),
                "DBSCAN_cluster": (
                    dbscan_final_labels
                ),
                "GMM_cluster": gmm_final_labels,
            }
        )

        assignments.to_parquet(
            PROCESSED_DIR
            / "phase2_cluster_assignments.parquet",
            index=False,
        )

        # -------------------------------------------------------------
        # Final metric comparison
        # -------------------------------------------------------------
        final_rows = []

        final_rows.append(
            evaluate_clustering(
                algorithm="KMeans",
                parameters=f"k={silhouette_k}",
                x=x,
                labels=kmeans_final_labels,
                y_true=y,
                runtime_seconds=kmeans_runtime,
                silhouette_sample_size=(
                    silhouette_sample_size
                ),
                random_seed=seed,
            )
        )

        final_rows.append(
            evaluate_clustering(
                algorithm="Hierarchical",
                parameters=(
                    f"k={selected_hierarchy_k},"
                    f"linkage={selected_linkage}"
                ),
                x=x,
                labels=hierarchy_final_labels,
                y_true=y,
                runtime_seconds=hierarchy_runtime,
                silhouette_sample_size=(
                    silhouette_sample_size
                ),
                random_seed=seed,
            )
        )

        dbscan_runtime = float(
            selected_dbscan_row[
                "runtime_seconds"
            ]
        )

        final_rows.append(
            evaluate_clustering(
                algorithm="DBSCAN",
                parameters=(
                    selected_dbscan_parameters
                ),
                x=x,
                labels=dbscan_final_labels,
                y_true=y,
                runtime_seconds=dbscan_runtime,
                silhouette_sample_size=(
                    silhouette_sample_size
                ),
                random_seed=seed,
            )
        )

        final_rows.append(
            evaluate_clustering(
                algorithm="GMM",
                parameters=(
                    f"k={selected_gmm_k},"
                    f"covariance={selected_covariance}"
                ),
                x=x,
                labels=gmm_final_labels,
                y_true=y,
                runtime_seconds=gmm_runtime,
                silhouette_sample_size=(
                    silhouette_sample_size
                ),
                random_seed=seed,
            )
        )

        final_results = pd.DataFrame(
            final_rows
        )

        final_results.to_csv(
            REPORT_DIR / "final_comparison.csv",
            index=False,
        )

        plot_final_metric_comparison(
            final_results
        )

        write_phase2_summary(
            elbow_k=elbow_k,
            silhouette_k=silhouette_k,
            gap_k=gap_k,
            hierarchical_method=selected_linkage,
            hierarchical_k=selected_hierarchy_k,
            dbscan_parameters=(
                selected_dbscan_parameters
            ),
            gmm_k=selected_gmm_k,
            gmm_covariance=selected_covariance,
            final_results=final_results,
        )

        elapsed = time.perf_counter() - start_time

        execution_record = {
            "status": "completed",
            "elapsed_seconds": float(elapsed),
            "elapsed_minutes": float(
                elapsed / 60
            ),
            "representation": representation,
            "evaluation_size": int(len(x)),
            "random_seed": seed,
            "selected_models": {
                "kmeans": {
                    "k": silhouette_k,
                },
                "hierarchical": {
                    "k": selected_hierarchy_k,
                    "linkage": selected_linkage,
                },
                "dbscan": {
                    "parameters": (
                        selected_dbscan_parameters
                    ),
                },
                "gmm": {
                    "k": selected_gmm_k,
                    "covariance_type": (
                        selected_covariance
                    ),
                },
            },
        }

        save_json(
            execution_record,
            REPORT_DIR / "execution_record.json",
        )

        logging.info("=" * 65)
        logging.info(
            "Phase 2 completed successfully in %.2f minutes",
            elapsed / 60,
        )
        logging.info(
            "Reports: %s",
            REPORT_DIR,
        )
        logging.info(
            "Figures: %s",
            FIGURE_DIR,
        )
        logging.info("=" * 65)

    except Exception as error:
        logging.exception(
            "Phase 2 pipeline failed"
        )

        save_json(
            {
                "status": "failed",
                "error_type": type(error).__name__,
                "error_message": str(error),
            },
            REPORT_DIR / "execution_failed.json",
        )

        raise


if __name__ == "__main__":
    run_phase2()