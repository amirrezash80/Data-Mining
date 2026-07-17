import json
import logging
import re
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
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


ROOT = Path(__file__).resolve().parents[1]

PHASE1_ARRAY_PATH = (
    ROOT
    / "data"
    / "processed"
    / "phase1_arrays.npz"
)

PHASE3_ASSIGNMENT_PATH = (
    ROOT
    / "data"
    / "processed"
    / "phase3_consensus_assignments.parquet"
)

PHASE2_RECOMMENDATION_PATH = (
    ROOT
    / "reports"
    / "phase2"
    / "phase2_final_recommendation.json"
)

PHASE2_FINAL_COMPARISON_PATH = (
    ROOT
    / "reports"
    / "phase2"
    / "final_comparison.csv"
)

PARAMS_PHASE2_PATH = (
    ROOT
    / "params_phase2.yaml"
)

PARAMS_PHASE3_PATH = (
    ROOT
    / "params_phase3.yaml"
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
        format=(
            "%(asctime)s | "
            "%(levelname)s | "
            "%(message)s"
        ),
        handlers=[
            logging.StreamHandler(
                sys.stdout
            ),
            logging.FileHandler(
                REPORT_DIR
                / "phase3_completion_step1.log",
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
        data = yaml.safe_load(file)

    if not isinstance(data, dict):
        raise ValueError(
            f"Invalid configuration: {path}"
        )

    return data


def load_json(
    path: Path,
) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"JSON file not found: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    if not isinstance(data, dict):
        raise ValueError(
            f"Invalid JSON object: {path}"
        )

    return data


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


def normalise_algorithm_name(
    value: Any,
) -> str:
    cleaned = str(value).strip().lower()

    if (
        "kmeans" in cleaned
        or "k-means" in cleaned
    ):
        return "KMeans"

    if (
        "hierarchical" in cleaned
        or "agglomerative" in cleaned
        or "ward" in cleaned
    ):
        return "Hierarchical"

    if "dbscan" in cleaned:
        return "DBSCAN"

    if (
        "gmm" in cleaned
        or "gaussian" in cleaned
    ):
        return "GMM"

    raise ValueError(
        f"Unsupported Phase 2 winner: {value}"
    )


def parse_parameter(
    text: str,
    name: str,
) -> str:
    expression = (
        rf"(?:^|,)\s*"
        rf"{re.escape(name)}"
        rf"\s*=\s*([^,]+)"
    )

    match = re.search(
        expression,
        str(text),
        flags=re.IGNORECASE,
    )

    if match is None:
        return None

    return match.group(1).strip()


def parse_integer_parameter(
    text: str,
    name: str,
    default: int ,
) -> int :
    value = parse_parameter(
        text,
        name,
    )

    if value is None:
        return default

    try:
        return int(float(value))
    except Exception:
        return default


def parse_float_parameter(
    text: str,
    name: str,
    default: float ,
) -> float:
    value = parse_parameter(
        text,
        name,
    )

    if value is None:
        return default

    try:
        return float(value)
    except Exception:
        return default


def select_phase2_winner_from_table() -> dict[str, Any]:
    if not PHASE2_FINAL_COMPARISON_PATH.exists():
        raise FileNotFoundError(
            "Neither Phase 2 recommendation nor "
            "final comparison is available"
        )

    results = pd.read_csv(
        PHASE2_FINAL_COMPARISON_PATH
    )

    required = {
        "algorithm",
        "silhouette",
        "davies_bouldin",
        "calinski_harabasz",
    }

    missing = required - set(
        results.columns
    )

    if missing:
        raise KeyError(
            "Missing Phase 2 comparison columns: "
            f"{sorted(missing)}"
        )

    candidates = results.dropna(
        subset=[
            "silhouette",
            "davies_bouldin",
            "calinski_harabasz",
        ]
    ).copy()

    if candidates.empty:
        raise ValueError(
            "No valid Phase 2 winner candidate"
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

    winner = candidates.sort_values(
        [
            "rank_sum",
            "silhouette",
        ],
        ascending=[
            True,
            False,
        ],
    ).iloc[0]

    return {
        "recommended_algorithm": (
            normalise_algorithm_name(
                winner["algorithm"]
            )
        ),
        "recommended_parameters": str(
            winner.get(
                "parameters",
                "",
            )
        ),
        "recommended_k": (
            int(winner["n_clusters"])
            if (
                "n_clusters"
                in winner.index
                and pd.notna(
                    winner["n_clusters"]
                )
            )
            else None
        ),
        "selection_source": (
            "Reconstructed from final_comparison.csv"
        ),
    }


def load_phase2_winner() -> dict[str, Any]:
    if PHASE2_RECOMMENDATION_PATH.exists():
        recommendation = load_json(
            PHASE2_RECOMMENDATION_PATH
        )

        algorithm = normalise_algorithm_name(
            recommendation[
                "recommended_algorithm"
            ]
        )

        return {
            "recommended_algorithm": algorithm,
            "recommended_parameters": str(
                recommendation.get(
                    "recommended_parameters",
                    "",
                )
            ),
            "recommended_k": (
                int(
                    recommendation[
                        "recommended_k"
                    ]
                )
                if recommendation.get(
                    "recommended_k"
                )
                is not None
                else None
            ),
            "selection_source": str(
                PHASE2_RECOMMENDATION_PATH
                .relative_to(ROOT)
            ),
        }

    return select_phase2_winner_from_table()


def load_common_sample() -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    pd.DataFrame,
]:
    if not PHASE1_ARRAY_PATH.exists():
        raise FileNotFoundError(
            f"Phase 1 arrays not found: "
            f"{PHASE1_ARRAY_PATH}"
        )

    if not PHASE3_ASSIGNMENT_PATH.exists():
        raise FileNotFoundError(
            "Phase 3 consensus assignments "
            "were not found. Run phase3.py first."
        )

    with np.load(
        PHASE1_ARRAY_PATH,
        allow_pickle=False,
    ) as arrays:
        required = {
            "X_pca",
            "y",
            "row_id",
        }

        missing = required - set(
            arrays.files
        )

        if missing:
            raise KeyError(
                f"Missing Phase 1 arrays: "
                f"{sorted(missing)}"
            )

        x_all = arrays[
            "X_pca"
        ].astype(np.float64)

        y_all = arrays[
            "y"
        ].astype(np.int8)

        row_id_all = arrays[
            "row_id"
        ].astype(np.int64)

    assignments = pd.read_parquet(
        PHASE3_ASSIGNMENT_PATH
    )

    required_columns = {
        "array_index",
        "Consensus_cluster",
    }

    missing_columns = (
        required_columns
        - set(assignments.columns)
    )

    if missing_columns:
        raise KeyError(
            "Missing Phase 3 assignment columns: "
            f"{sorted(missing_columns)}"
        )

    if assignments[
        "array_index"
    ].duplicated().any():
        raise ValueError(
            "Duplicate array_index values "
            "in Phase 3 assignments"
        )

    array_indices = assignments[
        "array_index"
    ].to_numpy(dtype=np.int64)

    if np.any(array_indices < 0):
        raise IndexError(
            "Negative array index found"
        )

    if np.any(
        array_indices >= len(x_all)
    ):
        raise IndexError(
            "Phase 3 index exceeds Phase 1 arrays"
        )

    x = x_all[array_indices]
    y_true = y_all[array_indices]
    row_ids = row_id_all[
        array_indices
    ]

    consensus_labels = assignments[
        "Consensus_cluster"
    ].to_numpy(dtype=np.int32)

    if not np.isfinite(x).all():
        raise ValueError(
            "Common PCA sample contains "
            "NaN or infinite values"
        )

    if len(x) != len(
        consensus_labels
    ):
        raise ValueError(
            "Consensus labels and PCA sample "
            "have different lengths"
        )

    return (
        x,
        y_true,
        row_ids,
        consensus_labels,
        assignments,
    )


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


def calculate_internal_metrics(
    x: np.ndarray,
    labels: np.ndarray,
    random_seed: int,
    sample_size: int,
) -> dict[str, float]:
    valid_mask = labels != -1

    valid_x = x[
        valid_mask
    ]

    valid_labels = labels[
        valid_mask
    ]

    unique_labels = np.unique(
        valid_labels
    )

    if (
        len(valid_x) < 3
        or len(unique_labels) < 2
        or len(unique_labels)
        >= len(valid_x)
    ):
        return {
            "silhouette": float("nan"),
            "davies_bouldin": float("nan"),
            "calinski_harabasz": float("nan"),
        }

    effective_sample_size = min(
        sample_size,
        len(valid_x),
    )

    return {
        "silhouette": float(
            silhouette_score(
                valid_x,
                valid_labels,
                metric="euclidean",
                sample_size=(
                    effective_sample_size
                ),
                random_state=random_seed,
            )
        ),
        "davies_bouldin": float(
            davies_bouldin_score(
                valid_x,
                valid_labels,
            )
        ),
        "calinski_harabasz": float(
            calinski_harabasz_score(
                valid_x,
                valid_labels,
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
    method: str,
    parameters: str,
    x: np.ndarray,
    labels: np.ndarray,
    y_true: np.ndarray,
    runtime_seconds: float,
    random_seed: int,
    silhouette_sample_size: int,
) -> dict[str, Any]:
    non_noise = labels[
        labels != -1
    ]

    cluster_count = (
        len(np.unique(non_noise))
        if len(non_noise) > 0
        else 0
    )

    internal = calculate_internal_metrics(
        x=x,
        labels=labels,
        random_seed=random_seed,
        sample_size=(
            silhouette_sample_size
        ),
    )

    external = calculate_external_metrics(
        y_true=y_true,
        labels=labels,
    )

    return {
        "method": method,
        "parameters": parameters,
        "n_records": int(len(x)),
        "n_clusters": int(
            cluster_count
        ),
        "noise_fraction": float(
            np.mean(labels == -1)
        ),
        "runtime_seconds": float(
            runtime_seconds
        ),
        **internal,
        **external,
    }


def fit_phase2_winner(
    x: np.ndarray,
    winner: dict[str, Any],
    params_phase2: dict[str, Any],
    random_seed: int,
) -> tuple[
    np.ndarray,
    str,
    float,
]:
    algorithm = winner[
        "recommended_algorithm"
    ]

    parameter_text = winner.get(
        "recommended_parameters",
        "",
    )

    recommended_k = winner.get(
        "recommended_k"
    )

    k = parse_integer_parameter(
        parameter_text,
        "k",
        default=(
            int(recommended_k)
            if recommended_k
            is not None
            else 2
        ),
    )

    if k is None or k < 2:
        k = 2

    started = time.perf_counter()

    if algorithm == "KMeans":
        model = KMeans(
            n_clusters=k,
            init="k-means++",
            n_init=int(
                params_phase2.get(
                    "kmeans",
                    {},
                ).get(
                    "n_init",
                    20,
                )
            ),
            max_iter=int(
                params_phase2.get(
                    "kmeans",
                    {},
                ).get(
                    "max_iter",
                    300,
                )
            ),
            random_state=random_seed,
        )

        labels = model.fit_predict(
            x
        ).astype(np.int32)

        effective_parameters = (
            f"k={k},"
            f"n_init={model.n_init},"
            "init=k-means++"
        )

    elif algorithm == "Hierarchical":
        linkage_method = (
            parse_parameter(
                parameter_text,
                "linkage",
            )
            or "ward"
        ).lower()

        distance_metric = (
            parse_parameter(
                parameter_text,
                "metric",
            )
            or "euclidean"
        ).lower()

        if linkage_method == "ward":
            distance_metric = "euclidean"

        model = AgglomerativeClustering(
            n_clusters=k,
            linkage=linkage_method,
            metric=distance_metric,
        )

        labels = model.fit_predict(
            x
        ).astype(np.int32)

        effective_parameters = (
            f"k={k},"
            f"linkage={linkage_method},"
            f"metric={distance_metric}"
        )

    elif algorithm == "DBSCAN":
        eps = parse_float_parameter(
            parameter_text,
            "eps",
            default=None,
        )

        min_samples = (
            parse_integer_parameter(
                parameter_text,
                "min_samples",
                default=None,
            )
        )

        if (
            eps is None
            or min_samples is None
        ):
            raise ValueError(
                "The Phase 2 DBSCAN recommendation "
                "does not contain eps and min_samples"
            )

        model = DBSCAN(
            eps=float(eps),
            min_samples=int(
                min_samples
            ),
            metric="euclidean",
            n_jobs=-1,
        )

        labels = model.fit_predict(
            x
        ).astype(np.int32)

        effective_parameters = (
            f"eps={eps},"
            f"min_samples={min_samples},"
            "metric=euclidean"
        )

    elif algorithm == "GMM":
        covariance_type = (
            parse_parameter(
                parameter_text,
                "covariance",
            )
            or parse_parameter(
                parameter_text,
                "covariance_type",
            )
            or "full"
        ).lower()

        model = GaussianMixture(
            n_components=k,
            covariance_type=(
                covariance_type
            ),
            n_init=int(
                params_phase2.get(
                    "gmm",
                    {},
                ).get(
                    "n_init",
                    3,
                )
            ),
            max_iter=int(
                params_phase2.get(
                    "gmm",
                    {},
                ).get(
                    "max_iter",
                    300,
                )
            ),
            reg_covar=1e-6,
            random_state=random_seed,
        )

        with warnings.catch_warnings():
            warnings.simplefilter(
                "ignore",
                ConvergenceWarning,
            )

            labels = model.fit_predict(
                x
            ).astype(np.int32)

        effective_parameters = (
            f"k={k},"
            f"covariance={covariance_type},"
            f"n_init={model.n_init}"
        )

    else:
        raise ValueError(
            f"Unsupported winner: {algorithm}"
        )

    runtime = (
        time.perf_counter()
        - started
    )

    return (
        labels,
        effective_parameters,
        runtime,
    )


def calculate_rank_score(
    comparison: pd.DataFrame,
) -> pd.DataFrame:
    ranked = comparison.copy()

    ranked["rank_silhouette"] = (
        ranked["silhouette"].rank(
            ascending=False,
            method="min",
        )
    )

    ranked["rank_db"] = (
        ranked["davies_bouldin"].rank(
            ascending=True,
            method="min",
        )
    )

    ranked["rank_ch"] = (
        ranked[
            "calinski_harabasz"
        ].rank(
            ascending=False,
            method="min",
        )
    )

    ranked["internal_rank_sum"] = (
        ranked["rank_silhouette"]
        + ranked["rank_db"]
        + ranked["rank_ch"]
    )

    ranked["internal_winner"] = (
        ranked[
            "internal_rank_sum"
        ]
        == ranked[
            "internal_rank_sum"
        ].min()
    )

    return ranked


def build_comparison(
    x: np.ndarray,
    y_true: np.ndarray,
    consensus_labels: np.ndarray,
    winner: dict[str, Any],
    params_phase2: dict[str, Any],
    params_phase3: dict[str, Any],
) -> tuple[
    pd.DataFrame,
    np.ndarray,
    dict[str, Any],
]:
    random_seed = int(
        params_phase3[
            "project"
        ]["random_seed"]
    )

    silhouette_sample_size = int(
        params_phase3.get(
            "evaluation",
            {},
        ).get(
            "silhouette_sample_size",
            2000,
        )
    )

    (
        phase2_labels,
        phase2_parameters,
        phase2_runtime,
    ) = fit_phase2_winner(
        x=x,
        winner=winner,
        params_phase2=(
            params_phase2
        ),
        random_seed=random_seed,
    )

    phase2_row = evaluate_partition(
        method=(
            "Phase 2 winner: "
            + winner[
                "recommended_algorithm"
            ]
        ),
        parameters=phase2_parameters,
        x=x,
        labels=phase2_labels,
        y_true=y_true,
        runtime_seconds=(
            phase2_runtime
        ),
        random_seed=random_seed,
        silhouette_sample_size=(
            silhouette_sample_size
        ),
    )

    consensus_started = (
        time.perf_counter()
    )

    consensus_row = evaluate_partition(
        method="Phase 3 Consensus",
        parameters=(
            "average linkage on "
            "1 - coassociation"
        ),
        x=x,
        labels=consensus_labels,
        y_true=y_true,
        runtime_seconds=(
            time.perf_counter()
            - consensus_started
        ),
        random_seed=random_seed,
        silhouette_sample_size=(
            silhouette_sample_size
        ),
    )

    comparison = calculate_rank_score(
        pd.DataFrame(
            [
                phase2_row,
                consensus_row,
            ]
        )
    )

    partition_agreement = float(
        adjusted_rand_score(
            phase2_labels,
            consensus_labels,
        )
    )

    phase2_metrics = comparison.iloc[0]
    consensus_metrics = comparison.iloc[1]

    improvements = {
        "silhouette_change": float(
            consensus_metrics[
                "silhouette"
            ]
            - phase2_metrics[
                "silhouette"
            ]
        ),
        "davies_bouldin_improvement": float(
            phase2_metrics[
                "davies_bouldin"
            ]
            - consensus_metrics[
                "davies_bouldin"
            ]
        ),
        "calinski_harabasz_change": float(
            consensus_metrics[
                "calinski_harabasz"
            ]
            - phase2_metrics[
                "calinski_harabasz"
            ]
        ),
        "external_ari_change": float(
            consensus_metrics["ari"]
            - phase2_metrics["ari"]
        ),
        "external_nmi_change": float(
            consensus_metrics["nmi"]
            - phase2_metrics["nmi"]
        ),
        "ari_between_phase2_and_consensus": (
            partition_agreement
        ),
    }

    consensus_wins = int(
        improvements[
            "silhouette_change"
        ] > 0
    )

    consensus_wins += int(
        improvements[
            "davies_bouldin_improvement"
        ] > 0
    )

    consensus_wins += int(
        improvements[
            "calinski_harabasz_change"
        ] > 0
    )

    if consensus_wins >= 2:
        conclusion = (
            "Consensus improves the majority "
            "of internal geometric metrics."
        )
    elif consensus_wins == 1:
        conclusion = (
            "Consensus produces a mixed result: "
            "one internal metric improves while "
            "the others do not."
        )
    else:
        conclusion = (
            "Consensus does not improve the "
            "Phase 2 winner on the majority of "
            "internal geometric metrics."
        )

    improvements[
        "internal_metrics_improved_count"
    ] = consensus_wins

    improvements[
        "conclusion"
    ] = conclusion

    return (
        comparison,
        phase2_labels,
        improvements,
    )


def plot_metric_comparison(
    comparison: pd.DataFrame,
) -> Path:
    metrics = [
        "silhouette",
        "davies_bouldin",
        "calinski_harabasz",
        "ari",
        "nmi",
        "ami",
    ]

    melted = comparison.melt(
        id_vars=["method"],
        value_vars=metrics,
        var_name="metric",
        value_name="value",
    )

    graph = sns.catplot(
        data=melted,
        x="method",
        y="value",
        col="metric",
        col_wrap=3,
        kind="bar",
        sharey=False,
        height=4,
        aspect=1.15,
    )

    graph.set_xticklabels(
        rotation=20
    )

    graph.fig.suptitle(
        "Phase 2 Winner versus "
        "Phase 3 Consensus",
        y=1.03,
    )

    path = (
        FIGURE_DIR
        / "phase2_winner_vs_consensus.png"
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


def plot_partition_comparison(
    x: np.ndarray,
    phase2_labels: np.ndarray,
    consensus_labels: np.ndarray,
    winner_name: str,
) -> Path:
    if x.shape[1] < 2:
        raise ValueError(
            "At least two PCA dimensions "
            "are required for plotting"
        )

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(15, 6),
        sharex=True,
        sharey=True,
    )

    first = axes[0].scatter(
        x[:, 0],
        x[:, 1],
        c=phase2_labels,
        cmap="tab20",
        s=14,
        alpha=0.65,
        rasterized=True,
    )

    axes[0].set_title(
        f"Phase 2 Winner: {winner_name}"
    )

    axes[0].set_xlabel("PC1")
    axes[0].set_ylabel("PC2")

    figure.colorbar(
        first,
        ax=axes[0],
        label="Cluster",
    )

    second = axes[1].scatter(
        x[:, 0],
        x[:, 1],
        c=consensus_labels,
        cmap="tab20",
        s=14,
        alpha=0.65,
        rasterized=True,
    )

    axes[1].set_title(
        "Phase 3 Consensus"
    )

    axes[1].set_xlabel("PC1")
    axes[1].set_ylabel("PC2")

    figure.colorbar(
        second,
        ax=axes[1],
        label="Cluster",
    )

    figure.suptitle(
        "Head-to-Head Comparison on "
        "the Same Records"
    )

    return save_figure(
        "phase2_winner_vs_consensus_pca.png"
    )


def write_report(
    winner: dict[str, Any],
    comparison: pd.DataFrame,
    improvements: dict[str, Any],
) -> None:
    phase2 = comparison.iloc[0]
    consensus = comparison.iloc[1]

    lines = [
        "# Phase 3 Advanced-Track Comparison",
        "",
        "The Phase 3 consensus clustering was "
        "compared directly with the recommended "
        "Phase 2 model.",
        "",
        "Both methods were evaluated on the same "
        "records and in the same Phase 1 PCA space.",
        "",
        "The fraud label was not used to fit either "
        "method or choose their hyperparameters. "
        "External metrics are post-hoc only.",
        "",
        "## Phase 2 Winner",
        "",
        f"- Algorithm: "
        f"{winner['recommended_algorithm']}",
        f"- Original recommendation parameters: "
        f"{winner['recommended_parameters']}",
        f"- Selection source: "
        f"{winner['selection_source']}",
        f"- Number of clusters: "
        f"{int(phase2['n_clusters'])}",
        f"- Silhouette: "
        f"{phase2['silhouette']:.6f}",
        f"- Davies-Bouldin: "
        f"{phase2['davies_bouldin']:.6f}",
        f"- Calinski-Harabasz: "
        f"{phase2['calinski_harabasz']:.6f}",
        f"- External ARI: "
        f"{phase2['ari']:.6f}",
        "",
        "## Phase 3 Consensus",
        "",
        f"- Number of clusters: "
        f"{int(consensus['n_clusters'])}",
        f"- Silhouette: "
        f"{consensus['silhouette']:.6f}",
        f"- Davies-Bouldin: "
        f"{consensus['davies_bouldin']:.6f}",
        f"- Calinski-Harabasz: "
        f"{consensus['calinski_harabasz']:.6f}",
        f"- External ARI: "
        f"{consensus['ari']:.6f}",
        "",
        "## Changes Produced by Consensus",
        "",
        f"- Silhouette change: "
        f"{improvements['silhouette_change']:.6f}",
        f"- Davies-Bouldin improvement: "
        f"{improvements['davies_bouldin_improvement']:.6f}",
        f"- Calinski-Harabasz change: "
        f"{improvements['calinski_harabasz_change']:.6f}",
        f"- External ARI change: "
        f"{improvements['external_ari_change']:.6f}",
        f"- ARI between the two partitions: "
        f"{improvements['ari_between_phase2_and_consensus']:.6f}",
        "",
        "## Conclusion",
        "",
        improvements["conclusion"],
        "",
        "A failure of consensus to outperform the "
        "Phase 2 winner is still a valid scientific "
        "result. Consensus is intended to reconcile "
        "algorithm disagreement and can improve "
        "robustness even when geometric separation "
        "does not improve.",
    ]

    (
        REPORT_DIR
        / "phase2_winner_vs_consensus.md"
    ).write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main() -> None:
    ensure_directories()
    setup_logging()
    setup_plots()

    params_phase2 = load_yaml(
        PARAMS_PHASE2_PATH
    )

    params_phase3 = load_yaml(
        PARAMS_PHASE3_PATH
    )

    winner = load_phase2_winner()

    (
        x,
        y_true,
        row_ids,
        consensus_labels,
        original_assignments,
    ) = load_common_sample()

    (
        comparison,
        phase2_labels,
        improvements,
    ) = build_comparison(
        x=x,
        y_true=y_true,
        consensus_labels=(
            consensus_labels
        ),
        winner=winner,
        params_phase2=(
            params_phase2
        ),
        params_phase3=(
            params_phase3
        ),
    )

    comparison.to_csv(
        REPORT_DIR
        / "phase2_winner_vs_consensus.csv",
        index=False,
    )

    save_json(
        improvements,
        REPORT_DIR
        / "phase2_winner_vs_consensus_summary.json",
    )

    output_assignments = pd.DataFrame(
        {
            "array_index": (
                original_assignments[
                    "array_index"
                ].to_numpy(
                    dtype=np.int64
                )
            ),
            "row_id": row_ids,
            "Class_external_only": (
                y_true
            ),
            "Phase2_winner_cluster": (
                phase2_labels
            ),
            "Consensus_cluster": (
                consensus_labels
            ),
        }
    )

    output_assignments.to_parquet(
        PROCESSED_DIR
        / "phase3_head_to_head_assignments.parquet",
        index=False,
    )

    metrics_figure = (
        plot_metric_comparison(
            comparison
        )
    )

    partition_figure = (
        plot_partition_comparison(
            x=x,
            phase2_labels=(
                phase2_labels
            ),
            consensus_labels=(
                consensus_labels
            ),
            winner_name=winner[
                "recommended_algorithm"
            ],
        )
    )

    write_report(
        winner=winner,
        comparison=comparison,
        improvements=improvements,
    )

    completion_record = {
        "status": "completed",
        "phase2_winner": winner,
        "sample_size": int(
            len(x)
        ),
        "pca_dimensions": int(
            x.shape[1]
        ),
        "phase2_cluster_count": int(
            comparison.iloc[0][
                "n_clusters"
            ]
        ),
        "consensus_cluster_count": int(
            comparison.iloc[1][
                "n_clusters"
            ]
        ),
        **improvements,
        "generated_figures": [
            str(
                metrics_figure.relative_to(
                    ROOT
                )
            ),
            str(
                partition_figure.relative_to(
                    ROOT
                )
            ),
        ],
    }

    save_json(
        completion_record,
        REPORT_DIR
        / "phase3_completion_step1_record.json",
    )

    logging.info(
        "Phase 3 completion step 1 succeeded"
    )


if __name__ == "__main__":
    main()