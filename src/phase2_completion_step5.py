import json
import logging
import math
import sys
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages


ROOT = Path(__file__).resolve().parents[1]

REPORT_DIR = ROOT / "reports" / "phase2"
FIGURE_DIR = ROOT / "reports" / "figures" / "phase2"
FINAL_REPORT_DIR = ROOT / "reports" / "final"

FINAL_COMPARISON_PATH = (
    REPORT_DIR / "final_comparison.csv"
)

OUTPUT_MARKDOWN = (
    REPORT_DIR / "phase2_integrated_report.md"
)

OUTPUT_PDF = (
    REPORT_DIR / "phase2_integrated_report.pdf"
)

OUTPUT_RECOMMENDATION = (
    REPORT_DIR / "phase2_final_recommendation.json"
)

OUTPUT_SCOREBOARD = (
    REPORT_DIR / "phase2_final_scoreboard.csv"
)

OUTPUT_PERFORMANCE_TABLE = (
    REPORT_DIR / "phase2_unified_performance_table.csv"
)


def ensure_directories() -> None:
    for directory in [
        REPORT_DIR,
        FIGURE_DIR,
        FINAL_REPORT_DIR,
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
                / "phase2_completion_step5.log",
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


def read_csv_optional(
    path: Path,
) -> pd.DataFrame:
    if not path.exists():
        logging.warning(
            "Optional CSV was not found: %s",
            path,
        )
        return pd.DataFrame()

    try:
        return pd.read_csv(path)
    except Exception as error:
        logging.warning(
            "Could not read %s: %s",
            path,
            error,
        )
        return pd.DataFrame()


def read_json_optional(
    path: Path,
) -> dict[str, Any]:
    if not path.exists():
        logging.warning(
            "Optional JSON was not found: %s",
            path,
        )
        return {}

    try:
        with path.open(
            "r",
            encoding="utf-8",
        ) as file:
            value = json.load(file)

        if isinstance(value, dict):
            return value

        return {
            "value": value
        }

    except Exception as error:
        logging.warning(
            "Could not read %s: %s",
            path,
            error,
        )
        return {}


def read_text_optional(
    path: Path,
) -> str:
    if not path.exists():
        return ""

    try:
        return path.read_text(
            encoding="utf-8",
        )
    except Exception:
        return ""


def safe_float(
    value: Any,
) -> float:
    try:
        number = float(value)

        if math.isnan(number):
            return None

        if math.isinf(number):
            return None

        return number

    except Exception:
        return None


def format_number(
    value: Any,
    digits: int = 5,
) -> str:
    number = safe_float(value)

    if number is None:
        return "N/A"

    return f"{number:.{digits}f}"


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

    return str(value).strip()


def detect_k(
    row: pd.Series,
) -> int :
    for column in [
        "k",
        "requested_k",
        "n_clusters",
        "actual_k",
    ]:
        if (
            column in row.index
            and pd.notna(row[column])
        ):
            try:
                return int(
                    float(row[column])
                )
            except Exception:
                pass

    parameters = str(
        row.get(
            "parameters",
            "",
        )
    )

    fragments = parameters.replace(
        " ",
        "",
    ).split(",")

    for fragment in fragments:
        if fragment.startswith("k="):
            try:
                return int(
                    fragment.split(
                        "=",
                        maxsplit=1,
                    )[1]
                )
            except Exception:
                pass

    return None


def standardise_experiment_table(
    frame: pd.DataFrame,
    source: str,
    default_algorithm: str ,
) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()

    output = frame.copy()

    if "algorithm" not in output.columns:
        output["algorithm"] = (
            default_algorithm
            if default_algorithm is not None
            else source
        )

    output["algorithm"] = output[
        "algorithm"
    ].map(
        normalise_algorithm_name
    )

    output["source_experiment"] = source

    if "parameters" not in output.columns:
        parameter_parts = []

        for _, row in output.iterrows():
            parts = []

            for column in [
                "k",
                "requested_k",
                "actual_k",
                "linkage",
                "strategy",
                "eps",
                "min_samples",
                "covariance_type",
                "distance_metric",
            ]:
                if (
                    column in output.columns
                    and pd.notna(
                        row.get(column)
                    )
                ):
                    parts.append(
                        f"{column}={row.get(column)}"
                    )

            parameter_parts.append(
                ",".join(parts)
            )

        output["parameters"] = (
            parameter_parts
        )

    output["selected_k"] = [
        detect_k(row)
        for _, row in output.iterrows()
    ]

    rename_map = {
        "ari_external": "ari",
        "nmi_external": "nmi",
        "ami_external": "ami",
        "purity_external": "purity",
        "fowlkes_mallows_external": (
            "fowlkes_mallows"
        ),
        "homogeneity_external": (
            "homogeneity"
        ),
        "completeness_external": (
            "completeness"
        ),
        "v_measure_external": (
            "v_measure"
        ),
        "silhouette_matching_metric": (
            "silhouette"
        ),
    }

    for source_column, target_column in (
        rename_map.items()
    ):
        if (
            target_column not in output.columns
            and source_column in output.columns
        ):
            output[target_column] = output[
                source_column
            ]

    required_columns = [
        "algorithm",
        "source_experiment",
        "parameters",
        "selected_k",
        "n_clusters",
        "noise_fraction",
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
        "runtime_seconds",
    ]

    for column in required_columns:
        if column not in output.columns:
            output[column] = np.nan

    return output[
        required_columns
    ]


def build_unified_performance_table() -> pd.DataFrame:
    specifications = [
        (
            REPORT_DIR / "kmeans_search.csv",
            "KMeans grid search",
            "KMeans",
        ),
        (
            REPORT_DIR / "hierarchical_search.csv",
            "Hierarchical grid search",
            "Hierarchical",
        ),
        (
            REPORT_DIR
            / "hierarchical_cut_strategy_comparison.csv",
            "Hierarchical cutting strategies",
            "Hierarchical",
        ),
        (
            REPORT_DIR / "dbscan_search.csv",
            "DBSCAN grid search",
            "DBSCAN",
        ),
        (
            REPORT_DIR / "gmm_search.csv",
            "GMM grid search",
            "GMM",
        ),
        (
            REPORT_DIR
            / "distance_metric_search.csv",
            "Distance metric sensitivity",
            "Hierarchical",
        ),
        (
            REPORT_DIR / "final_comparison.csv",
            "Final selected configurations",
            None,
        ),
    ]

    tables = []

    for path, source, algorithm in specifications:
        frame = read_csv_optional(path)

        standardised = (
            standardise_experiment_table(
                frame=frame,
                source=source,
                default_algorithm=algorithm,
            )
        )

        if not standardised.empty:
            tables.append(
                standardised
            )

    if not tables:
        raise FileNotFoundError(
            "No Phase 2 experiment tables were found"
        )

    unified = pd.concat(
        tables,
        ignore_index=True,
    )

    unified.to_csv(
        OUTPUT_PERFORMANCE_TABLE,
        index=False,
    )

    return unified


def load_final_comparison() -> pd.DataFrame:
    if not FINAL_COMPARISON_PATH.exists():
        raise FileNotFoundError(
            "Required final comparison was not found: "
            f"{FINAL_COMPARISON_PATH}"
        )

    frame = pd.read_csv(
        FINAL_COMPARISON_PATH
    )

    required_columns = {
        "algorithm",
        "silhouette",
        "davies_bouldin",
        "calinski_harabasz",
    }

    missing = required_columns - set(
        frame.columns
    )

    if missing:
        raise KeyError(
            "Missing final comparison columns: "
            f"{sorted(missing)}"
        )

    frame["algorithm_original"] = frame[
        "algorithm"
    ].astype(str)

    frame["algorithm"] = frame[
        "algorithm"
    ].map(
        normalise_algorithm_name
    )

    return frame


def calculate_seed_stability_summary() -> pd.DataFrame:
    seed_stability = read_csv_optional(
        REPORT_DIR / "seed_stability.csv"
    )

    if seed_stability.empty:
        return pd.DataFrame()

    if not {
        "algorithm",
        "pairwise_ari",
    }.issubset(
        seed_stability.columns
    ):
        return pd.DataFrame()

    seed_stability["algorithm"] = (
        seed_stability[
            "algorithm"
        ].map(
            normalise_algorithm_name
        )
    )

    summary = (
        seed_stability.groupby(
            "algorithm"
        )
        .agg(
            seed_ari_mean=(
                "pairwise_ari",
                "mean",
            ),
            seed_ari_std=(
                "pairwise_ari",
                "std",
            ),
            seed_ari_min=(
                "pairwise_ari",
                "min",
            ),
            seed_ari_max=(
                "pairwise_ari",
                "max",
            ),
            seed_pair_count=(
                "pairwise_ari",
                "count",
            ),
        )
        .reset_index()
    )

    return summary


def calculate_bootstrap_summary() -> pd.DataFrame:
    preferred_path = (
        REPORT_DIR
        / "bootstrap_k_stability_summary.csv"
    )

    if preferred_path.exists():
        return pd.read_csv(
            preferred_path
        )

    legacy_path = (
        REPORT_DIR
        / "bootstrap_stability.csv"
    )

    legacy = read_csv_optional(
        legacy_path
    )

    if legacy.empty:
        return pd.DataFrame()

    if not {
        "algorithm",
        "pairwise_ari",
    }.issubset(
        legacy.columns
    ):
        return pd.DataFrame()

    legacy["algorithm"] = legacy[
        "algorithm"
    ].map(
        normalise_algorithm_name
    )

    return (
        legacy.groupby("algorithm")
        .agg(
            bootstrap_ari_mean=(
                "pairwise_ari",
                "mean",
            ),
            bootstrap_ari_std=(
                "pairwise_ari",
                "std",
            ),
            bootstrap_ari_min=(
                "pairwise_ari",
                "min",
            ),
        )
        .reset_index()
    )


def calculate_internal_scoreboard(
    final_comparison: pd.DataFrame,
) -> pd.DataFrame:
    candidates = final_comparison.dropna(
        subset=[
            "silhouette",
            "davies_bouldin",
            "calinski_harabasz",
        ]
    ).copy()

    if candidates.empty:
        raise ValueError(
            "No final configuration has all "
            "three internal metrics"
        )

    candidates["rank_silhouette"] = (
        candidates["silhouette"].rank(
            ascending=False,
            method="min",
        )
    )

    candidates["rank_davies_bouldin"] = (
        candidates[
            "davies_bouldin"
        ].rank(
            ascending=True,
            method="min",
        )
    )

    candidates["rank_calinski_harabasz"] = (
        candidates[
            "calinski_harabasz"
        ].rank(
            ascending=False,
            method="min",
        )
    )

    candidates["internal_rank_sum"] = (
        candidates["rank_silhouette"]
        + candidates[
            "rank_davies_bouldin"
        ]
        + candidates[
            "rank_calinski_harabasz"
        ]
    )

    candidates["internal_rank_average"] = (
        candidates["internal_rank_sum"]
        / 3.0
    )

    seed_summary = (
        calculate_seed_stability_summary()
    )

    if not seed_summary.empty:
        candidates = candidates.merge(
            seed_summary,
            on="algorithm",
            how="left",
        )
    else:
        candidates["seed_ari_mean"] = np.nan
        candidates["seed_ari_std"] = np.nan
        candidates["seed_ari_min"] = np.nan
        candidates["seed_ari_max"] = np.nan
        candidates["seed_pair_count"] = np.nan

    candidates["selected_k"] = [
        detect_k(row)
        for _, row in candidates.iterrows()
    ]

    candidates["final_order"] = (
        candidates.sort_values(
            [
                "internal_rank_sum",
                "silhouette",
                "davies_bouldin",
                "calinski_harabasz",
            ],
            ascending=[
                True,
                False,
                True,
                False,
            ],
        )
        .reset_index()
        .reset_index()
        .set_index("index")[
            "level_0"
        ]
        + 1
    )

    candidates = candidates.sort_values(
        "final_order"
    ).reset_index(
        drop=True
    )

    candidates.to_csv(
        OUTPUT_SCOREBOARD,
        index=False,
    )

    return candidates


def load_k_determination_summary() -> dict[str, Any]:
    result: dict[str, Any] = {}

    kmeans = read_csv_optional(
        REPORT_DIR / "kmeans_search.csv"
    )

    if not kmeans.empty:
        if {
            "k",
            "silhouette",
        }.issubset(kmeans.columns):
            valid = kmeans.dropna(
                subset=["silhouette"]
            )

            if not valid.empty:
                best = valid.loc[
                    valid[
                        "silhouette"
                    ].idxmax()
                ]

                result[
                    "silhouette_selected_k"
                ] = int(best["k"])

                result[
                    "maximum_kmeans_silhouette"
                ] = float(
                    best["silhouette"]
                )

        if {
            "k",
            "inertia",
        }.issubset(kmeans.columns):
            result[
                "kmeans_candidate_range"
            ] = [
                int(kmeans["k"].min()),
                int(kmeans["k"].max()),
            ]

    gap = read_csv_optional(
        REPORT_DIR / "gap_statistic.csv"
    )

    if not gap.empty and {
        "k",
        "gap",
    }.issubset(gap.columns):
        selected_gap_k = None

        for position in range(
            len(gap) - 1
        ):
            current = gap.iloc[position]
            following = gap.iloc[
                position + 1
            ]

            standard_error = float(
                following.get(
                    "standard_error",
                    0.0,
                )
            )

            if (
                current["gap"]
                >= following["gap"]
                - standard_error
            ):
                selected_gap_k = int(
                    current["k"]
                )
                break

        if selected_gap_k is None:
            selected_gap_k = int(
                gap.loc[
                    gap["gap"].idxmax(),
                    "k",
                ]
            )

        result["gap_selected_k"] = (
            selected_gap_k
        )

    gmm = read_csv_optional(
        REPORT_DIR / "gmm_search.csv"
    )

    if not gmm.empty and {
        "k",
        "bic",
    }.issubset(gmm.columns):
        valid = gmm.dropna(
            subset=["bic"]
        )

        if not valid.empty:
            best = valid.loc[
                valid["bic"].idxmin()
            ]

            result["gmm_bic_selected_k"] = int(
                best["k"]
            )

            result[
                "gmm_bic_selected_covariance"
            ] = str(
                best.get(
                    "covariance_type",
                    "unknown",
                )
            )

            result["minimum_gmm_bic"] = float(
                best["bic"]
            )

    return result


def load_agreement_summary() -> dict[str, Any]:
    path = (
        REPORT_DIR
        / "algorithm_agreement.csv"
    )

    frame = read_csv_optional(path)

    if frame.empty:
        return {}

    first_column = frame.columns[0]

    if (
        first_column not in frame.columns[1:]
        and not pd.api.types.is_numeric_dtype(
            frame[first_column]
        )
    ):
        frame = frame.set_index(
            first_column
        )

    numeric = frame.select_dtypes(
        include=[np.number]
    )

    if numeric.empty:
        return {}

    matrix = numeric.to_numpy(
        dtype=float
    )

    if matrix.shape[0] == matrix.shape[1]:
        off_diagonal_mask = ~np.eye(
            matrix.shape[0],
            dtype=bool,
        )

        off_diagonal = matrix[
            off_diagonal_mask
        ]
    else:
        off_diagonal = matrix.reshape(-1)

    off_diagonal = off_diagonal[
        np.isfinite(off_diagonal)
    ]

    if len(off_diagonal) == 0:
        return {}

    return {
        "mean_pairwise_algorithm_ari": float(
            off_diagonal.mean()
        ),
        "minimum_pairwise_algorithm_ari": float(
            off_diagonal.min()
        ),
        "maximum_pairwise_algorithm_ari": float(
            off_diagonal.max()
        ),
    }


def classify_stability(
    value: float ,
) -> str:
    if value is None:
        return "not available"

    if value >= 0.90:
        return "very high"

    if value >= 0.75:
        return "high"

    if value >= 0.50:
        return "moderate"

    return "low"


def classify_silhouette(
    value: float ,
) -> str:
    if value is None:
        return "not available"

    if value >= 0.70:
        return "strong"

    if value >= 0.50:
        return "reasonably separated"

    if value >= 0.25:
        return "weak to moderate"

    return "weak"


def extract_winner_stability(
    winner: pd.Series,
    bootstrap_summary: pd.DataFrame,
) -> dict[str, Any]:
    algorithm = str(
        winner["algorithm"]
    )

    selected_k = detect_k(winner)

    result: dict[str, Any] = {}

    seed_value = safe_float(
        winner.get(
            "seed_ari_mean"
        )
    )

    if seed_value is not None:
        result["seed_ari_mean"] = seed_value
        result["seed_stability_level"] = (
            classify_stability(
                seed_value
            )
        )

    if bootstrap_summary.empty:
        return result

    if (
        selected_k is not None
        and "k" in bootstrap_summary.columns
    ):
        selected = bootstrap_summary[
            bootstrap_summary["k"]
            == selected_k
        ]

        if not selected.empty:
            row = selected.iloc[0]

            for column in [
                "ari_mean",
                "mean_jaccard",
                "worst_cluster_mean_jaccard",
            ]:
                if column in row.index:
                    value = safe_float(
                        row[column]
                    )

                    if value is not None:
                        result[
                            f"bootstrap_{column}"
                        ] = value

            jaccard_value = safe_float(
                row.get(
                    "mean_jaccard"
                )
            )

            if jaccard_value is not None:
                result[
                    "bootstrap_stability_level"
                ] = classify_stability(
                    jaccard_value
                )

    elif "algorithm" in bootstrap_summary.columns:
        selected = bootstrap_summary[
            bootstrap_summary[
                "algorithm"
            ].map(
                normalise_algorithm_name
            )
            == algorithm
        ]

        if not selected.empty:
            row = selected.iloc[0]

            value = safe_float(
                row.get(
                    "bootstrap_ari_mean"
                )
            )

            if value is not None:
                result[
                    "bootstrap_ari_mean"
                ] = value

                result[
                    "bootstrap_stability_level"
                ] = classify_stability(
                    value
                )

    return result


def build_recommendation(
    scoreboard: pd.DataFrame,
) -> dict[str, Any]:
    winner = scoreboard.iloc[0]
    runner_up = (
        scoreboard.iloc[1]
        if len(scoreboard) > 1
        else None
    )

    bootstrap_summary = (
        calculate_bootstrap_summary()
    )

    winner_stability = (
        extract_winner_stability(
            winner=winner,
            bootstrap_summary=(
                bootstrap_summary
            ),
        )
    )

    distance_sensitivity = (
        read_json_optional(
            REPORT_DIR
            / "distance_metric_selected_agreement.json"
        )
    )

    error_summary = read_json_optional(
        REPORT_DIR
        / "preferred_model_error_summary.json"
    )

    bootstrap_diagnostics = (
        read_json_optional(
            REPORT_DIR
            / "bootstrap_coclustering_diagnostics.json"
        )
    )

    agreement_summary = (
        load_agreement_summary()
    )

    k_summary = (
        load_k_determination_summary()
    )

    winner_algorithm = str(
        winner["algorithm"]
    )

    winner_parameters = str(
        winner.get(
            "parameters",
            "",
        )
    )

    winner_k = detect_k(winner)

    silhouette = safe_float(
        winner.get("silhouette")
    )

    davies_bouldin = safe_float(
        winner.get(
            "davies_bouldin"
        )
    )

    calinski_harabasz = safe_float(
        winner.get(
            "calinski_harabasz"
        )
    )

    runtime = safe_float(
        winner.get(
            "runtime_seconds"
        )
    )

    reasons = [
        (
            "It achieved the strongest aggregate rank "
            "across Silhouette, Davies-Bouldin, and "
            "Calinski-Harabasz among the final candidate "
            "configurations."
        )
    ]

    if silhouette is not None:
        reasons.append(
            "Its Silhouette coefficient was "
            f"{silhouette:.6f}, indicating "
            f"{classify_silhouette(silhouette)} "
            "geometric separation."
        )

    seed_level = winner_stability.get(
        "seed_stability_level"
    )

    if seed_level is not None:
        reasons.append(
            "Its random-seed stability was classified "
            f"as {seed_level}."
        )

    bootstrap_level = winner_stability.get(
        "bootstrap_stability_level"
    )

    if bootstrap_level is not None:
        reasons.append(
            "Its bootstrap membership stability was "
            f"classified as {bootstrap_level}."
        )

    distance_level = (
        distance_sensitivity.get(
            "distance_sensitivity_level"
        )
    )

    limitations = [
        (
            "The external fraud label was not used for "
            "model selection; external scores are "
            "post-hoc descriptive evidence only."
        ),
        (
            "The fraud class is highly imbalanced, so "
            "purity can be misleading."
        ),
        (
            "A transaction profile cluster is not the "
            "same thing as a fraud class."
        ),
    ]

    if distance_level:
        limitations.append(
            "Sensitivity to changing Euclidean distance "
            "to Manhattan distance was classified as "
            f"{distance_level}."
        )

    negative_fraction = safe_float(
        error_summary.get(
            "negative_silhouette_fraction"
        )
    )

    if negative_fraction is not None:
        limitations.append(
            "Among the inspected lowest-silhouette "
            "records, the negative-silhouette fraction "
            f"was {negative_fraction:.6f}."
        )

    if winner_algorithm == "KMeans":
        limitations.append(
            "K-Means assumes approximately convex, "
            "isotropic clusters of comparable variance."
        )

    elif winner_algorithm == "Hierarchical":
        limitations.append(
            "Hierarchical partitions depend on linkage "
            "choice and dendrogram cutting strategy."
        )

    elif winner_algorithm == "DBSCAN":
        limitations.append(
            "DBSCAN results depend strongly on eps and "
            "min_samples, and varying cluster densities "
            "can reduce reliability."
        )

    elif winner_algorithm == "GMM":
        limitations.append(
            "GMM assumes that data are adequately "
            "represented by a finite mixture of Gaussian "
            "components."
        )

    external_metrics = {}

    for column in [
        "ari",
        "nmi",
        "ami",
        "fowlkes_mallows",
        "homogeneity",
        "completeness",
        "v_measure",
        "purity",
    ]:
        value = safe_float(
            winner.get(column)
        )

        if value is not None:
            external_metrics[column] = value

    recommendation = {
        "selection_policy": (
            "Winner selected exclusively by aggregate "
            "ranking of internal metrics: Silhouette "
            "descending, Davies-Bouldin ascending, and "
            "Calinski-Harabasz descending."
        ),
        "recommended_algorithm": (
            winner_algorithm
        ),
        "recommended_parameters": (
            winner_parameters
        ),
        "recommended_k": winner_k,
        "internal_metrics": {
            "silhouette": silhouette,
            "davies_bouldin": (
                davies_bouldin
            ),
            "calinski_harabasz": (
                calinski_harabasz
            ),
        },
        "runtime_seconds": runtime,
        "external_metrics_posthoc_only": (
            external_metrics
        ),
        "stability": winner_stability,
        "distance_sensitivity": (
            distance_sensitivity
        ),
        "bootstrap_matrix_diagnostics": (
            bootstrap_diagnostics
        ),
        "algorithm_agreement": (
            agreement_summary
        ),
        "determining_k_summary": (
            k_summary
        ),
        "reasons": reasons,
        "limitations": limitations,
        "runner_up": (
            {
                "algorithm": str(
                    runner_up["algorithm"]
                ),
                "parameters": str(
                    runner_up.get(
                        "parameters",
                        "",
                    )
                ),
                "k": detect_k(
                    runner_up
                ),
                "silhouette": safe_float(
                    runner_up.get(
                        "silhouette"
                    )
                ),
                "davies_bouldin": safe_float(
                    runner_up.get(
                        "davies_bouldin"
                    )
                ),
                "calinski_harabasz": safe_float(
                    runner_up.get(
                        "calinski_harabasz"
                    )
                ),
            }
            if runner_up is not None
            else None
        ),
        "generated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
    }

    save_json(
        recommendation,
        OUTPUT_RECOMMENDATION,
    )

    return recommendation


def dataframe_to_markdown(
    frame: pd.DataFrame,
    columns: list[str] ,
    maximum_rows: int = 50,
) -> str:
    if frame.empty:
        return "No data available."

    display = frame.copy()

    if columns is not None:
        available = [
            column
            for column in columns
            if column in display.columns
        ]

        display = display[
            available
        ]

    display = display.head(
        maximum_rows
    )

    for column in display.select_dtypes(
        include=[np.number]
    ).columns:
        display[column] = display[
            column
        ].map(
            lambda value: (
                f"{value:.6f}"
                if pd.notna(value)
                else ""
            )
        )

    headers = [
        str(column)
        for column in display.columns
    ]

    header_line = (
        "| "
        + " | ".join(headers)
        + " |"
    )

    separator_line = (
        "| "
        + " | ".join(
            ["---"] * len(headers)
        )
        + " |"
    )

    rows = []

    for values in display.astype(
        str
    ).itertuples(
        index=False,
        name=None,
    ):
        cleaned = [
            value.replace(
                "|",
                "\\|",
            ).replace(
                "\n",
                " ",
            )
            for value in values
        ]

        rows.append(
            "| "
            + " | ".join(cleaned)
            + " |"
        )

    return "\n".join(
        [
            header_line,
            separator_line,
            *rows,
        ]
    )


def recommendation_to_text(
    recommendation: dict[str, Any],
) -> str:
    algorithm = recommendation[
        "recommended_algorithm"
    ]

    parameters = recommendation[
        "recommended_parameters"
    ]

    selected_k = recommendation[
        "recommended_k"
    ]

    metrics = recommendation[
        "internal_metrics"
    ]

    lines = [
        (
            f"The recommended Phase 2 clustering is "
            f"{algorithm} with parameters "
            f"{parameters}."
        )
    ]

    if selected_k is not None:
        lines.append(
            f"The recommended number of clusters is "
            f"k={selected_k}."
        )

    lines.append(
        "This recommendation was determined only from "
        "internal clustering metrics."
    )

    lines.append(
        "The selected configuration obtained "
        f"Silhouette={format_number(metrics.get('silhouette'))}, "
        f"Davies-Bouldin="
        f"{format_number(metrics.get('davies_bouldin'))}, "
        f"and Calinski-Harabasz="
        f"{format_number(metrics.get('calinski_harabasz'))}."
    )

    for reason in recommendation[
        "reasons"
    ]:
        lines.append(reason)

    return " ".join(lines)


def build_markdown_report(
    unified: pd.DataFrame,
    scoreboard: pd.DataFrame,
    recommendation: dict[str, Any],
) -> str:
    kmeans = read_csv_optional(
        REPORT_DIR / "kmeans_search.csv"
    )

    gap = read_csv_optional(
        REPORT_DIR / "gap_statistic.csv"
    )

    gmm = read_csv_optional(
        REPORT_DIR / "gmm_search.csv"
    )

    hierarchical = read_csv_optional(
        REPORT_DIR
        / "hierarchical_cut_strategy_comparison.csv"
    )

    seed_stability = read_csv_optional(
        REPORT_DIR / "seed_stability.csv"
    )

    bootstrap = read_csv_optional(
        REPORT_DIR
        / "bootstrap_k_stability_summary.csv"
    )

    metric_sensitivity = read_csv_optional(
        REPORT_DIR
        / "distance_metric_selected_results.csv"
    )

    error_summary = read_json_optional(
        REPORT_DIR
        / "preferred_model_error_summary.json"
    )

    agreement_summary = (
        recommendation.get(
            "algorithm_agreement",
            {},
        )
    )

    determining_k = recommendation.get(
        "determining_k_summary",
        {},
    )

    winner_text = recommendation_to_text(
        recommendation
    )

    scoreboard_table = dataframe_to_markdown(
        scoreboard,
        columns=[
            "final_order",
            "algorithm",
            "parameters",
            "selected_k",
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
            "seed_ari_mean",
            "internal_rank_sum",
        ],
    )

    final_table = dataframe_to_markdown(
        unified[
            unified[
                "source_experiment"
            ]
            == "Final selected configurations"
        ],
        columns=[
            "algorithm",
            "parameters",
            "selected_k",
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
        ],
    )

    report = f"""# Phase 2 Integrated Clustering Report

## 1. Objective

Phase 2 implements, tunes, and compares clustering algorithms from four families on the same Phase 1 representation:

- Partitioning: K-Means
- Hierarchical: Agglomerative clustering
- Density-based: DBSCAN
- Model-based: Gaussian Mixture Model

The fraud label is excluded from clustering, hyperparameter tuning, determining k, distance selection, and stability selection. It is used only after model selection for external evaluation.

## 2. Experimental Data

The analysis uses the PCA representation produced in Phase 1. Hyperparameter searches are conducted on a reproducible representative subset selected from the Phase 1 training partition without consulting the fraud label.

## 3. Determining the Number of Clusters

The following methods were applied:

- Elbow and Kneedle on K-Means inertia
- Average Silhouette
- Gap Statistic
- Davies-Bouldin and Calinski-Harabasz
- GMM BIC and AIC
- Bootstrap Jaccard stability

Summary:

    {json.dumps(determining_k, ensure_ascii=False, indent=2).replace(chr(10), chr(10) + "    ")}

### 3.1 K-Means search

{dataframe_to_markdown(
    kmeans,
    columns=[
        "k",
        "inertia",
        "silhouette",
        "davies_bouldin",
        "calinski_harabasz",
        "runtime_seconds",
    ],
)}

### 3.2 Gap Statistic

{dataframe_to_markdown(
    gap,
    columns=[
        "k",
        "gap",
        "standard_error",
        "real_log_dispersion",
    ],
)}

### 3.3 GMM information criteria

{dataframe_to_markdown(
    gmm,
    columns=[
        "k",
        "covariance_type",
        "bic",
        "aic",
        "silhouette",
        "davies_bouldin",
        "calinski_harabasz",
        "converged",
        "runtime_seconds",
    ],
    maximum_rows=100,
)}

## 4. Hierarchical Clustering

Single, complete, average, and Ward linkage were evaluated. Cophenetic correlation was used to assess dendrogram fidelity. Two cutting strategies were compared:

- fixed-height cut
- cut selected by maximum Silhouette

{dataframe_to_markdown(
    hierarchical,
    columns=[
        "linkage",
        "strategy",
        "requested_k",
        "actual_k",
        "cut_height",
        "cophenetic_correlation",
        "silhouette",
        "davies_bouldin",
        "calinski_harabasz",
        "ari",
        "runtime_seconds",
    ],
    maximum_rows=50,
)}

## 5. Internal and External Evaluation

Internal metrics determine the recommendation:

- Silhouette: higher is better
- Davies-Bouldin: lower is better
- Calinski-Harabasz: higher is better

External metrics are reported post-hoc:

- ARI
- NMI
- AMI
- Fowlkes-Mallows
- Homogeneity
- Completeness
- V-measure
- Purity

### 5.1 Final selected configurations

{final_table}

### 5.2 Internal ranking scoreboard

{scoreboard_table}

## 6. Stability Analysis

### 6.1 Seed stability

K-Means and GMM were rerun under multiple random seeds. Pairwise ARI between runs measures sensitivity to initialisation.

{dataframe_to_markdown(
    seed_stability,
    columns=[
        "algorithm",
        "pair_id",
        "pairwise_ari",
    ],
    maximum_rows=40,
)}

### 6.2 Bootstrap stability

Bootstrap resampling was evaluated through ARI to a reference partition, cluster-membership Jaccard, and the co-clustering probability matrix.

{dataframe_to_markdown(
    bootstrap,
    columns=[
        "k",
        "bootstrap_runs",
        "ari_mean",
        "ari_std",
        "mean_jaccard",
        "jaccard_std",
        "minimum_jaccard",
        "worst_cluster_mean_jaccard",
    ],
)}

## 7. Algorithm Agreement

Pairwise ARI between final algorithms was calculated and visualised as a heatmap.

    {json.dumps(agreement_summary, ensure_ascii=False, indent=2).replace(chr(10), chr(10) + "    ")}

Low agreement indicates that alternative clustering assumptions recover different transaction structures. This disagreement is scientifically informative rather than necessarily an implementation error.

## 8. Per-Point Silhouette and Error Analysis

The lowest-silhouette records of the preferred clustering were inspected. Negative values indicate probable misassignment, while values near zero indicate boundary observations.

    {json.dumps(error_summary, ensure_ascii=False, indent=2).replace(chr(10), chr(10) + "    ")}

The full record-level analysis is stored in `lowest_silhouette_records.csv`.

## 9. Distance-Metric Sensitivity

Agglomerative clustering with average linkage was evaluated under Euclidean and Manhattan distances while holding the data, linkage, and candidate k values fixed.

{dataframe_to_markdown(
    metric_sensitivity,
    columns=[
        "distance_metric",
        "requested_k",
        "actual_k",
        "silhouette_matching_metric",
        "silhouette_euclidean",
        "silhouette_manhattan",
        "davies_bouldin",
        "calinski_harabasz",
        "ari_external",
        "runtime_seconds",
    ],
)}

The agreement between the two selected partitions is documented in `distance_metric_selected_agreement.json`.

## 10. Final Recommendation

{winner_text}

### Reasons

{chr(10).join(
    "- " + reason
    for reason in recommendation["reasons"]
)}

### Limitations

{chr(10).join(
    "- " + limitation
    for limitation in recommendation["limitations"]
)}

## 11. Answer to the Research Question

The Phase 2 experiments indicate whether transaction records admit stable profile clusters under classical clustering assumptions. The recommended partition represents transaction-profile structure rather than a direct reconstruction of the binary fraud label.

External fraud metrics and fraud composition indicate whether discovered profiles are enriched for fraud. Low external agreement does not automatically invalidate the clustering, because fraud may occur across multiple transaction profiles rather than form one isolated cluster.

## 12. Reproducibility

All reported values are read from persisted experiment outputs. Random seeds are fixed in `params_phase2.yaml`. The recommendation is regenerated by running:

    python src/phase2_completion_step5.py

## 13. Generated Artifacts

- `phase2_unified_performance_table.csv`
- `phase2_final_scoreboard.csv`
- `phase2_final_recommendation.json`
- `phase2_integrated_report.md`
- `phase2_integrated_report.pdf`
"""

    OUTPUT_MARKDOWN.write_text(
        report,
        encoding="utf-8",
    )

    return report


def wrap_text(
    value: str,
    width: int = 105,
) -> str:
    lines = []

    for paragraph in value.splitlines():
        if not paragraph.strip():
            lines.append("")
            continue

        wrapped = textwrap.wrap(
            paragraph,
            width=width,
            replace_whitespace=False,
            drop_whitespace=True,
        )

        if wrapped:
            lines.extend(wrapped)
        else:
            lines.append("")

    return "\n".join(lines)


def add_text_pages(
    pdf: PdfPages,
    title: str,
    text: str,
) -> None:
    wrapped = wrap_text(
        text,
        width=105,
    )

    lines = wrapped.splitlines()

    maximum_lines = 52

    chunks = [
        lines[position:position + maximum_lines]
        for position in range(
            0,
            len(lines),
            maximum_lines,
        )
    ]

    if not chunks:
        chunks = [[]]

    for page_number, chunk in enumerate(
        chunks,
        start=1,
    ):
        figure = plt.figure(
            figsize=(8.27, 11.69)
        )

        display_title = (
            title
            if page_number == 1
            else f"{title} — continued"
        )

        figure.text(
            0.07,
            0.95,
            display_title,
            fontsize=16,
            weight="bold",
            va="top",
        )

        figure.text(
            0.07,
            0.91,
            "\n".join(chunk),
            fontsize=8.8,
            family="monospace",
            va="top",
        )

        pdf.savefig(
            figure,
            bbox_inches="tight",
        )

        plt.close(figure)


def add_table_pages(
    pdf: PdfPages,
    title: str,
    frame: pd.DataFrame,
    columns: list[str],
    rows_per_page: int = 18,
) -> None:
    if frame.empty:
        add_text_pages(
            pdf,
            title,
            "No data available.",
        )
        return

    available = [
        column
        for column in columns
        if column in frame.columns
    ]

    display = frame[
        available
    ].copy()

    for column in display.select_dtypes(
        include=[np.number]
    ).columns:
        display[column] = display[
            column
        ].map(
            lambda value: (
                f"{value:.4f}"
                if pd.notna(value)
                else ""
            )
        )

    page_count = max(
        1,
        math.ceil(
            len(display)
            / rows_per_page
        ),
    )

    for page in range(page_count):
        start = page * rows_per_page
        end = start + rows_per_page

        chunk = display.iloc[
            start:end
        ]

        figure, axis = plt.subplots(
            figsize=(11.69, 8.27)
        )

        axis.axis("off")

        axis.set_title(
            (
                title
                if page_count == 1
                else (
                    f"{title} "
                    f"({page + 1}/{page_count})"
                )
            ),
            fontsize=15,
            pad=20,
        )

        table = axis.table(
            cellText=chunk.astype(
                str
            ).values,
            colLabels=chunk.columns,
            cellLoc="center",
            colLoc="center",
            loc="center",
        )

        table.auto_set_font_size(False)
        table.set_fontsize(7)
        table.scale(1, 1.5)

        try:
            table.auto_set_column_width(
                col=list(
                    range(
                        len(chunk.columns)
                    )
                )
            )
        except Exception:
            pass

        pdf.savefig(
            figure,
            bbox_inches="tight",
        )

        plt.close(figure)


def add_image_page(
    pdf: PdfPages,
    path: Path,
    title: str,
) -> None:
    if not path.exists():
        return

    try:
        image = mpimg.imread(path)
    except Exception:
        return

    figure = plt.figure(
        figsize=(11.69, 8.27)
    )

    axis = figure.add_subplot(111)

    axis.imshow(image)
    axis.axis("off")
    axis.set_title(
        title,
        fontsize=15,
        pad=18,
    )

    pdf.savefig(
        figure,
        bbox_inches="tight",
    )

    plt.close(figure)


def build_pdf_report(
    scoreboard: pd.DataFrame,
    recommendation: dict[str, Any],
) -> None:
    final_comparison = read_csv_optional(
        FINAL_COMPARISON_PATH
    )

    kmeans = read_csv_optional(
        REPORT_DIR / "kmeans_search.csv"
    )

    bootstrap = read_csv_optional(
        REPORT_DIR
        / "bootstrap_k_stability_summary.csv"
    )

    metric_sensitivity = read_csv_optional(
        REPORT_DIR
        / "distance_metric_selected_results.csv"
    )

    recommendation_text = (
        recommendation_to_text(
            recommendation
        )
        + "\n\nReasons:\n"
        + "\n".join(
            "- " + value
            for value in recommendation[
                "reasons"
            ]
        )
        + "\n\nLimitations:\n"
        + "\n".join(
            "- " + value
            for value in recommendation[
                "limitations"
            ]
        )
    )

    figure_specs = [
        (
            FIGURE_DIR
            / "kmeans_k_selection.png",
            "K-Means Determining-k Study",
        ),
        (
            FIGURE_DIR
            / "gap_statistic.png",
            "Gap Statistic",
        ),
        (
            FIGURE_DIR
            / "hierarchical_cut_strategy_dendrograms.png",
            "Hierarchical Cutting Strategies",
        ),
        (
            FIGURE_DIR
            / "dbscan_k_distance_10.png",
            "DBSCAN k-Distance Diagnostic",
        ),
        (
            FIGURE_DIR
            / "gmm_bic_aic.png",
            "GMM BIC and AIC",
        ),
        (
            FIGURE_DIR
            / "algorithm_agreement_heatmap.png",
            "Algorithm Agreement",
        ),
        (
            FIGURE_DIR
            / "stability_comparison.png",
            "Seed and Bootstrap Stability",
        ),
        (
            FIGURE_DIR
            / "bootstrap_stability_by_k.png",
            "Bootstrap Stability by k",
        ),
        (
            FIGURE_DIR
            / "bootstrap_coclustering_heatmap.png",
            "Bootstrap Co-clustering Matrix",
        ),
        (
            FIGURE_DIR
            / "preferred_model_error_analysis.png",
            "Preferred Model Error Analysis",
        ),
        (
            FIGURE_DIR
            / "distance_metric_search_comparison.png",
            "Distance-Metric Search",
        ),
        (
            FIGURE_DIR
            / "distance_metric_partition_agreement.png",
            "Distance-Metric Agreement",
        ),
        (
            FIGURE_DIR
            / "final_metric_comparison.png",
            "Final Algorithm Comparison",
        ),
    ]

    with PdfPages(
        OUTPUT_PDF
    ) as pdf:
        add_text_pages(
            pdf,
            "Phase 2 Integrated Clustering Report",
            (
                "Credit Card Transaction Clustering\n\n"
                "This report compares partitioning, "
                "hierarchical, density-based, and "
                "model-based clustering algorithms. "
                "Model selection uses only internal "
                "metrics. Fraud labels are used only "
                "for post-hoc external evaluation.\n\n"
                + recommendation_text
            ),
        )

        add_table_pages(
            pdf,
            "Final Algorithm Comparison",
            final_comparison,
            columns=[
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
            ],
        )

        add_table_pages(
            pdf,
            "Internal Ranking Scoreboard",
            scoreboard,
            columns=[
                "final_order",
                "algorithm",
                "parameters",
                "selected_k",
                "silhouette",
                "davies_bouldin",
                "calinski_harabasz",
                "seed_ari_mean",
                "internal_rank_sum",
            ],
        )

        add_table_pages(
            pdf,
            "K-Means Search",
            kmeans,
            columns=[
                "k",
                "inertia",
                "silhouette",
                "davies_bouldin",
                "calinski_harabasz",
                "runtime_seconds",
            ],
        )

        add_table_pages(
            pdf,
            "Bootstrap Stability",
            bootstrap,
            columns=[
                "k",
                "bootstrap_runs",
                "ari_mean",
                "ari_std",
                "mean_jaccard",
                "jaccard_std",
                "worst_cluster_mean_jaccard",
            ],
        )

        add_table_pages(
            pdf,
            "Distance-Metric Sensitivity",
            metric_sensitivity,
            columns=[
                "distance_metric",
                "requested_k",
                "actual_k",
                "silhouette_matching_metric",
                "davies_bouldin",
                "calinski_harabasz",
                "ari_external",
                "runtime_seconds",
            ],
        )

        for path, title in figure_specs:
            add_image_page(
                pdf,
                path,
                title,
            )

        metadata = pdf.infodict()

        metadata["Title"] = (
            "Phase 2 Integrated Clustering Report"
        )

        metadata["Author"] = (
            "Credit Card Clustering Project"
        )

        metadata["Subject"] = (
            "Core clustering algorithms, evaluation, "
            "stability, and recommendation"
        )

        metadata["Keywords"] = (
            "clustering, K-Means, DBSCAN, GMM, "
            "hierarchical, stability"
        )

    logging.info(
        "Phase 2 PDF saved: %s",
        OUTPUT_PDF,
    )


def create_summary_figure(
    scoreboard: pd.DataFrame,
) -> Path:
    metrics = [
        "silhouette",
        "davies_bouldin",
        "calinski_harabasz",
    ]

    available = [
        metric
        for metric in metrics
        if metric in scoreboard.columns
    ]

    melted = scoreboard.melt(
        id_vars=["algorithm"],
        value_vars=available,
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

    graph.set_xticklabels(
        rotation=25
    )

    graph.fig.suptitle(
        "Phase 2 Internal Metric Scoreboard",
        y=1.03,
    )

    path = (
        FIGURE_DIR
        / "phase2_internal_scoreboard.png"
    )

    graph.savefig(
        path,
        bbox_inches="tight",
        dpi=200,
    )

    plt.close("all")

    return path


def main() -> None:
    ensure_directories()
    setup_logging()
    setup_plots()

    unified = (
        build_unified_performance_table()
    )

    final_comparison = (
        load_final_comparison()
    )

    scoreboard = (
        calculate_internal_scoreboard(
            final_comparison
        )
    )

    recommendation = (
        build_recommendation(
            scoreboard
        )
    )

    summary_figure = (
        create_summary_figure(
            scoreboard
        )
    )

    build_markdown_report(
        unified=unified,
        scoreboard=scoreboard,
        recommendation=recommendation,
    )

    build_pdf_report(
        scoreboard=scoreboard,
        recommendation=recommendation,
    )

    completion_record = {
        "status": "completed",
        "recommended_algorithm": (
            recommendation[
                "recommended_algorithm"
            ]
        ),
        "recommended_parameters": (
            recommendation[
                "recommended_parameters"
            ]
        ),
        "recommended_k": (
            recommendation[
                "recommended_k"
            ]
        ),
        "selection_policy": (
            recommendation[
                "selection_policy"
            ]
        ),
        "unified_experiment_rows": int(
            len(unified)
        ),
        "scoreboard_rows": int(
            len(scoreboard)
        ),
        "generated_files": [
            str(
                path.relative_to(ROOT)
            )
            for path in [
                OUTPUT_PERFORMANCE_TABLE,
                OUTPUT_SCOREBOARD,
                OUTPUT_RECOMMENDATION,
                OUTPUT_MARKDOWN,
                OUTPUT_PDF,
                summary_figure,
            ]
        ],
        "generated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
    }

    save_json(
        completion_record,
        REPORT_DIR
        / "phase2_completion_step5_record.json",
    )

    logging.info(
        "Phase 2 integrated report completed"
    )

    logging.info(
        "Recommended algorithm: %s",
        recommendation[
            "recommended_algorithm"
        ],
    )

    logging.info(
        "Recommended parameters: %s",
        recommendation[
            "recommended_parameters"
        ],
    )


if __name__ == "__main__":
    main()