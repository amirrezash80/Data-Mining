from __future__ import annotations

import argparse
import hashlib
import json
import logging
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns
import yaml
from scipy.special import expit, logsumexp
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import RobustScaler, StandardScaler
from tqdm.auto import tqdm

try:
    import umap
except ImportError:
    umap = None


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_INTERIM = ROOT / "data" / "interim"
DATA_PROCESSED = ROOT / "data" / "processed"
REPORT_PHASE1 = ROOT / "reports" / "phase1"
REPORT_FIGURES = ROOT / "reports" / "figures"
MODELS = ROOT / "models"

PARAMS_PATH = ROOT / "params.yaml"


# Public mirrors of the original ULB credit-card fraud dataset.
# If one mirror fails, the next one is attempted.
DATA_URLS = [
    "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv",
    (
        "https://raw.githubusercontent.com/"
        "gastonstat/CreditScoring/master/creditcard.csv"
    ),
]

EXPECTED_COLUMNS = (
    ["Time"]
    + [f"V{i}" for i in range(1, 29)]
    + ["Amount", "Class"]
)


# ---------------------------------------------------------------------
# General utilities
# ---------------------------------------------------------------------

def ensure_directories() -> None:
    for path in [
        DATA_RAW,
        DATA_INTERIM,
        DATA_PROCESSED,
        REPORT_PHASE1,
        REPORT_FIGURES,
        MODELS,
    ]:
        path.mkdir(parents=True, exist_ok=True)


def setup_logging() -> None:
    log_path = REPORT_PHASE1 / "phase1_execution.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_path, mode="w", encoding="utf-8"),
        ],
    )


def load_params() -> dict[str, Any]:
    if not PARAMS_PATH.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {PARAMS_PATH}"
        )

    with PARAMS_PATH.open("r", encoding="utf-8") as file:
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


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        while True:
            chunk = file.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)

    return digest.hexdigest()


def set_plot_style() -> None:
    sns.set_theme(
        style="whitegrid",
        context="notebook",
        palette="colorblind",
    )

    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 200,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "figure.autolayout": True,
        }
    )


def save_figure(filename: str) -> None:
    path = REPORT_FIGURES / filename
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    logging.info("Saved figure: %s", path)


# ---------------------------------------------------------------------
# Data acquisition
# ---------------------------------------------------------------------

def download_file(
    urls: list[str],
    destination: Path,
    force: bool = False,
    timeout: int = 60,
) -> Path:
    if destination.exists() and destination.stat().st_size > 0 and not force:
        logging.info(
            "Raw file already exists; download skipped: %s",
            destination,
        )
        return destination

    temporary_path = destination.with_suffix(".download")

    for url in urls:
        logging.info("Attempting download from: %s", url)

        try:
            with requests.get(
                url,
                stream=True,
                timeout=timeout,
                headers={"User-Agent": "credit-card-clustering-project/1.0"},
            ) as response:
                response.raise_for_status()

                total = int(response.headers.get("content-length", 0))

                with temporary_path.open("wb") as file:
                    progress = tqdm(
                        total=total,
                        unit="B",
                        unit_scale=True,
                        desc="Downloading",
                    )

                    for chunk in response.iter_content(
                        chunk_size=1024 * 1024
                    ):
                        if chunk:
                            file.write(chunk)
                            progress.update(len(chunk))

                    progress.close()

            if temporary_path.stat().st_size < 1_000_000:
                raise ValueError(
                    "Downloaded file is unexpectedly small."
                )

            temporary_path.replace(destination)
            logging.info("Dataset downloaded to: %s", destination)
            return destination

        except Exception as exc:
            logging.warning("Download failed from %s: %s", url, exc)

            if temporary_path.exists():
                temporary_path.unlink()

    raise RuntimeError(
        "Dataset download failed from all configured URLs. "
        "Download creditcard.csv manually and place it in data/raw/."
    )


# ---------------------------------------------------------------------
# Validation and profiling
# ---------------------------------------------------------------------

def validate_raw_schema(df: pd.DataFrame) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []

    missing_columns = sorted(set(EXPECTED_COLUMNS) - set(df.columns))
    unexpected_columns = sorted(set(df.columns) - set(EXPECTED_COLUMNS))

    if missing_columns:
        errors.append(f"Missing columns: {missing_columns}")

    if unexpected_columns:
        warnings.append(f"Unexpected columns: {unexpected_columns}")

    if len(df) == 0:
        errors.append("Dataset is empty.")

    if "Class" in df.columns:
        class_values = set(
            pd.to_numeric(df["Class"], errors="coerce")
            .dropna()
            .astype(int)
            .unique()
            .tolist()
        )

        if not class_values.issubset({0, 1}):
            errors.append(
                f"Class contains values outside {{0, 1}}: {class_values}"
            )

    numeric_columns = [
        column
        for column in EXPECTED_COLUMNS
        if column in df.columns
    ]

    non_numeric_after_coercion: dict[str, int] = {}

    for column in numeric_columns:
        coerced = pd.to_numeric(df[column], errors="coerce")
        introduced_nan = int(
            coerced.isna().sum() - df[column].isna().sum()
        )

        if introduced_nan > 0:
            non_numeric_after_coercion[column] = introduced_nan

    if non_numeric_after_coercion:
        errors.append(
            "Non-numeric values found in numeric columns: "
            f"{non_numeric_after_coercion}"
        )

    if "Amount" in df.columns:
        negative_amounts = int(
            (pd.to_numeric(df["Amount"], errors="coerce") < 0).sum()
        )

        if negative_amounts:
            errors.append(
                f"Negative Amount records found: {negative_amounts}"
            )

    if "Time" in df.columns:
        negative_times = int(
            (pd.to_numeric(df["Time"], errors="coerce") < 0).sum()
        )

        if negative_times:
            errors.append(
                f"Negative Time records found: {negative_times}"
            )

    result = {
        "n_rows": int(df.shape[0]),
        "n_columns": int(df.shape[1]),
        "expected_columns": EXPECTED_COLUMNS,
        "missing_columns": missing_columns,
        "unexpected_columns": unexpected_columns,
        "errors": errors,
        "warnings": warnings,
        "valid": len(errors) == 0,
    }

    if errors:
        raise ValueError(
            "Raw-data validation failed:\n- " + "\n- ".join(errors)
        )

    return result


def create_data_profile(df: pd.DataFrame) -> pd.DataFrame:
    records = []

    for column in df.columns:
        series = df[column]

        records.append(
            {
                "column": column,
                "dtype": str(series.dtype),
                "n_rows": len(series),
                "missing_count": int(series.isna().sum()),
                "missing_percent": float(
                    100 * series.isna().mean()
                ),
                "unique_count": int(series.nunique(dropna=True)),
                "duplicate_count": int(series.duplicated().sum()),
                "minimum": (
                    float(series.min())
                    if pd.api.types.is_numeric_dtype(series)
                    else None
                ),
                "maximum": (
                    float(series.max())
                    if pd.api.types.is_numeric_dtype(series)
                    else None
                ),
                "mean": (
                    float(series.mean())
                    if pd.api.types.is_numeric_dtype(series)
                    else None
                ),
                "median": (
                    float(series.median())
                    if pd.api.types.is_numeric_dtype(series)
                    else None
                ),
                "std": (
                    float(series.std())
                    if pd.api.types.is_numeric_dtype(series)
                    else None
                ),
                "skewness": (
                    float(series.skew())
                    if pd.api.types.is_numeric_dtype(series)
                    else None
                ),
            }
        )

    return pd.DataFrame(records)


# ---------------------------------------------------------------------
# Cleaning and feature engineering
# ---------------------------------------------------------------------

def clean_data(
    df: pd.DataFrame,
    remove_exact_duplicates: bool,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    cleaned = df.copy()

    initial_rows = len(cleaned)
    initial_missing = int(cleaned.isna().sum().sum())
    initial_duplicates = int(cleaned.duplicated().sum())

    for column in EXPECTED_COLUMNS:
        cleaned[column] = pd.to_numeric(
            cleaned[column],
            errors="coerce",
        )

    missing_by_column = cleaned.isna().sum()
    columns_with_missing = missing_by_column[
        missing_by_column > 0
    ].to_dict()

    # The official dataset is expected to have no missing values.
    # Rows with malformed or missing values are removed rather than
    # being assigned an arbitrary imputation that could create clusters.
    rows_before_missing_removal = len(cleaned)
    cleaned = cleaned.dropna(subset=EXPECTED_COLUMNS)
    removed_missing_rows = rows_before_missing_removal - len(cleaned)

    invalid_amount_mask = cleaned["Amount"] < 0
    invalid_time_mask = cleaned["Time"] < 0
    invalid_class_mask = ~cleaned["Class"].isin([0, 1])

    invalid_amount_count = int(invalid_amount_mask.sum())
    invalid_time_count = int(invalid_time_mask.sum())
    invalid_class_count = int(invalid_class_mask.sum())

    cleaned = cleaned.loc[
        ~invalid_amount_mask
        & ~invalid_time_mask
        & ~invalid_class_mask
    ].copy()

    removed_duplicates = 0

    if remove_exact_duplicates:
        before = len(cleaned)
        cleaned = cleaned.drop_duplicates(
            subset=EXPECTED_COLUMNS,
            keep="first",
        ).copy()
        removed_duplicates = before - len(cleaned)

    cleaned["Class"] = cleaned["Class"].astype("int8")
    cleaned = cleaned.sort_values(
        ["Time"],
        kind="stable",
    ).reset_index(drop=True)

    cleaned.insert(
        0,
        "row_id",
        np.arange(len(cleaned), dtype=np.int64),
    )

    decision_log = {
        "initial_rows": int(initial_rows),
        "final_rows": int(len(cleaned)),
        "initial_missing_cells": int(initial_missing),
        "missing_by_column": {
            key: int(value)
            for key, value in columns_with_missing.items()
        },
        "removed_rows_with_missing_or_malformed_values": int(
            removed_missing_rows
        ),
        "initial_exact_duplicate_rows": int(initial_duplicates),
        "remove_exact_duplicates": bool(remove_exact_duplicates),
        "removed_exact_duplicates": int(removed_duplicates),
        "removed_negative_amount_rows": invalid_amount_count,
        "removed_negative_time_rows": invalid_time_count,
        "removed_invalid_class_rows": invalid_class_count,
        "sorting": "Ascending by Time using stable sort.",
        "label_policy": (
            "Class is retained only for external evaluation and is never "
            "included in scaling, PCA, UMAP fitting, Hopkins, VAT, or "
            "clustering input."
        ),
        "missing_value_policy": (
            "No arbitrary imputation is used. Malformed/missing records "
            "are removed and counted because artificial fills can create "
            "spurious distance structure."
        ),
        "duplicate_policy": (
            "Exact duplicates are removed when enabled because duplicate "
            "records artificially inflate local density. A later "
            "sensitivity analysis should also rerun the final clustering "
            "without duplicate removal."
        ),
        "genuine_outlier_policy": (
            "Large but valid transactions and unusual PCA-feature values "
            "are retained because genuine anomalies are central to the "
            "research question."
        ),
    }

    return cleaned, decision_log


def build_features(
    df: pd.DataFrame,
    params: dict[str, Any],
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    feature_config = params["features"]

    features = pd.DataFrame(index=df.index)
    metadata = df[["row_id", "Time", "Amount", "Class"]].copy()

    if feature_config["include_v_features"]:
        for index in range(1, 29):
            column = f"V{index}"
            features[column] = df[column].astype("float64")

    if feature_config["include_log_amount"]:
        features["Amount_log1p"] = np.log1p(
            df["Amount"].astype("float64")
        )

    if feature_config["include_cyclic_time"]:
        seconds_per_day = 24 * 60 * 60
        phase = 2 * np.pi * (
            df["Time"].astype("float64") % seconds_per_day
        ) / seconds_per_day

        features["Time_sin"] = np.sin(phase)
        features["Time_cos"] = np.cos(phase)

    if feature_config.get("include_elapsed_day", False):
        features["Elapsed_day"] = (
            df["Time"].astype("float64") / 86400.0
        )

    labels = df["Class"].copy()

    if "Class" in features.columns:
        raise RuntimeError("Label leakage detected: Class in feature matrix.")

    if features.isna().any().any():
        raise ValueError("Engineered feature matrix contains missing values.")

    return features, labels, metadata


def temporal_split(
    features: pd.DataFrame,
    metadata: pd.DataFrame,
    train_fraction: float,
) -> tuple[np.ndarray, np.ndarray]:
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1.")

    order = np.argsort(
        metadata["Time"].to_numpy(),
        kind="stable",
    )

    cutoff = int(len(order) * train_fraction)
    train_indices = order[:cutoff]
    test_indices = order[cutoff:]

    return train_indices, test_indices


# ---------------------------------------------------------------------
# Scaling and PCA
# ---------------------------------------------------------------------

def fit_scalers(
    features: pd.DataFrame,
    train_indices: np.ndarray,
) -> dict[str, dict[str, Any]]:
    x = features.to_numpy(dtype=np.float64)
    x_train = x[train_indices]

    scalers = {
        "standard": StandardScaler(),
        "robust": RobustScaler(
            with_centering=True,
            with_scaling=True,
            quantile_range=(25.0, 75.0),
        ),
    }

    outputs: dict[str, dict[str, Any]] = {}

    for name, scaler in scalers.items():
        logging.info("Fitting %s scaler on training partition.", name)

        scaler.fit(x_train)
        transformed = scaler.transform(x)

        outputs[name] = {
            "scaler": scaler,
            "all": transformed,
            "train": transformed[train_indices],
        }

        joblib.dump(
            scaler,
            MODELS / f"{name}_scaler.joblib",
        )

    return outputs


def fit_pca(
    x_train: np.ndarray,
    x_all: np.ndarray,
    variance_threshold: float,
    max_components: int,
    random_seed: int,
) -> tuple[PCA, np.ndarray, np.ndarray, pd.DataFrame]:
    logging.info("Fitting diagnostic PCA.")

    diagnostic_pca = PCA(
        n_components=None,
        svd_solver="full",
    )
    diagnostic_pca.fit(x_train)

    cumulative_variance = np.cumsum(
        diagnostic_pca.explained_variance_ratio_
    )

    selected_components = int(
        np.searchsorted(
            cumulative_variance,
            variance_threshold,
        )
        + 1
    )

    selected_components = min(
        selected_components,
        max_components,
        x_train.shape[1],
    )

    logging.info(
        "Selected %d PCA components for variance threshold %.3f.",
        selected_components,
        variance_threshold,
    )

    pca = PCA(
        n_components=selected_components,
        svd_solver="randomized",
        random_state=random_seed,
    )

    x_train_pca = pca.fit_transform(x_train)
    x_all_pca = pca.transform(x_all)

    diagnostics = pd.DataFrame(
        {
            "component": np.arange(
                1,
                len(diagnostic_pca.explained_variance_ratio_) + 1,
            ),
            "explained_variance_ratio": (
                diagnostic_pca.explained_variance_ratio_
            ),
            "cumulative_explained_variance": cumulative_variance,
        }
    )

    joblib.dump(pca, MODELS / "pca_standard.joblib")

    return (
        pca,
        x_train_pca,
        x_all_pca,
        diagnostics,
    )


# ---------------------------------------------------------------------
# Hopkins statistic
# ---------------------------------------------------------------------

def hopkins_statistic(
    x: np.ndarray,
    m: int,
    random_state: int,
    power: int | None = None,
) -> float:
    """
    Compute Hopkins statistic.

    The project specification presents distances raised to dimension d.
    Direct exponentiation may overflow in high-dimensional data, so this
    implementation computes the ratio in log-space.
    """
    rng = np.random.default_rng(random_state)

    n_samples, n_features = x.shape

    if n_samples < 3:
        raise ValueError("At least three observations are required.")

    m = min(m, n_samples - 1)
    exponent = n_features if power is None else power

    sampled_indices = rng.choice(
        n_samples,
        size=m,
        replace=False,
    )
    real_sample = x[sampled_indices]

    minimum = np.min(x, axis=0)
    maximum = np.max(x, axis=0)

    uniform_sample = rng.uniform(
        minimum,
        maximum,
        size=(m, n_features),
    )

    neighbors = NearestNeighbors(
        n_neighbors=2,
        metric="euclidean",
        n_jobs=-1,
    )
    neighbors.fit(x)

    real_distances, _ = neighbors.kneighbors(
        real_sample,
        n_neighbors=2,
    )
    random_distances, _ = neighbors.kneighbors(
        uniform_sample,
        n_neighbors=1,
    )

    # For real observations, first neighbour is the point itself.
    w = np.maximum(real_distances[:, 1], np.finfo(float).tiny)
    u = np.maximum(random_distances[:, 0], np.finfo(float).tiny)

    log_sum_u = logsumexp(exponent * np.log(u))
    log_sum_w = logsumexp(exponent * np.log(w))

    return float(expit(log_sum_u - log_sum_w))


def repeated_hopkins(
    x: np.ndarray,
    m: int,
    repetitions: int,
    base_seed: int,
) -> pd.DataFrame:
    rows = []

    for repetition in range(repetitions):
        seed = base_seed + repetition

        value = hopkins_statistic(
            x=x,
            m=m,
            random_state=seed,
            power=x.shape[1],
        )

        rows.append(
            {
                "repetition": repetition + 1,
                "seed": seed,
                "sample_size_m": min(m, len(x) - 1),
                "dimension_power": x.shape[1],
                "hopkins": value,
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# VAT
# ---------------------------------------------------------------------

def vat_ordering(distance_matrix: np.ndarray) -> np.ndarray:
    """
    Produce a VAT-style ordering.

    Starts from a point participating in a largest dissimilarity and
    repeatedly adds the unselected point nearest to the selected set.
    """
    n = distance_matrix.shape[0]

    if distance_matrix.shape != (n, n):
        raise ValueError("Distance matrix must be square.")

    row_maxima = distance_matrix.max(axis=1)
    current = int(np.argmax(row_maxima))

    selected = [current]
    remaining = np.ones(n, dtype=bool)
    remaining[current] = False

    minimum_distance_to_selected = distance_matrix[current].copy()
    minimum_distance_to_selected[current] = np.inf

    for _ in range(n - 1):
        candidate_scores = np.where(
            remaining,
            minimum_distance_to_selected,
            np.inf,
        )
        next_index = int(np.argmin(candidate_scores))

        selected.append(next_index)
        remaining[next_index] = False

        minimum_distance_to_selected = np.minimum(
            minimum_distance_to_selected,
            distance_matrix[next_index],
        )
        minimum_distance_to_selected[~remaining] = np.inf

    return np.asarray(selected, dtype=int)


def compute_vat(
    x: np.ndarray,
    sample_size: int,
    random_seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(random_seed)
    sample_size = min(sample_size, len(x))

    indices = rng.choice(
        len(x),
        size=sample_size,
        replace=False,
    )

    sample = x[indices]

    distance_matrix = squareform(
        pdist(sample, metric="euclidean")
    )

    ordering = vat_ordering(distance_matrix)

    reordered = distance_matrix[
        np.ix_(ordering, ordering)
    ]

    return reordered, indices[ordering]


# ---------------------------------------------------------------------
# Metric diagnostics
# ---------------------------------------------------------------------

def metric_diagnostics(
    x: np.ndarray,
    sample_size: int,
    random_seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_seed)
    sample_size = min(sample_size, len(x))

    indices = rng.choice(
        len(x),
        size=sample_size,
        replace=False,
    )
    sample = x[indices]

    metrics = {
        "euclidean": {"metric": "euclidean"},
        "manhattan": {"metric": "manhattan"},
    }

    rows = []

    for name, settings in metrics.items():
        nn = NearestNeighbors(
            n_neighbors=6,
            n_jobs=-1,
            **settings,
        ).fit(sample)

        distances, _ = nn.kneighbors(sample)

        # Exclude self-distance at column zero.
        nearest = distances[:, 1]
        fifth_nearest = distances[:, 5]

        rows.append(
            {
                "metric": name,
                "sample_size": sample_size,
                "nearest_neighbor_mean": float(nearest.mean()),
                "nearest_neighbor_median": float(
                    np.median(nearest)
                ),
                "nearest_neighbor_q95": float(
                    np.quantile(nearest, 0.95)
                ),
                "fifth_neighbor_mean": float(
                    fifth_nearest.mean()
                ),
                "fifth_neighbor_median": float(
                    np.median(fifth_nearest)
                ),
                "fifth_neighbor_q95": float(
                    np.quantile(fifth_nearest, 0.95)
                ),
            }
        )

    # Mahalanobis is evaluated on the sample with a regularised
    # covariance inverse. It is retained as a diagnostic rather than
    # chosen automatically.
    covariance = np.cov(sample, rowvar=False)
    regularization = 1e-6 * np.eye(covariance.shape[0])
    inverse_covariance = np.linalg.pinv(
        covariance + regularization
    )

    # Use a smaller subset for pairwise Mahalanobis due to cost.
    mah_size = min(1500, sample_size)
    mah_sample = sample[:mah_size]

    mah_distances = pairwise_distances(
        mah_sample,
        metric="mahalanobis",
        VI=inverse_covariance,
        n_jobs=-1,
    )

    np.fill_diagonal(mah_distances, np.inf)
    sorted_distances = np.sort(mah_distances, axis=1)

    rows.append(
        {
            "metric": "mahalanobis",
            "sample_size": mah_size,
            "nearest_neighbor_mean": float(
                sorted_distances[:, 0].mean()
            ),
            "nearest_neighbor_median": float(
                np.median(sorted_distances[:, 0])
            ),
            "nearest_neighbor_q95": float(
                np.quantile(sorted_distances[:, 0], 0.95)
            ),
            "fifth_neighbor_mean": float(
                sorted_distances[:, 4].mean()
            ),
            "fifth_neighbor_median": float(
                np.median(sorted_distances[:, 4])
            ),
            "fifth_neighbor_q95": float(
                np.quantile(sorted_distances[:, 4], 0.95)
            ),
        }
    )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------

def plot_class_distribution(df: pd.DataFrame) -> None:
    counts = df["Class"].value_counts().sort_index()

    plt.figure(figsize=(7, 5))
    ax = sns.barplot(
        x=counts.index.astype(str),
        y=counts.values,
    )
    ax.set_yscale("log")
    ax.set_xlabel("Class — external evaluation only")
    ax.set_ylabel("Number of records, logarithmic scale")
    ax.set_title("Class distribution—not used to fit representations")

    for index, value in enumerate(counts.values):
        ax.text(
            index,
            value,
            f"{value:,}",
            ha="center",
            va="bottom",
        )

    save_figure("phase1_class_distribution.png")


def plot_amount_transform(df: pd.DataFrame) -> None:
    sample = df.sample(
        min(50000, len(df)),
        random_state=42,
    ).copy()

    sample["Amount_log1p"] = np.log1p(sample["Amount"])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    sns.histplot(
        sample["Amount"],
        bins=80,
        ax=axes[0],
        color="#4472C4",
    )
    axes[0].set_title("Raw transaction Amount")
    axes[0].set_xlabel("Amount")

    sns.histplot(
        sample["Amount_log1p"],
        bins=80,
        ax=axes[1],
        color="#ED7D31",
    )
    axes[1].set_title("Amount after log1p transformation")
    axes[1].set_xlabel("log(1 + Amount)")

    save_figure("phase1_amount_log_transform.png")


def plot_feature_distributions(features: pd.DataFrame) -> None:
    columns = [
        column
        for column in [
            "V1",
            "V2",
            "V3",
            "V4",
            "V10",
            "V11",
            "V12",
            "V14",
            "V17",
            "Amount_log1p",
            "Time_sin",
            "Time_cos",
        ]
        if column in features.columns
    ]

    sample = features.sample(
        min(50000, len(features)),
        random_state=42,
    )

    n_columns = 3
    n_rows = int(np.ceil(len(columns) / n_columns))

    fig, axes = plt.subplots(
        n_rows,
        n_columns,
        figsize=(14, 3.2 * n_rows),
    )
    axes = np.asarray(axes).reshape(-1)

    for axis, column in zip(axes, columns):
        sns.histplot(
            sample[column],
            bins=60,
            kde=True,
            ax=axis,
        )
        axis.set_title(column)

    for axis in axes[len(columns):]:
        axis.axis("off")

    save_figure("phase1_univariate_distributions.png")


def plot_correlation_heatmap(features: pd.DataFrame) -> None:
    correlations = features.corr(method="pearson")

    plt.figure(figsize=(14, 12))
    sns.heatmap(
        correlations,
        cmap="vlag",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        xticklabels=True,
        yticklabels=True,
        cbar_kws={"label": "Pearson correlation"},
    )
    plt.title("Correlation matrix of engineered clustering features")
    save_figure("phase1_feature_correlation.png")


def plot_scaling_comparison(
    features: pd.DataFrame,
    scaler_outputs: dict[str, dict[str, Any]],
) -> None:
    display_columns = [
        column
        for column in ["V1", "Amount_log1p"]
        if column in features.columns
    ]

    if len(display_columns) < 2:
        display_columns = list(features.columns[:2])

    indices = [
        features.columns.get_loc(column)
        for column in display_columns
    ]

    rng = np.random.default_rng(42)
    sample_indices = rng.choice(
        len(features),
        size=min(20000, len(features)),
        replace=False,
    )

    raw = features.iloc[sample_indices, indices].to_numpy()
    standard = scaler_outputs["standard"]["all"][
        sample_indices
    ][:, indices]
    robust = scaler_outputs["robust"]["all"][
        sample_indices
    ][:, indices]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    axes[0].scatter(
        raw[:, 0],
        raw[:, 1],
        alpha=0.12,
        s=5,
        rasterized=True,
    )
    axes[0].set_title("Before scaling")

    axes[1].scatter(
        standard[:, 0],
        standard[:, 1],
        alpha=0.12,
        s=5,
        rasterized=True,
    )
    axes[1].set_title("StandardScaler")

    axes[2].scatter(
        robust[:, 0],
        robust[:, 1],
        alpha=0.12,
        s=5,
        rasterized=True,
    )
    axes[2].set_title("RobustScaler")

    for axis in axes:
        axis.set_xlabel(display_columns[0])
        axis.set_ylabel(display_columns[1])

    save_figure("phase1_scaling_before_after.png")


def plot_pca_diagnostics(
    diagnostics: pd.DataFrame,
    selected_components: int,
) -> None:
    plt.figure(figsize=(8, 5))

    plt.plot(
        diagnostics["component"],
        diagnostics["cumulative_explained_variance"],
        marker="o",
        markersize=3,
    )

    plt.axhline(
        0.95,
        color="red",
        linestyle="--",
        label="95% variance",
    )
    plt.axvline(
        selected_components,
        color="black",
        linestyle=":",
        label=f"Selected components = {selected_components}",
    )

    plt.xlabel("Number of principal components")
    plt.ylabel("Cumulative explained variance")
    plt.title("PCA cumulative explained variance")
    plt.legend()

    save_figure("phase1_pca_explained_variance.png")


def plot_pca_density(
    x_pca: np.ndarray,
    sample_indices: np.ndarray,
) -> None:
    sample = x_pca[sample_indices]

    plt.figure(figsize=(8, 7))
    plt.hexbin(
        sample[:, 0],
        sample[:, 1],
        gridsize=70,
        cmap="viridis",
        mincnt=1,
        bins="log",
    )
    plt.colorbar(label="log-scaled point count")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA projection—density view")
    save_figure("phase1_pca_density.png")


def fit_and_plot_umap(
    x: np.ndarray,
    metadata: pd.DataFrame,
    sample_indices: np.ndarray,
    params: dict[str, Any],
    random_seed: int,
) -> pd.DataFrame:
    if umap is None:
        raise ImportError(
            "umap-learn is not installed. Run: pip install umap-learn"
        )

    config = params["umap"]
    sample = x[sample_indices]

    logging.info(
        "Fitting UMAP on %d records.",
        len(sample_indices),
    )

    reducer = umap.UMAP(
        n_neighbors=config["n_neighbors"],
        min_dist=config["min_dist"],
        n_components=config["n_components"],
        metric=config["metric"],
        random_state=random_seed,
        transform_seed=random_seed,
        low_memory=True,
        n_jobs=1,
    )

    embedding = reducer.fit_transform(sample)

    joblib.dump(reducer, MODELS / "umap_standard.joblib")

    output = metadata.iloc[sample_indices].copy()
    output["UMAP1"] = embedding[:, 0]
    output["UMAP2"] = embedding[:, 1]

    output.to_parquet(
        DATA_PROCESSED / "umap_embedding_sample.parquet",
        index=False,
    )

    plt.figure(figsize=(8, 7))
    plt.hexbin(
        embedding[:, 0],
        embedding[:, 1],
        gridsize=70,
        cmap="viridis",
        mincnt=1,
        bins="log",
    )
    plt.colorbar(label="log-scaled point count")
    plt.xlabel("UMAP1")
    plt.ylabel("UMAP2")
    plt.title("UMAP embedding—density view without class labels")
    save_figure("phase1_umap_density.png")

    return output


def plot_hopkins(hopkins_results: pd.DataFrame) -> None:
    plt.figure(figsize=(8, 5))

    sns.boxplot(
        data=hopkins_results,
        x="scaler",
        y="hopkins",
    )
    sns.stripplot(
        data=hopkins_results,
        x="scaler",
        y="hopkins",
        color="black",
        alpha=0.7,
    )

    plt.axhline(
        0.50,
        color="gray",
        linestyle="--",
        label="Random-like reference: H ≈ 0.5",
    )
    plt.axhline(
        0.70,
        color="red",
        linestyle=":",
        label="Strong tendency guideline: H ≈ 0.7",
    )

    plt.ylim(0, 1.02)
    plt.title("Repeated Hopkins clustering-tendency assessment")
    plt.legend(loc="lower right")

    save_figure("phase1_hopkins_results.png")


def plot_vat(vat_matrix: np.ndarray) -> None:
    upper_limit = float(
        np.quantile(vat_matrix, 0.98)
    )

    plt.figure(figsize=(8, 7))
    sns.heatmap(
        vat_matrix,
        cmap="gray_r",
        vmin=0,
        vmax=upper_limit,
        xticklabels=False,
        yticklabels=False,
        cbar_kws={"label": "Reordered Euclidean dissimilarity"},
    )
    plt.title("VAT-style reordered dissimilarity matrix")
    plt.xlabel("Reordered observations")
    plt.ylabel("Reordered observations")

    save_figure("phase1_vat_heatmap.png")


# ---------------------------------------------------------------------
# Reports and manifests
# ---------------------------------------------------------------------

def create_scaler_summary(
    features: pd.DataFrame,
    scaler_outputs: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    rows = []

    for scaler_name, result in scaler_outputs.items():
        transformed = result["train"]

        for index, feature_name in enumerate(features.columns):
            values = transformed[:, index]

            rows.append(
                {
                    "scaler": scaler_name,
                    "feature": feature_name,
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "median": float(np.median(values)),
                    "q25": float(np.quantile(values, 0.25)),
                    "q75": float(np.quantile(values, 0.75)),
                    "minimum": float(np.min(values)),
                    "maximum": float(np.max(values)),
                }
            )

    return pd.DataFrame(rows)


def create_environment_manifest() -> dict[str, Any]:
    packages = {}

    for package_name in [
        "numpy",
        "pandas",
        "scipy",
        "sklearn",
        "matplotlib",
        "seaborn",
        "umap",
        "joblib",
    ]:
        try:
            module = __import__(package_name)
            packages[package_name] = getattr(
                module,
                "__version__",
                "unknown",
            )
        except Exception:
            packages[package_name] = "not-installed"

    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "packages": packages,
    }


def write_phase1_summary(
    raw_df: pd.DataFrame,
    clean_df: pd.DataFrame,
    features: pd.DataFrame,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    pca: PCA,
    hopkins_results: pd.DataFrame,
    raw_hash: str,
) -> None:
    grouped_hopkins = (
        hopkins_results.groupby("scaler")["hopkins"]
        .agg(["mean", "std", "min", "max"])
        .round(6)
    )

    summary = f"""# Phase 1 Execution Summary

## Dataset

- Raw rows: {len(raw_df):,}
- Clean rows: {len(clean_df):,}
- Engineered features: {features.shape[1]}
- Training records: {len(train_indices):,}
- Held-out temporal records: {len(test_indices):,}
- Raw SHA-256: `{raw_hash}`

## Label-isolation policy

The `Class` field was not included in feature engineering, scaling, PCA,
UMAP, Hopkins, VAT, or distance diagnostics. It is retained only for
external evaluation in Phase 2.

## Engineered representation

The clustering matrix contains:

- PCA-anonymised transaction variables `V1` through `V28`
- `log(1 + Amount)`
- cyclic time-of-day features `Time_sin` and `Time_cos`

The raw `Amount` field was not added alongside `Amount_log1p` because
doing so would duplicate the contribution of transaction amount.

## Scaling

Both `StandardScaler` and `RobustScaler` were fitted only on the first
temporal training partition. The same fitted parameters were used to
transform the held-out portion. The primary Phase 1 representation is
StandardScaler so that Hopkins is reported on a standardised feature
matrix; RobustScaler is retained for Phase 2 sensitivity analysis.

## PCA

Selected principal components: {pca.n_components_}

Explained variance captured by selected PCA:
{float(np.sum(pca.explained_variance_ratio_)):.6f}

## Hopkins repeated results

```text
{grouped_hopkins.to_string()}
