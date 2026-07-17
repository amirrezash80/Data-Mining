import argparse
import hashlib
import json
import logging
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pandera.pandas as pa
import yaml
from pandera import Check, Column, DataFrameSchema
from scipy.stats import ks_2samp
from sklearn.metrics import pairwise_distances_argmin_min


ROOT = Path(__file__).resolve().parents[1]
PARAMS_PATH = ROOT / "params_deployment.yaml"


def load_params() -> dict[str, Any]:
    if not PARAMS_PATH.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {PARAMS_PATH}"
        )

    with PARAMS_PATH.open("r", encoding="utf-8") as file:
        params = yaml.safe_load(file)

    if not isinstance(params, dict):
        raise ValueError("Invalid deployment configuration")

    return params


def resolve_path(value: str) -> Path:
    return ROOT / value


def ensure_directories(params: dict[str, Any]) -> None:
    directories = [
        resolve_path(params["models"]["registry_directory"]),
        resolve_path(params["reports"]["output_directory"]),
        resolve_path(params["reports"]["final_report"]).parent,
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


def setup_logging(params: dict[str, Any]) -> None:
    report_directory = resolve_path(
        params["reports"]["output_directory"]
    )

    report_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(
                report_directory / "deployment.log",
                mode="w",
                encoding="utf-8",
            ),
        ],
        force=True,
    )


def save_json(data: Any, path: Path) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open("w", encoding="utf-8") as file:
        json.dump(
            data,
            file,
            ensure_ascii=False,
            indent=2,
            default=str,
        )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        while True:
            block = file.read(1024 * 1024)

            if not block:
                break

            digest.update(block)

    return digest.hexdigest()


def create_raw_schema(
    require_class: bool,
) -> DataFrameSchema:
    columns: dict[str, Column] = {
        "Time": Column(
            float,
            checks=Check.ge(0),
            coerce=True,
        ),
        "Amount": Column(
            float,
            checks=Check.ge(0),
            coerce=True,
        ),
    }

    for index in range(1, 29):
        columns[f"V{index}"] = Column(
            float,
            nullable=False,
            coerce=True,
        )

    columns["Class"] = Column(
        int,
        checks=Check.isin([0, 1]),
        nullable=False,
        coerce=True,
        required=require_class,
    )

    return DataFrameSchema(
        columns,
        strict=False,
        coerce=True,
    )


def validate_raw_dataframe(
    frame: pd.DataFrame,
    require_class: bool,
) -> pd.DataFrame:
    schema = create_raw_schema(
        require_class=require_class
    )

    validated = schema.validate(
        frame,
        lazy=True,
    )

    required_columns = [
        "Time",
        *[
            f"V{index}"
            for index in range(1, 29)
        ],
        "Amount",
    ]

    if require_class:
        required_columns.append("Class")

    if validated[
        required_columns
    ].isna().any().any():
        raise ValueError(
            "Validated data contains missing values"
        )

    return validated


def validate_phase1_outputs(
    params: dict[str, Any],
) -> dict[str, Any]:
    arrays_path = resolve_path(
        params["data"]["phase1_arrays"]
    )

    cleaned_path = resolve_path(
        params["data"]["cleaned_data"]
    )

    feature_path = resolve_path(
        params["data"]["unscaled_features"]
    )

    metadata_path = resolve_path(
        params["data"]["metadata"]
    )

    required_paths = [
        arrays_path,
        cleaned_path,
        feature_path,
        metadata_path,
    ]

    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(
                f"Required artifact not found: {path}"
            )

    cleaned = pd.read_parquet(cleaned_path)
    validated_cleaned = validate_raw_dataframe(
        cleaned,
        require_class=True,
    )

    features = pd.read_parquet(feature_path)
    metadata = pd.read_parquet(metadata_path)

    if len(features) != len(metadata):
        raise ValueError(
            "Feature and metadata row counts differ"
        )

    if len(features) != len(validated_cleaned):
        raise ValueError(
            "Cleaned and feature row counts differ"
        )

    if features.isna().any().any():
        raise ValueError(
            "Feature table contains missing values"
        )

    non_numeric = [
        column
        for column in features.columns
        if not pd.api.types.is_numeric_dtype(
            features[column]
        )
    ]

    if non_numeric:
        raise TypeError(
            f"Non-numeric feature columns: {non_numeric}"
        )

    with np.load(
        arrays_path,
        allow_pickle=False,
    ) as arrays:
        required_arrays = {
            "X_standard",
            "X_robust",
            "X_pca",
            "y",
            "row_id",
            "train_indices",
            "test_indices",
            "feature_names",
        }

        missing_arrays = (
            required_arrays
            - set(arrays.files)
        )

        if missing_arrays:
            raise KeyError(
                f"Missing arrays: {sorted(missing_arrays)}"
            )

        array_rows = len(arrays["y"])

        if array_rows != len(features):
            raise ValueError(
                "Array and feature row counts differ"
            )

        result = {
            "status": "valid",
            "row_count": int(array_rows),
            "unscaled_feature_count": int(
                features.shape[1]
            ),
            "standard_feature_count": int(
                arrays["X_standard"].shape[1]
            ),
            "pca_feature_count": int(
                arrays["X_pca"].shape[1]
            ),
            "train_count": int(
                len(arrays["train_indices"])
            ),
            "test_count": int(
                len(arrays["test_indices"])
            ),
        }

    return result


def load_phase1_arrays(
    params: dict[str, Any],
) -> dict[str, np.ndarray]:
    path = resolve_path(
        params["data"]["phase1_arrays"]
    )

    with np.load(
        path,
        allow_pickle=False,
    ) as arrays:
        return {
            name: arrays[name]
            for name in arrays.files
        }


def build_features(
    raw_frame: pd.DataFrame,
    feature_names: list[str],
) -> pd.DataFrame:
    validated = validate_raw_dataframe(
        raw_frame,
        require_class=False,
    )

    result = pd.DataFrame(
        index=validated.index
    )

    seconds_per_day = 86400.0

    phase = (
        2.0
        * np.pi
        * (
            validated["Time"].astype(float)
            % seconds_per_day
        )
        / seconds_per_day
    )

    for feature_name in feature_names:
        if feature_name.startswith("V"):
            if feature_name not in validated.columns:
                raise KeyError(
                    f"Missing feature: {feature_name}"
                )

            result[feature_name] = validated[
                feature_name
            ].astype(float)

        elif feature_name == "Amount_log1p":
            result[feature_name] = np.log1p(
                validated["Amount"].astype(float)
            )

        elif feature_name == "Time_sin":
            result[feature_name] = np.sin(phase)

        elif feature_name == "Time_cos":
            result[feature_name] = np.cos(phase)

        elif feature_name == "Elapsed_day":
            result[feature_name] = (
                validated["Time"].astype(float)
                / seconds_per_day
            )

        elif feature_name in validated.columns:
            result[feature_name] = validated[
                feature_name
            ].astype(float)

        else:
            raise KeyError(
                f"Unsupported engineered feature: {feature_name}"
            )

    if list(result.columns) != feature_names:
        raise ValueError(
            "Engineered feature order differs from training"
        )

    if result.isna().any().any():
        raise ValueError(
            "Engineered features contain missing values"
        )

    return result


def load_consensus_assignments(
    params: dict[str, Any],
) -> pd.DataFrame:
    path = resolve_path(
        params["data"]["consensus_assignments"]
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Consensus assignments not found: {path}"
        )

    assignments = pd.read_parquet(path)

    required_columns = {
        "array_index",
        "Consensus_cluster",
    }

    missing = (
        required_columns
        - set(assignments.columns)
    )

    if missing:
        raise KeyError(
            f"Missing assignment columns: {sorted(missing)}"
        )

    assignments["array_index"] = assignments[
        "array_index"
    ].astype(np.int64)

    assignments["Consensus_cluster"] = assignments[
        "Consensus_cluster"
    ].astype(np.int32)

    if assignments["array_index"].duplicated().any():
        raise ValueError(
            "Duplicate array indices in consensus assignments"
        )

    return assignments


def fit_assignment_registry(
    params: dict[str, Any],
) -> dict[str, Any]:
    arrays = load_phase1_arrays(params)
    assignments = load_consensus_assignments(
        params
    )

    indices = assignments[
        "array_index"
    ].to_numpy(dtype=np.int64)

    labels = assignments[
        "Consensus_cluster"
    ].to_numpy(dtype=np.int32)

    if np.any(indices < 0):
        raise IndexError(
            "Negative assignment indices found"
        )

    if np.any(indices >= len(arrays["X_pca"])):
        raise IndexError(
            "Assignment index exceeds PCA array"
        )

    matrix = arrays["X_pca"][
        indices
    ].astype(np.float64)

    cluster_ids = np.unique(labels)

    centroids = np.vstack(
        [
            matrix[
                labels == cluster_id
            ].mean(axis=0)
            for cluster_id in cluster_ids
        ]
    )

    cluster_sizes = np.asarray(
        [
            np.sum(labels == cluster_id)
            for cluster_id in cluster_ids
        ],
        dtype=np.int64,
    )

    registry_directory = resolve_path(
        params["models"]["registry_directory"]
    )

    registry_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    registry_path = (
        registry_directory
        / "consensus_assignment_registry.npz"
    )

    np.savez_compressed(
        registry_path,
        cluster_ids=cluster_ids,
        centroids=centroids.astype(np.float32),
        cluster_sizes=cluster_sizes,
        feature_names=arrays[
            "feature_names"
        ].astype(str),
    )

    scaler_source = resolve_path(
        params["models"]["scaler"]
    )

    reducer_source = resolve_path(
        params["models"]["reducer"]
    )

    if not scaler_source.exists():
        raise FileNotFoundError(
            f"Scaler not found: {scaler_source}"
        )

    if not reducer_source.exists():
        raise FileNotFoundError(
            f"Reducer not found: {reducer_source}"
        )

    scaler_target = (
        registry_directory
        / "standard_scaler.joblib"
    )

    reducer_target = (
        registry_directory
        / "pca_standard.joblib"
    )

    shutil.copy2(
        scaler_source,
        scaler_target,
    )

    shutil.copy2(
        reducer_source,
        reducer_target,
    )

    source_paths = {
        "phase1_arrays": resolve_path(
            params["data"]["phase1_arrays"]
        ),
        "assignments": resolve_path(
            params["data"]["consensus_assignments"]
        ),
        "scaler": scaler_source,
        "reducer": reducer_source,
        "registry": registry_path,
    }

    artifact_hashes = {
        name: sha256_file(path)
        for name, path in source_paths.items()
    }

    metric_files = [
        resolve_path(
            params["reports"]["phase2_comparison"]
        ),
        resolve_path(
            params["reports"]["phase3_comparison"]
        ),
    ]

    metrics = {}

    for metric_path in metric_files:
        if metric_path.exists():
            metrics[
                str(metric_path.relative_to(ROOT))
            ] = pd.read_csv(
                metric_path
            ).to_dict(
                orient="records"
            )

    metadata = {
        "model_type": (
            "Nearest-centroid assignment proxy "
            "for Phase 3 consensus clusters"
        ),
        "fit_date_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "cluster_ids": cluster_ids.tolist(),
        "cluster_sizes": cluster_sizes.tolist(),
        "pca_dimensions": int(
            centroids.shape[1]
        ),
        "training_sample_size": int(
            len(matrix)
        ),
        "distance_metric": "euclidean",
        "class_used_for_assignment": False,
        "artifact_hashes": artifact_hashes,
        "metric_scoreboard": metrics,
    }

    metadata_path = (
        registry_directory
        / "registry_metadata.json"
    )

    save_json(
        metadata,
        metadata_path,
    )

    logging.info(
        "Assignment registry saved: %s",
        registry_path,
    )

    return metadata


def load_registry(
    params: dict[str, Any],
) -> dict[str, Any]:
    registry_directory = resolve_path(
        params["models"]["registry_directory"]
    )

    registry_path = (
        registry_directory
        / "consensus_assignment_registry.npz"
    )

    if not registry_path.exists():
        raise FileNotFoundError(
            "Assignment registry has not been fitted"
        )

    with np.load(
        registry_path,
        allow_pickle=False,
    ) as registry:
        result = {
            name: registry[name]
            for name in registry.files
        }

    result["scaler"] = joblib.load(
        registry_directory
        / "standard_scaler.joblib"
    )

    result["reducer"] = joblib.load(
        registry_directory
        / "pca_standard.joblib"
    )

    return result


def assign_dataframe(
    raw_frame: pd.DataFrame,
    params: dict[str, Any] = None,
) -> pd.DataFrame:
    if params is None:
        params = load_params()

    registry = load_registry(params)

    feature_names = (
        registry["feature_names"]
        .astype(str)
        .tolist()
    )

    features = build_features(
        raw_frame,
        feature_names,
    )

    scaled = registry["scaler"].transform(
        features.to_numpy(dtype=np.float64)
    )

    reduced = registry["reducer"].transform(
        scaled
    )

    centroid_positions, nearest_distances = (
        pairwise_distances_argmin_min(
            reduced,
            registry["centroids"],
            metric="euclidean",
        )
    )

    cluster_ids = registry["cluster_ids"]

    assigned_clusters = cluster_ids[
        centroid_positions
    ]

    result = raw_frame.reset_index(
        drop=True
    ).copy()

    result["assigned_cluster"] = (
        assigned_clusters.astype(np.int32)
    )

    result["nearest_centroid_distance"] = (
        nearest_distances.astype(float)
    )

    all_distances = np.linalg.norm(
        reduced[:, None, :]
        - registry["centroids"][None, :, :],
        axis=2,
    )

    for position, cluster_id in enumerate(
        cluster_ids
    ):
        result[
            f"distance_to_cluster_{int(cluster_id)}"
        ] = all_distances[:, position]

    return result


def assign_csv(
    input_path: Path,
    output_path: Path,
    params: dict[str, Any],
) -> None:
    if not input_path.exists():
        raise FileNotFoundError(
            f"Input CSV not found: {input_path}"
        )

    frame = pd.read_csv(input_path)

    assigned = assign_dataframe(
        frame,
        params,
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    assigned.to_csv(
        output_path,
        index=False,
    )

    logging.info(
        "Assignments saved: %s",
        output_path,
    )


def calculate_psi(
    reference: np.ndarray,
    current: np.ndarray,
    bins: int,
) -> float:
    reference = np.asarray(
        reference,
        dtype=float,
    )

    current = np.asarray(
        current,
        dtype=float,
    )

    quantiles = np.linspace(
        0.0,
        1.0,
        bins + 1,
    )

    edges = np.unique(
        np.quantile(
            reference,
            quantiles,
        )
    )

    if len(edges) < 3:
        minimum = min(
            reference.min(),
            current.min(),
        )

        maximum = max(
            reference.max(),
            current.max(),
        )

        if minimum == maximum:
            return 0.0

        edges = np.linspace(
            minimum,
            maximum,
            bins + 1,
        )

    edges[0] = -np.inf
    edges[-1] = np.inf

    reference_counts = np.histogram(
        reference,
        bins=edges,
    )[0].astype(float)

    current_counts = np.histogram(
        current,
        bins=edges,
    )[0].astype(float)

    epsilon = 1e-8

    reference_distribution = (
        reference_counts
        / max(reference_counts.sum(), 1.0)
    )

    current_distribution = (
        current_counts
        / max(current_counts.sum(), 1.0)
    )

    reference_distribution = np.clip(
        reference_distribution,
        epsilon,
        None,
    )

    current_distribution = np.clip(
        current_distribution,
        epsilon,
        None,
    )

    psi = np.sum(
        (
            current_distribution
            - reference_distribution
        )
        * np.log(
            current_distribution
            / reference_distribution
        )
    )

    return float(psi)


def calculate_drift_report(
    params: dict[str, Any],
) -> pd.DataFrame:
    arrays = load_phase1_arrays(params)

    matrix = arrays[
        "X_standard"
    ].astype(np.float64)

    train_indices = arrays[
        "train_indices"
    ].astype(np.int64)

    test_indices = arrays[
        "test_indices"
    ].astype(np.int64)

    feature_names = arrays[
        "feature_names"
    ].astype(str)

    bins = int(
        params["drift"]["bins"]
    )

    psi_warning = float(
        params["drift"][
            "psi_warning_threshold"
        ]
    )

    psi_refit = float(
        params["drift"][
            "psi_refit_threshold"
        ]
    )

    ks_warning = float(
        params["drift"][
            "ks_warning_threshold"
        ]
    )

    pvalue_threshold = float(
        params["drift"][
            "ks_pvalue_threshold"
        ]
    )

    rows = []

    for position, feature_name in enumerate(
        feature_names
    ):
        reference = matrix[
            train_indices,
            position,
        ]

        current = matrix[
            test_indices,
            position,
        ]

        psi = calculate_psi(
            reference,
            current,
            bins,
        )

        ks_result = ks_2samp(
            reference,
            current,
            alternative="two-sided",
            method="auto",
        )

        if psi >= psi_refit:
            status = "refit"
        elif (
            psi >= psi_warning
            or (
                ks_result.statistic
                >= ks_warning
                and ks_result.pvalue
                <= pvalue_threshold
            )
        ):
            status = "warning"
        else:
            status = "stable"

        rows.append(
            {
                "feature": feature_name,
                "psi": float(psi),
                "ks_statistic": float(
                    ks_result.statistic
                ),
                "ks_pvalue": float(
                    ks_result.pvalue
                ),
                "status": status,
                "reference_mean": float(
                    np.mean(reference)
                ),
                "current_mean": float(
                    np.mean(current)
                ),
                "reference_std": float(
                    np.std(reference)
                ),
                "current_std": float(
                    np.std(current)
                ),
            }
        )

    report = pd.DataFrame(rows)

    report_directory = resolve_path(
        params["reports"]["output_directory"]
    )

    report.to_csv(
        report_directory
        / "feature_drift_report.csv",
        index=False,
    )

    summary = {
        "created_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "reference_records": int(
            len(train_indices)
        ),
        "current_records": int(
            len(test_indices)
        ),
        "stable_features": int(
            np.sum(report["status"] == "stable")
        ),
        "warning_features": int(
            np.sum(report["status"] == "warning")
        ),
        "refit_features": int(
            np.sum(report["status"] == "refit")
        ),
        "maximum_psi": float(
            report["psi"].max()
        ),
        "maximum_ks": float(
            report["ks_statistic"].max()
        ),
        "refit_recommended": bool(
            np.any(report["status"] == "refit")
        ),
    }

    save_json(
        summary,
        report_directory
        / "drift_summary.json",
    )

    return report


def dataframe_to_text(
    frame: pd.DataFrame,
    maximum_rows: int = 30,
) -> str:
    if frame.empty:
        return "No data available."

    return frame.head(
        maximum_rows
    ).to_string(
        index=False
    )


def read_text_if_exists(
    path: Path,
) -> str:
    if not path.exists():
        return "Not generated."

    return path.read_text(
        encoding="utf-8"
    )


def read_csv_if_exists(
    path: Path,
) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()

    return pd.read_csv(path)


def read_json_if_exists(
    path: Path,
) -> Any:
    if not path.exists():
        return {
            "status": "not generated"
        }

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def build_final_report(
    params: dict[str, Any],
) -> Path:
    phase1_summary = read_text_if_exists(
        resolve_path(
            params["reports"]["phase1_summary"]
        )
    )

    phase2_summary = read_text_if_exists(
        resolve_path(
            params["reports"]["phase2_summary"]
        )
    )

    phase3_summary = read_text_if_exists(
        resolve_path(
            params["reports"]["phase3_summary"]
        )
    )

    phase2_comparison = read_csv_if_exists(
        resolve_path(
            params["reports"][
                "phase2_comparison"
            ]
        )
    )

    phase3_comparison = read_csv_if_exists(
        resolve_path(
            params["reports"][
                "phase3_comparison"
            ]
        )
    )

    cluster_profiles = read_csv_if_exists(
        resolve_path(
            params["reports"][
                "cluster_profiles"
            ]
        )
    )

    cluster_labels = read_csv_if_exists(
        resolve_path(
            params["reports"][
                "cluster_labels"
            ]
        )
    )

    fraud_composition = read_csv_if_exists(
        resolve_path(
            params["reports"][
                "fraud_composition"
            ]
        )
    )

    sensitivity = read_csv_if_exists(
        resolve_path(
            params["reports"]["sensitivity"]
        )
    )

    anomaly_summary = read_json_if_exists(
        resolve_path(
            params["reports"][
                "anomaly_summary"
            ]
        )
    )

    deployment_directory = resolve_path(
        params["reports"]["output_directory"]
    )

    drift_summary = read_json_if_exists(
        deployment_directory
        / "drift_summary.json"
    )

    registry_metadata = read_json_if_exists(
        resolve_path(
            params["models"][
                "registry_directory"
            ]
        )
        / "registry_metadata.json"
    )

    report = f"""# Clustering Analysis on Credit Card Transactions

## Abstract

This project investigates whether credit-card transactions form natural and density-distinct subpopulations and whether small clusters or anomalous observations are enriched for fraudulent transactions. The analysis uses the anonymised European credit-card transaction dataset containing PCA-transformed transaction variables, transaction time, amount, and an external fraud label. The fraud label is excluded from preprocessing, dimensionality reduction, clustering, hyperparameter selection, consensus construction, and anomaly scoring. It is used only for post-hoc external evaluation.

The project proceeds through reproducible preprocessing, clustering-tendency assessment, comparison of partitioning, hierarchical, density-based, and model-based algorithms, and an advanced ensemble-consensus analysis. The advanced method combines multiple clusterings generated under different algorithms, values of k, random seeds, PCA dimensions, and covariance structures. Cluster interpretation is performed through statistical profiles, exemplars, boundary cases, shallow decision-tree rules, and optional SHAP explanations. A downstream anomaly score ranks transactions by robust within-cluster distance. Sensitivity to preprocessing is assessed by comparing partitions under standard scaling, robust scaling, and PCA. A production layer adds schema validation, artifact hashing, model registration, temporal drift monitoring, and live nearest-centroid assignment.

All numerical claims in this report are produced by the committed scripts and persisted artifacts.

## 1. Introduction

The central research question is:

Do credit-card transactions form natural density-distinct subpopulations, and do small clusters or low-density observations meaningfully overlap fraudulent transactions?

The project does not assume in advance that fraud forms one isolated cluster. Fraud may instead occur across several transaction profiles or appear as rare observations within otherwise legitimate clusters.

## 2. Data and Preprocessing

{phase1_summary}

## 3. Algorithm Portfolio

{phase2_summary}

## 4. Phase 2 Final Comparison

{dataframe_to_text(phase2_comparison)}

## 5. Advanced Consensus Track

{phase3_summary}

## 6. Consensus versus Best Base Model

{dataframe_to_text(phase3_comparison)}

## 7. Cluster Interpretation

### 7.1 Proposed cluster labels

{dataframe_to_text(cluster_labels)}

### 7.2 Cluster profiles

{dataframe_to_text(cluster_profiles)}

### 7.3 Interpretation limitation

Variables V1 through V28 are anonymised PCA features. Consequently, cluster labels describe statistical deviations rather than direct business concepts. The amount and temporal features permit limited operational interpretation, but the hidden semantics of the PCA variables prevent strong causal or behavioural naming.

## 8. Downstream Anomaly Analysis

{json.dumps(anomaly_summary, ensure_ascii=False, indent=2)}

The anomaly score represents unusualness relative to the assigned cluster. It is not a calibrated probability of fraud.

## 9. Post-hoc Fraud Composition

{dataframe_to_text(fraud_composition)}

Purity and raw fraud rates must be interpreted cautiously because the external fraud class is severely imbalanced.

## 10. Fairness and Sensitivity

The dataset does not contain explicit demographic attributes such as gender, race, age, or socio-economic category. A demographic fairness audit is therefore not possible. Fraud composition by cluster is an outcome audit and is not equivalent to a fairness audit.

### 10.1 Preprocessing sensitivity

{dataframe_to_text(sensitivity)}

High disagreement between preprocessing alternatives weakens the claim that the discovered partition is unique or intrinsic.

## 11. Production Pipeline

The production pipeline validates input schema, checks row consistency, versions artifacts with SHA-256 hashes, serialises the scaler, PCA reducer, consensus centroids, cluster identifiers, data version, fit date, and metric scoreboard.

Consensus hierarchical clustering does not provide a native prediction method for unseen observations. Live assignment is therefore implemented as a documented nearest-centroid proxy in the fitted PCA space. This operational approximation must not be confused with reconstructing the original consensus procedure for each new record.

### 11.1 Registry metadata

{json.dumps(registry_metadata, ensure_ascii=False, indent=2)}

### 11.2 Temporal drift monitoring

{json.dumps(drift_summary, ensure_ascii=False, indent=2)}

Feature drift is measured using Population Stability Index and the two-sample Kolmogorov-Smirnov statistic between the training period and the held-out later period.

## 12. Discussion and Limitations

1. The consensus matrix is constructed on a reproducible subset because its memory complexity is quadratic.
2. Consensus clustering has no native out-of-sample prediction rule; live assignment uses a centroid proxy.
3. Fraud is extremely rare, so purity can be high for uninformative partitions.
4. External fraud labels are not a complete definition of transaction-profile quality.
5. PCA anonymisation restricts domain interpretation.
6. Clustering and anomaly detection do not establish causality.
7. Sensitivity to scaling or dimensionality reduction weakens claims of intrinsic structure.
8. Drift thresholds are operational rules and should be calibrated with future data.

## 13. Conclusion

The project determines whether transaction-profile structure exists, compares classical clustering families, evaluates stability and external agreement, and constructs an advanced consensus partition. The final practical value lies in interpretable transaction segments, cluster-specific fraud enrichment, and anomaly prioritisation rather than assuming that all fraud constitutes one homogeneous cluster.

## Appendix A. Reproducibility

The project is reproduced by running:

1. python src/phase1.py
2. python src/phase2.py
3. python src/phase3.py
4. python src/deployment.py all
5. streamlit run dashboard/app.py

All random seeds, model artifacts, hashes, metrics, assignments, and drift outputs are persisted under data, models, and reports.
"""

    output_path = resolve_path(
        params["reports"]["final_report"]
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        report,
        encoding="utf-8",
    )

    logging.info(
        "Final report saved: %s",
        output_path,
    )

    return output_path


def create_artifact_manifest(
    params: dict[str, Any],
) -> dict[str, Any]:
    candidate_paths = [
        resolve_path(
            params["data"]["phase1_arrays"]
        ),
        resolve_path(
            params["data"]["cleaned_data"]
        ),
        resolve_path(
            params["data"]["unscaled_features"]
        ),
        resolve_path(
            params["data"]["metadata"]
        ),
        resolve_path(
            params["data"][
                "consensus_assignments"
            ]
        ),
        resolve_path(
            params["models"]["scaler"]
        ),
        resolve_path(
            params["models"]["reducer"]
        ),
        resolve_path(
            params["models"][
                "registry_directory"
            ]
        )
        / "consensus_assignment_registry.npz",
    ]

    artifacts = {}

    for path in candidate_paths:
        if path.exists() and path.is_file():
            artifacts[
                str(path.relative_to(ROOT))
            ] = {
                "sha256": sha256_file(path),
                "size_bytes": int(
                    path.stat().st_size
                ),
            }

    manifest = {
        "created_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "artifacts": artifacts,
    }

    save_json(
        manifest,
        resolve_path(
            params["reports"][
                "output_directory"
            ]
        )
        / "artifact_manifest.json",
    )

    return manifest


def run_validation(
    params: dict[str, Any],
) -> None:
    result = validate_phase1_outputs(
        params
    )

    save_json(
        result,
        resolve_path(
            params["reports"][
                "output_directory"
            ]
        )
        / "schema_validation.json",
    )

    logging.info(
        "Schema validation completed"
    )


def run_all(
    params: dict[str, Any],
) -> None:
    started = datetime.now(
        timezone.utc
    )

    run_validation(params)
    fit_assignment_registry(params)
    calculate_drift_report(params)
    create_artifact_manifest(params)
    build_final_report(params)

    completed = datetime.now(
        timezone.utc
    )

    save_json(
        {
            "status": "completed",
            "started_at_utc": (
                started.isoformat()
            ),
            "completed_at_utc": (
                completed.isoformat()
            ),
        },
        resolve_path(
            params["reports"][
                "output_directory"
            ]
        )
        / "execution_record.json",
    )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    subparsers.add_parser("validate")
    subparsers.add_parser("registry")
    subparsers.add_parser("monitor")
    subparsers.add_parser("manifest")
    subparsers.add_parser("report")
    subparsers.add_parser("all")

    assign_parser = subparsers.add_parser(
        "assign"
    )

    assign_parser.add_argument(
        "--input",
        required=True,
    )

    assign_parser.add_argument(
        "--output",
        required=True,
    )

    return parser.parse_args()


def main() -> None:
    params = load_params()
    ensure_directories(params)
    setup_logging(params)

    arguments = parse_arguments()

    if arguments.command == "validate":
        run_validation(params)

    elif arguments.command == "registry":
        fit_assignment_registry(params)

    elif arguments.command == "monitor":
        calculate_drift_report(params)

    elif arguments.command == "manifest":
        create_artifact_manifest(params)

    elif arguments.command == "report":
        build_final_report(params)

    elif arguments.command == "assign":
        assign_csv(
            Path(arguments.input),
            Path(arguments.output),
            params,
        )

    elif arguments.command == "all":
        run_all(params)


if __name__ == "__main__":
    main()