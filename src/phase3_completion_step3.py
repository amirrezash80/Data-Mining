import json
import logging
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from sklearn.tree import DecisionTreeClassifier


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
        directory.mkdir(parents=True, exist_ok=True)


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(
                REPORT_DIR / "phase3_completion_step3.log",
                mode="w",
                encoding="utf-8",
            ),
        ],
        force=True,
    )


def setup_plots() -> None:
    sns.set_theme(style="whitegrid", context="notebook", palette="colorblind")
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 200,
            "figure.autolayout": True,
        }
    )


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with path.open("r", encoding="utf-8") as file:
        params = yaml.safe_load(file)
    if not isinstance(params, dict):
        raise ValueError(f"Invalid configuration file: {path}")
    return params


def save_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2, default=str)


def save_figure(filename: str) -> Path:
    path = FIGURE_DIR / filename
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    logging.info("Saved figure: %s", path)
    return path


def load_inputs() -> tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame]:
    """بارگذاری داده‌های اصلی، برچسب‌های خوشه، و ویژگی‌ها."""
    for path in [ARRAY_PATH, ASSIGNMENT_PATH, FEATURE_PATH]:
        if not path.exists():
            raise FileNotFoundError(f"Required input was not found: {path}")

    with np.load(ARRAY_PATH, allow_pickle=False) as arrays:
        if "X_pca" not in arrays:
            raise KeyError("Missing X_pca in Phase 1 arrays")
        x_pca_all = arrays["X_pca"].astype(np.float64)

    assignments = pd.read_parquet(ASSIGNMENT_PATH)
    if "array_index" not in assignments.columns or "Consensus_cluster" not in assignments.columns:
        raise KeyError("Assignment file missing required columns")

    if assignments["array_index"].duplicated().any():
        raise ValueError("Duplicate array_index in assignments")

    array_indices = assignments["array_index"].to_numpy(dtype=np.int64)
    if np.any(array_indices >= len(x_pca_all)):
        raise IndexError("Assignment index exceeds Phase 1 array length")

    x = x_pca_all[array_indices]
    labels = assignments["Consensus_cluster"].to_numpy(dtype=np.int32)

    features = pd.read_parquet(FEATURE_PATH)
    if len(features) != len(x_pca_all):
        raise ValueError("Feature table and Phase 1 arrays row count mismatch")

    features_subset = features.iloc[array_indices].reset_index(drop=True).copy()

    return x, labels, features_subset, assignments


def train_tree_classifier(
    x: np.ndarray,
    labels: np.ndarray,
    params: dict[str, Any],
    random_seed: int,
) -> DecisionTreeClassifier:
    """آموزش یک درخت تصمیم کم‌عمق برای پیش‌بینی برچسب خوشه."""
    tree_depth = int(params.get("interpretation", {}).get("tree_max_depth", 4))
    min_samples_leaf = int(params.get("interpretation", {}).get("tree_min_samples_leaf", 30))

    model = DecisionTreeClassifier(
        max_depth=tree_depth,
        min_samples_leaf=min_samples_leaf,
        class_weight="balanced",
        random_state=random_seed,
    )

    model.fit(x, labels)
    return model


def compute_shap_per_cluster(
    model: DecisionTreeClassifier,
    x: np.ndarray,
    labels: np.ndarray,
    feature_names: list[str],
    sample_size: int,
    random_seed: int,
) -> tuple[Optional[pd.DataFrame], dict[int, pd.DataFrame], dict[str, Any]]:
    """
    محاسبه SHAP برای هر خوشه به‌صورت جداگانه.

    Returns:
        shap_values_all: (n_samples, n_features) (اگر shap نصب باشد)
        per_cluster_shap: دیکشنری {cluster_id: DataFrame(ویژگی‌ها و اهمیت)}
        status: وضعیت اجرا
    """
    try:
        import shap
    except ImportError:
        logging.warning("SHAP is not installed. Skipping SHAP explanation.")
        return None, {}, {"status": "skipped", "reason": "shap not installed"}

    rng = np.random.default_rng(random_seed)
    n_samples = len(x)
    sample_size = min(sample_size, n_samples)
    sample_indices = rng.choice(n_samples, size=sample_size, replace=False)
    x_sample = x[sample_indices]
    labels_sample = labels[sample_indices]

    logging.info("Computing SHAP values on %d samples", sample_size)

    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(x_sample)
    except Exception as e:
        logging.exception("SHAP calculation failed")
        return None, {}, {"status": "failed", "error": str(e)}

    # shap_values can be a list (multi-class) or a single array
    if isinstance(shap_values, list):
        # For multi-class, we need to get the SHAP values for the predicted class?
        # The per-cluster explanation usually averages over all classes? Actually,
        # we want per-cluster importance: for each cluster, we take the SHAP values
        # of records assigned to that cluster and average the absolute SHAP.
        # Since shap_values is a list of arrays (one per class), for each record we take
        # the class it was assigned (true label) and use the corresponding SHAP values.
        # But shap.TreeExplainer returns SHAP values for each class. We need to align
        # with predicted class for each sample.
        # Simpler: use the first element if it's binary? No, let's handle properly:
        # For each sample, we want the SHAP values for the true label (cluster).
        # We'll create a 3D array: (n_samples, n_classes, n_features) and then select.
        if len(shap_values) == 2 and all(v.ndim == 2 for v in shap_values):
            # Binary classification: shap_values is a list of two arrays (class 0, class 1)
            # Use the SHAP values for the predicted class? Actually we want the class of the sample.
            # For each sample, we pick the SHAP values corresponding to its actual label.
            shap_3d = np.stack(shap_values, axis=1)  # (n_samples, n_classes, n_features)
            # Now for each sample, select the index corresponding to its label
            labels_int = labels_sample.astype(int)
            selected_shap = shap_3d[np.arange(len(shap_3d)), labels_int]  # (n_samples, n_features)
            shap_values_used = selected_shap
        else:
            # For multi-class, we can do the same
            if isinstance(shap_values, list) and all(v.ndim == 2 for v in shap_values):
                shap_3d = np.stack(shap_values, axis=1)
                labels_int = labels_sample.astype(int)
                # Ensure labels are within range
                if np.max(labels_int) >= shap_3d.shape[1]:
                    # fallback: use first class
                    logging.warning("Labels exceed number of classes. Using first class for SHAP.")
                    selected_shap = shap_3d[:, 0, :]
                else:
                    selected_shap = shap_3d[np.arange(len(shap_3d)), labels_int]
                shap_values_used = selected_shap
            else:
                # Single output? (regression or binary with one output)
                shap_values_used = np.asarray(shap_values)
                if shap_values_used.ndim == 3:
                    # (n_samples, n_features, n_classes?) - unlikely
                    shap_values_used = shap_values_used.mean(axis=2)

    else:
        shap_values_used = np.asarray(shap_values)

    # Now shap_values_used is (sample_size, n_features)
    # Compute per-cluster mean absolute SHAP
    per_cluster_importance = {}
    for cluster_id in np.unique(labels_sample):
        mask = labels_sample == cluster_id
        if mask.sum() == 0:
            continue
        cluster_shap = shap_values_used[mask]  # (n_in_cluster, n_features)
        mean_abs_shap = np.abs(cluster_shap).mean(axis=0)  # (n_features,)
        per_cluster_importance[int(cluster_id)] = pd.DataFrame(
            {
                "feature": feature_names,
                "mean_abs_shap": mean_abs_shap,
            }
        ).sort_values("mean_abs_shap", ascending=False)

    # Also overall importance (across all samples)
    overall_mean_abs = np.abs(shap_values_used).mean(axis=0)
    overall_importance = pd.DataFrame(
        {
            "feature": feature_names,
            "mean_abs_shap": overall_mean_abs,
        }
    ).sort_values("mean_abs_shap", ascending=False)

    # Return shap values as DataFrame for potential inspection
    shap_df = pd.DataFrame(shap_values_used, columns=feature_names)
    shap_df["cluster"] = labels_sample

    return shap_df, per_cluster_importance, {"status": "completed", "sample_size": sample_size}


def plot_per_cluster_shap(
    per_cluster_importance: dict[int, pd.DataFrame],
    top_k: int = 15,
) -> Optional[Path]:
    """نمودار اهمیت ویژگی‌ها برای هر خوشه."""
    n_clusters = len(per_cluster_importance)
    if n_clusters == 0:
        logging.warning("No cluster SHAP data to plot")
        return None

    # Create a combined DataFrame for plotting
    rows = []
    for cluster_id, df in per_cluster_importance.items():
        top_features = df.head(top_k)
        for _, row in top_features.iterrows():
            rows.append(
                {
                    "cluster": f"Cluster {cluster_id}",
                    "feature": row["feature"],
                    "importance": row["mean_abs_shap"],
                }
            )
    combined = pd.DataFrame(rows)

    if combined.empty:
        return None

    # Plot using facet grid
    g = sns.catplot(
        data=combined,
        x="importance",
        y="feature",
        col="cluster",
        kind="bar",
        col_wrap=max(1, (n_clusters + 1) // 2),
        sharex=False,
        height=4,
        aspect=1.2,
    )
    g.set_axis_labels("Mean |SHAP|", "")
    g.set_titles(col_template="{col_name}")
    g.fig.suptitle("Top Features by Cluster (SHAP)", y=1.02)

    # Save
    path = save_figure("cluster_shap_importance.png")
    return path


def plot_overall_shap(overall_importance: pd.DataFrame, top_k: int = 20) -> Optional[Path]:
    """نمودار اهمیت کلی ویژگی‌ها."""
    if overall_importance is None or overall_importance.empty:
        return None

    top = overall_importance.head(top_k)
    plt.figure(figsize=(10, 7))
    sns.barplot(data=top, x="mean_abs_shap", y="feature")
    plt.xlabel("Mean |SHAP| (overall)")
    plt.title("Overall Feature Importance (SHAP)")
    return save_figure("overall_shap_importance.png")


def write_shap_report(
    per_cluster_importance: dict[int, pd.DataFrame],
    overall_importance: Optional[pd.DataFrame],
    model: DecisionTreeClassifier,
    x: np.ndarray,
    labels: np.ndarray,
    status: dict[str, Any],
) -> None:
    """نوشتن گزارش متنی SHAP."""
    # محاسبه دقت مدل
    train_accuracy = model.score(x, labels)

    lines = [
        "# SHAP-based Cluster Explanation",
        "",
        "A shallow decision tree was trained to predict cluster membership.",
        "SHAP values were then computed for a sampled subset of records.",
        "",
        "For each cluster, the mean absolute SHAP value of each feature is",
        "reported as the feature's influence on that cluster's assignment.",
        "",
        f"Model tree depth: {model.get_depth()}",
        f"Number of leaves: {model.get_n_leaves()}",
        f"Training accuracy: {train_accuracy:.4f}",
        "",
        f"SHAP computation status: {status.get('status', 'unknown')}",
    ]

    if status.get("status") == "completed" and per_cluster_importance:
        lines.append("")
        lines.append("## Per-Cluster Top Features")
        for cluster_id, df in sorted(per_cluster_importance.items()):
            lines.append(f"### Cluster {cluster_id}")
            lines.append("")
            top5 = df.head(5)
            for _, row in top5.iterrows():
                lines.append(f"- {row['feature']}: {row['mean_abs_shap']:.4f}")
            lines.append("")

        if overall_importance is not None and not overall_importance.empty:
            lines.append("## Overall Top Features (All Clusters)")
            lines.append("")
            top10 = overall_importance.head(10)
            for _, row in top10.iterrows():
                lines.append(f"- {row['feature']}: {row['mean_abs_shap']:.4f}")
    else:
        lines.append(f"SHAP explanation was not computed.")
        lines.append(f"Reason: {status.get('reason', 'unknown')}")

    path = REPORT_DIR / "shap_cluster_explanation.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    logging.info("SHAP report saved: %s", path)


def main() -> None:
    ensure_directories()
    setup_logging()
    setup_plots()

    params = load_yaml(PARAMS_PATH)
    seed = int(params.get("project", {}).get("random_seed", 42))
    np.random.seed(seed)

    # بارگذاری داده‌ها
    x, labels, features, assignments = load_inputs()
    feature_names = features.columns.tolist()

    # آموزش مدل
    model = train_tree_classifier(x, labels, params, seed)

    # محاسبه SHAP
    shap_sample_size = int(params.get("interpretation", {}).get("shap_sample_size", 2000))
    shap_df, per_cluster_shap, status = compute_shap_per_cluster(
        model=model,
        x=x,
        labels=labels,
        feature_names=feature_names,
        sample_size=shap_sample_size,
        random_seed=seed,
    )

    # ذخیره نتایج
    if per_cluster_shap:
        for cluster_id, df in per_cluster_shap.items():
            df.to_csv(REPORT_DIR / f"shap_cluster_{cluster_id}.csv", index=False)

    # Overall importance
    overall_importance = None
    if shap_df is not None:
        overall_importance = (
            pd.DataFrame(
                {
                    "feature": feature_names,
                    "mean_abs_shap": shap_df[feature_names].abs().mean(axis=0),
                }
            )
            .sort_values("mean_abs_shap", ascending=False)
            .reset_index(drop=True)
        )
        overall_importance.to_csv(REPORT_DIR / "shap_overall_importance.csv", index=False)

        # همچنین ذخیره shap_df برای بررسی
        shap_df.to_parquet(PROCESSED_DIR / "phase3_shap_values.parquet", index=False)

    # رسم نمودارها (اگر داده موجود باشد)
    fig1 = plot_per_cluster_shap(per_cluster_shap, top_k=15)
    fig2 = plot_overall_shap(overall_importance, top_k=20)

    # گزارش متنی
    write_shap_report(
        per_cluster_importance=per_cluster_shap,
        overall_importance=overall_importance,
        model=model,
        x=x,
        labels=labels,
        status=status,
    )

    # ذخیره وضعیت
    completion_record = {
        "status": "completed",
        "shap_status": status,
        "n_clusters": len(np.unique(labels)),
        "sample_size": shap_sample_size,
        "generated_figures": [
            str(fig1.relative_to(ROOT)) if fig1 else None,
            str(fig2.relative_to(ROOT)) if fig2 else None,
        ],
        "model": {
            "type": "DecisionTreeClassifier",
            "max_depth": model.get_depth(),
            "n_leaves": model.get_n_leaves(),
            "training_accuracy": float(model.score(x, labels)),
        },
    }
    save_json(completion_record, REPORT_DIR / "phase3_completion_step3_record.json")
    logging.info("Phase 3 completion step 3 succeeded")


if __name__ == "__main__":
    main()