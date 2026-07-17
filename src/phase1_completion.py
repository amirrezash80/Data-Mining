import json
import logging
import sys
from pathlib import Path
from typing import Any

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from matplotlib.backends.backend_pdf import PdfPages
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr, probplot


ROOT = Path(__file__).resolve().parents[1]
PARAMS_PATH = ROOT / "params.yaml"

DATA_PROCESSED = ROOT / "data" / "processed"
REPORT_DIR = ROOT / "reports" / "phase1"
FIGURE_DIR = ROOT / "reports" / "figures"
NOTEBOOK_DIR = ROOT / "notebooks"

FEATURE_PATH = (
    DATA_PROCESSED
    / "features_unscaled.parquet"
)

METADATA_PATH = (
    DATA_PROCESSED
    / "metadata_and_labels.parquet"
)

ARRAY_PATH = (
    DATA_PROCESSED
    / "phase1_arrays.npz"
)

HOPKINS_PATH = (
    REPORT_DIR
    / "hopkins_results.csv"
)

OUTPUT_PDF = (
    REPORT_DIR
    / "phase1_eda_report.pdf"
)


def ensure_directories() -> None:
    for directory in [
        REPORT_DIR,
        FIGURE_DIR,
        NOTEBOOK_DIR,
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
                / "phase1_completion.log",
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
            f"Configuration not found: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        data = yaml.safe_load(file)

    if not isinstance(data, dict):
        raise ValueError(
            "Invalid params.yaml"
        )

    return data


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


def load_inputs() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    dict[str, np.ndarray],
]:
    required_paths = [
        FEATURE_PATH,
        METADATA_PATH,
        ARRAY_PATH,
    ]

    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(
                f"Required Phase 1 output "
                f"not found: {path}"
            )

    features = pd.read_parquet(
        FEATURE_PATH
    )

    metadata = pd.read_parquet(
        METADATA_PATH
    )

    with np.load(
        ARRAY_PATH,
        allow_pickle=False,
    ) as loaded:
        arrays = {
            name: loaded[name]
            for name in loaded.files
        }

    if len(features) != len(metadata):
        raise ValueError(
            "Feature and metadata row "
            "counts differ"
        )

    if len(features) != len(
        arrays["X_standard"]
    ):
        raise ValueError(
            "Feature and array row "
            "counts differ"
        )

    if features.isna().any().any():
        raise ValueError(
            "Feature table contains "
            "missing values"
        )

    return (
        features,
        metadata,
        arrays,
    )


def choose_informative_features(
    features: pd.DataFrame,
    count: int = 8,
) -> list[str]:
    spread = (
        features.quantile(0.75)
        - features.quantile(0.25)
    )

    non_constant = spread[
        spread > 0
    ]

    selected = (
        non_constant
        .sort_values(
            ascending=False
        )
        .head(count)
        .index
        .tolist()
    )

    preferred = [
        feature
        for feature in [
            "V1",
            "V2",
            "V3",
            "V4",
            "V10",
            "V14",
            "V17",
            "Amount_log1p",
            "Time_sin",
            "Time_cos",
        ]
        if feature in features.columns
    ]

    result = []

    for feature in (
        preferred + selected
    ):
        if feature not in result:
            result.append(feature)

        if len(result) >= count:
            break

    return result


def vat_mst_order(
    distance_matrix: np.ndarray,
) -> np.ndarray:
    distance_matrix = np.asarray(
        distance_matrix,
        dtype=np.float64,
    )

    if (
        distance_matrix.ndim != 2
        or distance_matrix.shape[0]
        != distance_matrix.shape[1]
    ):
        raise ValueError(
            "Distance matrix must be square"
        )

    number_of_records = (
        distance_matrix.shape[0]
    )

    if number_of_records < 2:
        return np.arange(
            number_of_records
        )

    farthest_flat_index = int(
        np.argmax(distance_matrix)
    )

    first_endpoint, second_endpoint = (
        np.unravel_index(
            farthest_flat_index,
            distance_matrix.shape,
        )
    )

    selected = np.zeros(
        number_of_records,
        dtype=bool,
    )

    ordering = np.empty(
        number_of_records,
        dtype=np.int64,
    )

    ordering[0] = first_endpoint
    selected[first_endpoint] = True

    minimum_distance = (
        distance_matrix[
            first_endpoint
        ].copy()
    )

    minimum_distance[
        selected
    ] = np.inf

    parent = np.full(
        number_of_records,
        -1,
        dtype=np.int64,
    )

    parent[:] = first_endpoint

    mst_edges = []

    for position in range(
        1,
        number_of_records,
    ):
        next_record = int(
            np.argmin(
                np.where(
                    selected,
                    np.inf,
                    minimum_distance,
                )
            )
        )

        ordering[position] = (
            next_record
        )

        selected[next_record] = True

        mst_edges.append(
            (
                int(
                    parent[next_record]
                ),
                next_record,
                float(
                    minimum_distance[
                        next_record
                    ]
                ),
            )
        )

        candidate_distances = (
            distance_matrix[
                next_record
            ]
        )

        update_mask = (
            (~selected)
            & (
                candidate_distances
                < minimum_distance
            )
        )

        minimum_distance[
            update_mask
        ] = candidate_distances[
            update_mask
        ]

        parent[
            update_mask
        ] = next_record

        minimum_distance[
            selected
        ] = np.inf

    edge_table = pd.DataFrame(
        mst_edges,
        columns=[
            "parent",
            "child",
            "edge_weight",
        ],
    )

    edge_table.to_csv(
        REPORT_DIR
        / "vat_mst_edges.csv",
        index=False,
    )

    return ordering


def calculate_mst_vat(
    matrix: np.ndarray,
    sample_size: int,
    random_seed: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
]:
    rng = np.random.default_rng(
        random_seed
    )

    sample_size = min(
        sample_size,
        len(matrix),
    )

    sample_indices = rng.choice(
        len(matrix),
        size=sample_size,
        replace=False,
    )

    sample = matrix[
        sample_indices
    ]

    dissimilarity = squareform(
        pdist(
            sample,
            metric="euclidean",
        )
    )

    ordering = vat_mst_order(
        dissimilarity
    )

    ordered_matrix = (
        dissimilarity[
            np.ix_(
                ordering,
                ordering,
            )
        ]
    )

    ordered_indices = (
        sample_indices[
            ordering
        ]
    )

    return (
        ordered_matrix,
        ordered_indices,
    )


def plot_mst_vat(
    vat_matrix: np.ndarray,
) -> Path:
    upper_limit = float(
        np.quantile(
            vat_matrix,
            0.98,
        )
    )

    plt.figure(
        figsize=(9, 8)
    )

    sns.heatmap(
        vat_matrix,
        cmap="gray_r",
        vmin=0,
        vmax=upper_limit,
        xticklabels=False,
        yticklabels=False,
        cbar_kws={
            "label": (
                "Reordered Euclidean "
                "dissimilarity"
            )
        },
    )

    plt.title(
        "MST-based VAT dissimilarity matrix"
    )

    plt.xlabel(
        "MST/VAT reordered observations"
    )

    plt.ylabel(
        "MST/VAT reordered observations"
    )

    return save_figure(
        "phase1_vat_mst_heatmap.png"
    )


def plot_boxplots(
    features: pd.DataFrame,
    selected_features: list[str],
) -> Path:
    sample = features.sample(
        n=min(
            30000,
            len(features),
        ),
        random_state=42,
    )

    normalised = sample[
        selected_features
    ].copy()

    medians = normalised.median()
    iqrs = (
        normalised.quantile(0.75)
        - normalised.quantile(0.25)
    ).replace(0, 1)

    normalised = (
        normalised - medians
    ) / iqrs

    long_data = normalised.melt(
        var_name="feature",
        value_name="robust_scaled_value",
    )

    plt.figure(
        figsize=(13, 6)
    )

    sns.boxplot(
        data=long_data,
        x="feature",
        y="robust_scaled_value",
        showfliers=False,
    )

    plt.xticks(
        rotation=35,
        ha="right",
    )

    plt.title(
        "Feature box plots after "
        "median-IQR normalisation"
    )

    plt.xlabel("Feature")
    plt.ylabel(
        "Median-IQR normalised value"
    )

    return save_figure(
        "phase1_feature_boxplots.png"
    )


def plot_qq_grid(
    features: pd.DataFrame,
    selected_features: list[str],
) -> Path:
    selected = selected_features[:6]

    sample = features.sample(
        n=min(
            10000,
            len(features),
        ),
        random_state=42,
    )

    figure, axes = plt.subplots(
        2,
        3,
        figsize=(14, 9),
    )

    axes = axes.ravel()

    for axis, feature in zip(
        axes,
        selected,
    ):
        probplot(
            sample[feature].to_numpy(),
            dist="norm",
            plot=axis,
        )

        axis.set_title(
            f"QQ plot: {feature}"
        )

    for axis in axes[
        len(selected):
    ]:
        axis.axis("off")

    figure.suptitle(
        "Normal QQ plots for selected features",
        y=1.01,
    )

    return save_figure(
        "phase1_feature_qqplots.png"
    )


def plot_pair_matrix(
    features: pd.DataFrame,
    selected_features: list[str],
) -> Path:
    selected = (
        selected_features[:6]
    )

    sample = features[
        selected
    ].sample(
        n=min(
            3000,
            len(features),
        ),
        random_state=42,
    )

    graph = sns.pairplot(
        sample,
        corner=True,
        diag_kind="kde",
        plot_kws={
            "s": 8,
            "alpha": 0.25,
            "rasterized": True,
        },
        diag_kws={
            "fill": False,
        },
    )

    graph.fig.suptitle(
        "Scatter-plot matrix of selected features",
        y=1.01,
    )

    path = (
        FIGURE_DIR
        / "phase1_feature_pairplot.png"
    )

    graph.savefig(
        path,
        bbox_inches="tight",
        dpi=180,
    )

    plt.close("all")

    logging.info(
        "Saved figure: %s",
        path,
    )

    return path


def calculate_correlation_significance(
    features: pd.DataFrame,
    selected_features: list[str],
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    sample = features[
        selected_features
    ].sample(
        n=min(
            20000,
            len(features),
        ),
        random_state=42,
    )

    number_of_features = len(
        selected_features
    )

    correlations = np.eye(
        number_of_features,
        dtype=float,
    )

    pvalues = np.zeros(
        (
            number_of_features,
            number_of_features,
        ),
        dtype=float,
    )

    for first_index in range(
        number_of_features
    ):
        for second_index in range(
            first_index + 1,
            number_of_features,
        ):
            correlation, pvalue = pearsonr(
                sample.iloc[
                    :,
                    first_index,
                ],
                sample.iloc[
                    :,
                    second_index,
                ],
            )

            correlations[
                first_index,
                second_index,
            ] = correlation

            correlations[
                second_index,
                first_index,
            ] = correlation

            pvalues[
                first_index,
                second_index,
            ] = pvalue

            pvalues[
                second_index,
                first_index,
            ] = pvalue

    correlation_frame = pd.DataFrame(
        correlations,
        index=selected_features,
        columns=selected_features,
    )

    pvalue_frame = pd.DataFrame(
        pvalues,
        index=selected_features,
        columns=selected_features,
    )

    correlation_frame.to_csv(
        REPORT_DIR
        / "correlation_selected_features.csv"
    )

    pvalue_frame.to_csv(
        REPORT_DIR
        / "correlation_pvalues.csv"
    )

    return (
        correlation_frame,
        pvalue_frame,
    )


def plot_significance_masked_correlation(
    correlations: pd.DataFrame,
    pvalues: pd.DataFrame,
) -> Path:
    mask = (
        pvalues.to_numpy()
        > 0.01
    )

    np.fill_diagonal(
        mask,
        False,
    )

    plt.figure(
        figsize=(10, 8)
    )

    sns.heatmap(
        correlations,
        mask=mask,
        cmap="vlag",
        center=0,
        vmin=-1,
        vmax=1,
        annot=True,
        fmt=".2f",
        square=True,
        cbar_kws={
            "label": (
                "Pearson correlation"
            )
        },
    )

    plt.title(
        "Correlation heatmap with "
        "p-value masking at 0.01"
    )

    return save_figure(
        "phase1_significance_masked_correlation.png"
    )


def create_eda_summary_table(
    features: pd.DataFrame,
    selected_features: list[str],
) -> pd.DataFrame:
    summary = (
        features[
            selected_features
        ]
        .describe(
            percentiles=[
                0.01,
                0.05,
                0.25,
                0.50,
                0.75,
                0.95,
                0.99,
            ]
        )
        .T
    )

    summary["skewness"] = (
        features[
            selected_features
        ].skew()
    )

    summary["kurtosis"] = (
        features[
            selected_features
        ].kurtosis()
    )

    summary["iqr"] = (
        features[
            selected_features
        ].quantile(0.75)
        - features[
            selected_features
        ].quantile(0.25)
    )

    summary.to_csv(
        REPORT_DIR
        / "extended_eda_summary.csv"
    )

    return summary


def add_text_page(
    pdf: PdfPages,
    title: str,
    body: str,
) -> None:
    figure = plt.figure(
        figsize=(8.27, 11.69)
    )

    figure.text(
        0.08,
        0.94,
        title,
        fontsize=18,
        weight="bold",
        va="top",
    )

    figure.text(
        0.08,
        0.89,
        body,
        fontsize=10,
        va="top",
        family="monospace",
        wrap=True,
    )

    pdf.savefig(
        figure,
        bbox_inches="tight",
    )

    plt.close(figure)


def add_image_page(
    pdf: PdfPages,
    image_path: Path,
    title: str,
) -> None:
    if not image_path.exists():
        return

    image = mpimg.imread(
        image_path
    )

    figure = plt.figure(
        figsize=(11.69, 8.27)
    )

    axis = figure.add_subplot(111)
    axis.imshow(image)
    axis.axis("off")
    axis.set_title(
        title,
        fontsize=16,
        pad=15,
    )

    pdf.savefig(
        figure,
        bbox_inches="tight",
    )

    plt.close(figure)


def build_phase1_pdf(
    features: pd.DataFrame,
    metadata: pd.DataFrame,
    selected_features: list[str],
    summary_table: pd.DataFrame,
    figure_paths: list[
        tuple[Path, str]
    ],
) -> None:
    if HOPKINS_PATH.exists():
        hopkins = pd.read_csv(
            HOPKINS_PATH
        )

        hopkins_summary = (
            hopkins.groupby(
                "scaler"
            )["hopkins"]
            .agg(
                [
                    "mean",
                    "std",
                    "min",
                    "max",
                ]
            )
            .round(6)
            .to_string()
        )
    else:
        hopkins_summary = (
            "Hopkins results were not found."
        )

    class_counts = (
        metadata["Class"]
        .value_counts()
        .sort_index()
        .to_string()
        if "Class" in metadata.columns
        else "Class column unavailable."
    )

    dataset_text = (
        f"Records: {len(features):,}\n"
        f"Clustering features: "
        f"{features.shape[1]}\n\n"
        f"Selected EDA features:\n"
        f"{', '.join(selected_features)}\n\n"
        f"External class distribution "
        f"(not used for clustering):\n"
        f"{class_counts}"
    )

    tendency_text = (
        "Hopkins repeated results\n\n"
        f"{hopkins_summary}\n\n"
        "Interpretation guide:\n"
        "H near 0.5 indicates random-like "
        "structure.\n"
        "H above approximately 0.70-0.75 "
        "supports strong cluster tendency.\n\n"
        "VAT must be interpreted jointly "
        "with Hopkins and low-dimensional "
        "embeddings. Diagonal block structure "
        "supports cluster tendency."
    )

    feature_summary_text = (
        summary_table[
            [
                "mean",
                "std",
                "50%",
                "iqr",
                "skewness",
                "kurtosis",
            ]
        ]
        .round(4)
        .to_string()
    )

    with PdfPages(
        OUTPUT_PDF
    ) as pdf:
        add_text_page(
            pdf,
            "Phase 1: Clustering-Oriented EDA",
            dataset_text,
        )

        add_text_page(
            pdf,
            "Clustering Tendency Assessment",
            tendency_text,
        )

        add_text_page(
            pdf,
            "Selected Feature Statistics",
            feature_summary_text,
        )

        for image_path, title in (
            figure_paths
        ):
            add_image_page(
                pdf,
                image_path,
                title,
            )

        metadata_pdf = pdf.infodict()

        metadata_pdf[
            "Title"
        ] = (
            "Phase 1 Clustering-Oriented "
            "EDA Report"
        )

        metadata_pdf[
            "Author"
        ] = (
            "Credit Card Clustering Project"
        )

        metadata_pdf[
            "Subject"
        ] = (
            "Data preparation, EDA, "
            "Hopkins and MST-based VAT"
        )

    logging.info(
        "Saved Phase 1 PDF: %s",
        OUTPUT_PDF,
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
        features,
        metadata,
        arrays,
    ) = load_inputs()

    selected_features = (
        choose_informative_features(
            features,
            count=8,
        )
    )

    summary_table = (
        create_eda_summary_table(
            features,
            selected_features,
        )
    )

    vat_size = min(
        int(
            params["sampling"].get(
                "vat_size",
                1000,
            )
        ),
        1500,
    )

    train_indices = arrays[
        "train_indices"
    ].astype(np.int64)

    standard_train = arrays[
        "X_standard"
    ][train_indices].astype(
        np.float64
    )

    (
        vat_matrix,
        vat_ordered_indices,
    ) = calculate_mst_vat(
        standard_train,
        sample_size=vat_size,
        random_seed=random_seed,
    )

    np.save(
        DATA_PROCESSED
        / "vat_mst_reordered_matrix.npy",
        vat_matrix.astype(
            np.float32
        ),
    )

    pd.DataFrame(
        {
            "ordered_training_position":
                vat_ordered_indices
        }
    ).to_csv(
        REPORT_DIR
        / "vat_mst_ordering.csv",
        index=False,
    )

    vat_figure = plot_mst_vat(
        vat_matrix
    )

    boxplot_figure = plot_boxplots(
        features,
        selected_features,
    )

    qq_figure = plot_qq_grid(
        features,
        selected_features,
    )

    pairplot_figure = plot_pair_matrix(
        features,
        selected_features,
    )

    correlations, pvalues = (
        calculate_correlation_significance(
            features,
            selected_features,
        )
    )

    correlation_figure = (
        plot_significance_masked_correlation(
            correlations,
            pvalues,
        )
    )

    existing_figures = [
        (
            FIGURE_DIR
            / "phase1_amount_log_transform.png",
            "Amount Transformation",
        ),
        (
            FIGURE_DIR
            / "phase1_scaling_before_after.png",
            "Before and After Scaling",
        ),
        (
            FIGURE_DIR
            / "phase1_pca_explained_variance.png",
            "PCA Explained Variance",
        ),
        (
            FIGURE_DIR
            / "phase1_pca_density.png",
            "PCA Density Projection",
        ),
        (
            FIGURE_DIR
            / "phase1_umap_density.png",
            "UMAP Density Projection",
        ),
        (
            FIGURE_DIR
            / "phase1_hopkins_results.png",
            "Repeated Hopkins Results",
        ),
    ]

    new_figures = [
        (
            vat_figure,
            "MST-Based VAT",
        ),
        (
            boxplot_figure,
            "Selected Feature Box Plots",
        ),
        (
            qq_figure,
            "Selected Feature QQ Plots",
        ),
        (
            pairplot_figure,
            "Selected Feature Pair Plot",
        ),
        (
            correlation_figure,
            "Significance-Masked Correlation",
        ),
    ]

    figure_paths = [
        item
        for item in (
            existing_figures
            + new_figures
        )
        if item[0].exists()
    ]

    build_phase1_pdf(
        features=features,
        metadata=metadata,
        selected_features=(
            selected_features
        ),
        summary_table=summary_table,
        figure_paths=figure_paths,
    )

    completion_record = {
        "status": "completed",
        "selected_features": (
            selected_features
        ),
        "vat_method": (
            "MST-based Prim ordering"
        ),
        "vat_sample_size": int(
            len(vat_matrix)
        ),
        "generated_figures": [
            str(
                path.relative_to(ROOT)
            )
            for path, _ in figure_paths
        ],
        "generated_pdf": str(
            OUTPUT_PDF.relative_to(
                ROOT
            )
        ),
    }

    save_json(
        completion_record,
        REPORT_DIR
        / "phase1_completion_record.json",
    )

    logging.info(
        "Phase 1 completion succeeded"
    )


if __name__ == "__main__":
    main()