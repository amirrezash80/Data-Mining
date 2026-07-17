import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(ROOT),
    )

from src.deployment import assign_dataframe
from src.deployment import load_params


st.set_page_config(
    page_title="Credit Card Clustering",
    page_icon="📊",
    layout="wide",
)


PARAMS = load_params()


def resolve_path(value: str) -> Path:
    return ROOT / value


@st.cache_data
def load_assignments() -> pd.DataFrame:
    path = resolve_path(
        PARAMS["data"][
            "consensus_assignments"
        ]
    )

    if not path.exists():
        return pd.DataFrame()

    return pd.read_parquet(path)


@st.cache_data
def load_phase1_arrays() -> dict[str, np.ndarray]:
    path = resolve_path(
        PARAMS["data"]["phase1_arrays"]
    )

    if not path.exists():
        return {}

    with np.load(
        path,
        allow_pickle=False,
    ) as arrays:
        return {
            name: arrays[name]
            for name in arrays.files
        }


@st.cache_data
def load_csv(path_value: str) -> pd.DataFrame:
    path = resolve_path(path_value)

    if not path.exists():
        return pd.DataFrame()

    return pd.read_csv(path)


@st.cache_data
def load_json(path: Path) -> dict:
    if not path.exists():
        return {}

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def overview_page() -> None:
    st.title(
        "Credit Card Transaction Clustering"
    )

    assignments = load_assignments()
    arrays = load_phase1_arrays()

    if assignments.empty or not arrays:
        st.error(
            "Project outputs are missing. Run all phases first."
        )
        return

    cluster_column = "Consensus_cluster"

    first, second, third, fourth = st.columns(
        4
    )

    first.metric(
        "Dataset records",
        f"{len(arrays['y']):,}",
    )

    second.metric(
        "Consensus sample",
        f"{len(assignments):,}",
    )

    third.metric(
        "Clusters",
        int(
            assignments[
                cluster_column
            ].nunique()
        ),
    )

    fourth.metric(
        "PCA dimensions",
        int(
            arrays["X_pca"].shape[1]
        ),
    )

    st.subheader("Selected method")

    st.write(
        "Ensemble and consensus clustering with "
        "average-linkage clustering on one minus "
        "the co-association matrix."
    )

    st.subheader("Cluster sizes")

    sizes = (
        assignments[
            cluster_column
        ]
        .value_counts()
        .sort_index()
        .rename_axis("cluster")
        .reset_index(name="size")
    )

    figure = px.bar(
        sizes,
        x="cluster",
        y="size",
        color="cluster",
        title="Consensus cluster sizes",
    )

    st.plotly_chart(
        figure,
        use_container_width=True,
    )

    fraud_path = PARAMS[
        "reports"
    ]["fraud_composition"]

    fraud = load_csv(fraud_path)

    if not fraud.empty:
        st.subheader(
            "Post-hoc fraud composition"
        )

        st.dataframe(
            fraud,
            use_container_width=True,
            hide_index=True,
        )


def explorer_page() -> None:
    st.title("Cluster Explorer")

    assignments = load_assignments()
    arrays = load_phase1_arrays()

    if assignments.empty or not arrays:
        st.error(
            "Cluster assignments are unavailable."
        )
        return

    indices = assignments[
        "array_index"
    ].to_numpy(dtype=np.int64)

    pca = arrays["X_pca"][indices]

    explorer = assignments.copy()
    explorer["PC1"] = pca[:, 0]
    explorer["PC2"] = pca[:, 1]

    available_clusters = sorted(
        explorer[
            "Consensus_cluster"
        ].unique().tolist()
    )

    selected_clusters = st.multiselect(
        "Clusters",
        options=available_clusters,
        default=available_clusters,
    )

    maximum_points = st.slider(
        "Maximum displayed points",
        min_value=200,
        max_value=min(
            10000,
            len(explorer),
        ),
        value=min(
            2500,
            len(explorer),
        ),
        step=100,
    )

    filtered = explorer[
        explorer[
            "Consensus_cluster"
        ].isin(selected_clusters)
    ]

    if len(filtered) > maximum_points:
        filtered = filtered.sample(
            maximum_points,
            random_state=42,
        )

    hover_columns = [
        column
        for column in [
            "row_id",
            "Time",
            "Amount",
            "Class",
        ]
        if column in filtered.columns
    ]

    figure = px.scatter(
        filtered,
        x="PC1",
        y="PC2",
        color=filtered[
            "Consensus_cluster"
        ].astype(str),
        hover_data=hover_columns,
        opacity=0.65,
        title="Consensus clusters in PCA space",
        labels={
            "color": "Cluster",
        },
    )

    st.plotly_chart(
        figure,
        use_container_width=True,
    )

    labels = load_csv(
        PARAMS["reports"]["cluster_labels"]
    )

    profiles = load_csv(
        PARAMS[
            "reports"
        ]["cluster_profiles"]
    )

    selected_cluster = st.selectbox(
        "Profile cluster",
        options=available_clusters,
    )

    if not labels.empty:
        label_row = labels[
            labels["cluster"]
            == selected_cluster
        ]

        if not label_row.empty:
            st.subheader("Cluster label")
            st.write(
                label_row.iloc[0][
                    "proposed_domain_label"
                ]
            )

    if not profiles.empty:
        profile_row = profiles[
            profiles["cluster"]
            == selected_cluster
        ]

        if not profile_row.empty:
            st.subheader("Cluster profile")

            profile_long = (
                profile_row
                .drop(
                    columns=[
                        column
                        for column in [
                            "cluster",
                            "size",
                            "fraction",
                        ]
                        if column
                        in profile_row.columns
                    ]
                )
                .T
                .reset_index()
            )

            profile_long.columns = [
                "feature_statistic",
                "value",
            ]

            profile_long = profile_long[
                profile_long[
                    "feature_statistic"
                ].str.endswith("_z")
            ]

            profile_long[
                "absolute_value"
            ] = profile_long[
                "value"
            ].abs()

            profile_long = (
                profile_long
                .sort_values(
                    "absolute_value",
                    ascending=False,
                )
                .head(15)
            )

            profile_figure = px.bar(
                profile_long,
                x="value",
                y="feature_statistic",
                orientation="h",
                title=(
                    "Strongest deviations "
                    "from global means"
                ),
            )

            st.plotly_chart(
                profile_figure,
                use_container_width=True,
            )


def evaluation_page() -> None:
    st.title("Evaluation")

    phase2 = load_csv(
        PARAMS[
            "reports"
        ]["phase2_comparison"]
    )

    phase3 = load_csv(
        PARAMS[
            "reports"
        ]["phase3_comparison"]
    )

    if not phase2.empty:
        st.subheader(
            "Phase 2 final comparison"
        )

        st.dataframe(
            phase2,
            use_container_width=True,
            hide_index=True,
        )

    if not phase3.empty:
        st.subheader(
            "Consensus comparison"
        )

        st.dataframe(
            phase3,
            use_container_width=True,
            hide_index=True,
        )

        metric_columns = [
            column
            for column in [
                "silhouette",
                "davies_bouldin",
                "calinski_harabasz",
                "ari",
                "nmi",
                "ami",
            ]
            if column in phase3.columns
        ]

        melted = phase3.melt(
            id_vars=["algorithm"],
            value_vars=metric_columns,
            var_name="metric",
            value_name="value",
        )

        figure = px.bar(
            melted,
            x="algorithm",
            y="value",
            color="algorithm",
            facet_col="metric",
            facet_col_wrap=3,
            title=(
                "Consensus versus "
                "best base clustering"
            ),
        )

        figure.update_yaxes(
            matches=None
        )

        st.plotly_chart(
            figure,
            use_container_width=True,
        )

    sensitivity = load_csv(
        PARAMS[
            "reports"
        ]["sensitivity"]
    )

    if not sensitivity.empty:
        st.subheader(
            "Preprocessing sensitivity"
        )

        first_column = (
            sensitivity.columns[0]
        )

        matrix = sensitivity.set_index(
            first_column
        )

        heatmap = px.imshow(
            matrix,
            text_auto=".3f",
            zmin=0,
            zmax=1,
            color_continuous_scale="Viridis",
            title=(
                "Pairwise ARI between "
                "preprocessing alternatives"
            ),
        )

        st.plotly_chart(
            heatmap,
            use_container_width=True,
        )

    drift_directory = resolve_path(
        PARAMS[
            "reports"
        ]["output_directory"]
    )

    drift_path = (
        drift_directory
        / "feature_drift_report.csv"
    )

    if drift_path.exists():
        drift = pd.read_csv(drift_path)

        st.subheader(
            "Temporal feature drift"
        )

        st.dataframe(
            drift.sort_values(
                "psi",
                ascending=False,
            ),
            use_container_width=True,
            hide_index=True,
        )

        drift_figure = px.bar(
            drift.sort_values(
                "psi",
                ascending=False,
            ),
            x="feature",
            y="psi",
            color="status",
            title=(
                "Population Stability "
                "Index by feature"
            ),
        )

        st.plotly_chart(
            drift_figure,
            use_container_width=True,
        )


def create_manual_record() -> pd.DataFrame:
    values: dict[str, float] = {}

    first, second = st.columns(2)

    with first:
        values["Time"] = st.number_input(
            "Time",
            min_value=0.0,
            value=0.0,
            step=1.0,
        )

    with second:
        values["Amount"] = st.number_input(
            "Amount",
            min_value=0.0,
            value=1.0,
            step=1.0,
        )

    columns = st.columns(4)

    for index in range(1, 29):
        column = columns[
            (index - 1) % 4
        ]

        with column:
            values[f"V{index}"] = (
                st.number_input(
                    f"V{index}",
                    value=0.0,
                    format="%.6f",
                    key=f"manual_v_{index}",
                )
            )

    return pd.DataFrame([values])


def live_assignment_page() -> None:
    st.title("Live Cluster Assignment")

    registry_path = (
        resolve_path(
            PARAMS["models"][
                "registry_directory"
            ]
        )
        / "consensus_assignment_registry.npz"
    )

    if not registry_path.exists():
        st.error(
            "Assignment registry is unavailable. "
            "Run python src/deployment.py all"
        )
        return

    mode = st.radio(
        "Input method",
        options=[
            "Upload CSV",
            "Manual record",
        ],
        horizontal=True,
    )

    if mode == "Upload CSV":
        uploaded = st.file_uploader(
            "Upload a CSV containing Time, Amount, and V1 through V28",
            type=["csv"],
        )

        if uploaded is not None:
            frame = pd.read_csv(uploaded)

            st.subheader("Input preview")

            st.dataframe(
                frame.head(),
                use_container_width=True,
                hide_index=True,
            )

            if st.button(
                "Assign uploaded records",
                type="primary",
            ):
                try:
                    assigned = assign_dataframe(
                        frame,
                        PARAMS,
                    )

                    st.success(
                        "Assignment completed"
                    )

                    st.dataframe(
                        assigned,
                        use_container_width=True,
                        hide_index=True,
                    )

                    st.download_button(
                        "Download assignments",
                        data=assigned.to_csv(
                            index=False
                        ).encode("utf-8"),
                        file_name=(
                            "cluster_assignments.csv"
                        ),
                        mime="text/csv",
                    )

                except Exception as error:
                    st.exception(error)

    else:
        manual_frame = create_manual_record()

        if st.button(
            "Assign record",
            type="primary",
        ):
            try:
                assigned = assign_dataframe(
                    manual_frame,
                    PARAMS,
                )

                cluster_id = int(
                    assigned.iloc[0][
                        "assigned_cluster"
                    ]
                )

                distance = float(
                    assigned.iloc[0][
                        "nearest_centroid_distance"
                    ]
                )

                first, second = st.columns(2)

                first.metric(
                    "Assigned cluster",
                    cluster_id,
                )

                second.metric(
                    "Nearest-centroid distance",
                    f"{distance:.4f}",
                )

                distance_columns = [
                    column
                    for column in assigned.columns
                    if column.startswith(
                        "distance_to_cluster_"
                    )
                ]

                distance_frame = (
                    assigned[
                        distance_columns
                    ]
                    .T
                    .reset_index()
                )

                distance_frame.columns = [
                    "cluster",
                    "distance",
                ]

                distance_frame["cluster"] = (
                    distance_frame["cluster"]
                    .str.replace(
                        "distance_to_cluster_",
                        "",
                        regex=False,
                    )
                )

                figure = px.bar(
                    distance_frame,
                    x="cluster",
                    y="distance",
                    title=(
                        "Distance to every "
                        "cluster centroid"
                    ),
                )

                st.plotly_chart(
                    figure,
                    use_container_width=True,
                )

            except Exception as error:
                st.exception(error)


def main() -> None:
    st.sidebar.title("Navigation")

    page = st.sidebar.radio(
        "Page",
        options=[
            "Overview",
            "Cluster Explorer",
            "Evaluation",
            "Live Assignment",
        ],
    )

    if page == "Overview":
        overview_page()

    elif page == "Cluster Explorer":
        explorer_page()

    elif page == "Evaluation":
        evaluation_page()

    elif page == "Live Assignment":
        live_assignment_page()


if __name__ == "__main__":
    main()