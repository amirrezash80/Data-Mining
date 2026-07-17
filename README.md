# Credit Card Transaction Clustering

## Research question

Do credit-card transactions form natural and density-distinct subpopulations, and do small clusters or anomalous observations meaningfully overlap fraudulent transactions?

## Project stages

1. Phase 1: ingestion, cleaning, feature engineering, scaling, dimensionality reduction, Hopkins, VAT, and EDA
2. Phase 2: K-Means, hierarchical clustering, DBSCAN, GMM, determining k, internal and external evaluation, stability, and agreement
3. Phase 3: ensemble-consensus clustering, interpretation, exemplars, rules, SHAP, anomaly analysis, and preprocessing sensitivity
4. Delivery layer: schema validation, artifact versioning, registry, temporal drift monitoring, dashboard, live assignment, and final report

## Environment

Create a virtual environment:

    python -m venv .venv

Windows:

    .venv\Scripts\activate

Linux or macOS:

    source .venv/bin/activate

Install all dependencies:

    pip install -r requirements.txt
    pip install -r requirements_delivery.txt

## Run individual phases

    python src/phase1.py
    python src/phase2.py
    python src/phase3.py
    python src/deployment.py all

## Run the complete pipeline

    python src/run_project.py

## Validate schemas

    python src/deployment.py validate

## Build assignment registry

    python src/deployment.py registry

## Run drift monitoring

    python src/deployment.py monitor

## Generate artifact manifest

    python src/deployment.py manifest

## Generate final report

    python src/deployment.py report

The generated report is stored at:

    reports/final/final_technical_report.md

## Assign new transactions

The input CSV must include:

- Time
- Amount
- V1 through V28

Run:

    python src/deployment.py assign --input new_transactions.csv --output assigned_transactions.csv

The output includes:

- assigned_cluster
- nearest_centroid_distance
- distance_to_cluster_0
- distance_to_cluster_1
- other cluster-distance columns

## Dashboard

Run:

    streamlit run dashboard/app.py

Dashboard pages:

1. Overview
2. Cluster Explorer
3. Evaluation
4. Live Assignment

## Important assignment limitation

The Phase 3 consensus model is based on hierarchical clustering of a co-association matrix and has no native out-of-sample prediction method.

New records are assigned using the nearest final consensus-cluster centroid in the fitted PCA space. This is an operational proxy and not an exact reconstruction of consensus clustering.

## Main deployment outputs

    models/registry/consensus_assignment_registry.npz
    models/registry/standard_scaler.joblib
    models/registry/pca_standard.joblib
    models/registry/registry_metadata.json
    reports/deployment/schema_validation.json
    reports/deployment/feature_drift_report.csv
    reports/deployment/drift_summary.json
    reports/deployment/artifact_manifest.json
    reports/deployment/execution_record.json
    reports/final/final_technical_report.md

## Reproducibility

All random seeds are fixed in the YAML configuration files.

Major artifacts are hashed with SHA-256.

The fraud label is excluded from preprocessing, clustering, hyperparameter selection, consensus construction, live assignment, and anomaly scoring. It is used only for post-hoc external evaluation.

## AI disclosure

External AI assistance materially contributed to code structure, debugging, documentation, and experiment planning. All generated code must be reviewed, executed, understood, and defensible by the project team.