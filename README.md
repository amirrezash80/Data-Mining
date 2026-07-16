# Credit Card Fraud Clustering

## Research question

Do credit-card transactions form natural density-distinct
subpopulations, and do small clusters or low-density observations
meaningfully overlap fraudulent transactions?

## Dataset

The project uses the anonymised European credit-card fraud dataset:

- 284,807 original transactions
- 28 PCA-transformed variables
- Time and Amount
- Class label retained only for external evaluation

The Class label must not be used in preprocessing, dimensionality
reduction, clustering, selection of k, or hyperparameter tuning.

## Phase 1

Phase 1 implements:

1. Reproducible data acquisition
2. Raw-data persistence
3. Schema validation
4. Data profiling
5. Exact-duplicate analysis
6. Feature engineering
7. StandardScaler and RobustScaler comparison
8. PCA diagnostics
9. UMAP visualisation
10. Euclidean, Manhattan and Mahalanobis diagnostics
11. Repeated Hopkins statistic
12. VAT tendency visualisation
13. Artifact hashing and environment manifest

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt