# Clustering Analysis on Credit Card Transactions

## Abstract

This project investigates whether credit-card transactions form natural and density-distinct subpopulations and whether small clusters or anomalous observations are enriched for fraudulent transactions. The analysis uses the anonymised European credit-card transaction dataset containing PCA-transformed transaction variables, transaction time, amount, and an external fraud label. The fraud label is excluded from preprocessing, dimensionality reduction, clustering, hyperparameter selection, consensus construction, and anomaly scoring. It is used only for post-hoc external evaluation.

The project proceeds through reproducible preprocessing, clustering-tendency assessment, comparison of partitioning, hierarchical, density-based, and model-based algorithms, and an advanced ensemble-consensus analysis. The advanced method combines multiple clusterings generated under different algorithms, values of k, random seeds, PCA dimensions, and covariance structures. Cluster interpretation is performed through statistical profiles, exemplars, boundary cases, shallow decision-tree rules, and optional SHAP explanations. A downstream anomaly score ranks transactions by robust within-cluster distance. Sensitivity to preprocessing is assessed by comparing partitions under standard scaling, robust scaling, and PCA. A production layer adds schema validation, artifact hashing, model registration, temporal drift monitoring, and live nearest-centroid assignment.

All numerical claims in this report are produced by the committed scripts and persisted artifacts.

## 1. Introduction

The central research question is:

Do credit-card transactions form natural density-distinct subpopulations, and do small clusters or low-density observations meaningfully overlap fraudulent transactions?

The project does not assume in advance that fraud forms one isolated cluster. Fraud may instead occur across several transaction profiles or appear as rare observations within otherwise legitimate clusters.

## 2. Data and Preprocessing

# Phase 1 Execution Summary

## Dataset

- Raw rows: 284,807
- Clean rows: 283,726
- Engineered features: 31
- Training records: 226,980
- Held-out temporal records: 56,746
- Raw SHA-256: `76274b691b16a6c49d3f159c883398e03ccd6d1ee12d9d8ee38f4b4b98551a89`

## Label-isolation policy

The `Class` field was excluded from feature engineering, scaling,
PCA, UMAP, Hopkins, VAT, and distance diagnostics.

It is retained only for external evaluation in Phase 2.

## Engineered representation

The clustering matrix contains:

- Variables `V1` through `V28`
- `log(1 + Amount)`
- `Time_sin`
- `Time_cos`

## Scaling

Two scaling methods were fitted on the training partition:

- StandardScaler
- RobustScaler

The held-out partition was transformed using the same fitted
parameters.

## PCA

- Selected principal components: 28
- Explained variance: 0.951958

## Hopkins repeated results

```text
              mean       std       min  max
scaler                                     
robust    0.999999  0.000001  0.999997  1.0
standard  0.999957  0.000096  0.999785  1.0



## 3. Algorithm Portfolio

# Phase 2 Execution Summary

## Experimental policy

All hyperparameter selection was performed using internal metrics
only. The `Class` fraud label was used after fitting solely to compute
external evaluation metrics.

The evaluation subset was selected randomly from the Phase 1 training
indices without consulting `Class`.

## Determining the number of clusters

### K-Means

- Elbow/Kneedle recommendation: `3`
- Maximum-silhouette recommendation: `2`
- Gap-statistic recommendation: `2`

Disagreement between these methods is not automatically an error.
Each method measures a different aspect of cluster structure.

### Hierarchical clustering

- Selected linkage: `single`
- Selected k: `2`

Single, complete, average, and Ward linkages were compared using
silhouette and cophenetic correlation.

### DBSCAN

Selected configuration:

`eps=11.405267,min_samples=30`

DBSCAN does not require a predefined k. Its `eps` candidates were
derived from k-nearest-neighbour distance quantiles.

### Gaussian Mixture Model

- BIC-selected components: `8`
- BIC-selected covariance structure: `full`

## Final algorithm comparison

| algorithm    | parameters                   |   n_clusters |   noise_fraction |   silhouette |   davies_bouldin |   calinski_harabasz |      ari |     nmi |      ami |   purity |   runtime_seconds |
|:-------------|:-----------------------------|-------------:|-----------------:|-------------:|-----------------:|--------------------:|---------:|--------:|---------:|---------:|------------------:|
| KMeans       | k=2                          |            2 |          0       |      0.18071 |          2.22131 |             284.364 |  0.00368 | 0.00051 | -0.00015 |    0.998 |           0.1642  |
| Hierarchical | k=2,linkage=single           |            2 |          0       |      0.88923 |          0.07741 |             132.713 | -0.00031 | 4e-05   | -0.00026 |    0.998 |           0.35094 |
| DBSCAN       | eps=11.405267,min_samples=30 |            1 |          0.00417 |    nan       |        nan       |             nan     |  0.26708 | 0.15943 |  0.15838 |    0.998 |           0.64519 |
| GMM          | k=8,covariance=full          |            8 |          0       |      0.03618 |          4.14734 |             134.712 |  0.00143 | 0.00514 |  0.00442 |    0.998 |           1.33283 |

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


## 4. Phase 2 Final Comparison

   algorithm                   parameters  n_records  n_clusters  noise_fraction  runtime_seconds  silhouette  davies_bouldin  calinski_harabasz       ari      nmi       ami  fowlkes_mallows  homogeneity  completeness  v_measure  purity
      KMeans                          k=2       6000           2        0.000000         0.164199    0.180710        2.221315         284.363918  0.003678 0.000506 -0.000146         0.920106     0.005222      0.000266   0.000506   0.998
Hierarchical           k=2,linkage=single       6000           2        0.000000         0.350935    0.889233        0.077408         132.713361 -0.000307 0.000042 -0.000260         0.997835     0.000023      0.000206   0.000042   0.998
      DBSCAN eps=11.405267,min_samples=30       6000           1        0.004167         0.645193         NaN             NaN                NaN  0.267077 0.159430  0.158384         0.995494     0.228865      0.122319   0.159430   0.998
         GMM          k=8,covariance=full       6000           8        0.000000         1.332828    0.036175        4.147337         134.711995  0.001426 0.005139  0.004422         0.459180     0.319368      0.002590   0.005139   0.998

## 5. Advanced Consensus Track

Not generated.

## 6. Consensus versus Best Base Model

        algorithm                                          parameters  requested_k  n_records  n_clusters  silhouette  davies_bouldin  calinski_harabasz      ari      nmi      ami  fowlkes_mallows  homogeneity  completeness  v_measure  purity
Best base: KMeans base_id=0,k=2,dimension=5,seed=42.0,covariance=None            2       2500           2    0.826426        0.353934         153.018384 0.498447 0.370936 0.370245         0.998397     0.304635      0.474125   0.370936  0.9984
        Consensus                                 k=2,linkage=average            2       2500           2    0.826426        0.353934         153.018384 0.498447 0.370936 0.370245         0.998397     0.304635      0.474125   0.370936  0.9984

## 7. Cluster Interpretation

### 7.1 Proposed cluster labels

 cluster                                     proposed_domain_label
       0      Cluster 0: low V17, low V3, low V7, low V12, low V10
       1 Cluster 1: high V17, high V3, high V7, high V12, high V10

### 7.2 Cluster profiles

 cluster  size  fraction    V1_mean  V1_median      V1_z   V2_mean  V2_median      V2_z    V3_mean  V3_median       V3_z  V4_mean  V4_median      V4_z    V5_mean  V5_median      V5_z   V6_mean  V6_median      V6_z    V7_mean  V7_median       V7_z   V8_mean  V8_median      V8_z   V9_mean  V9_median      V9_z   V10_mean  V10_median      V10_z  V11_mean  V11_median     V11_z   V12_mean  V12_median      V12_z  V13_mean  V13_median     V13_z   V14_mean  V14_median      V14_z  V15_mean  V15_median     V15_z  V16_mean  V16_median     V16_z   V17_mean  V17_median      V17_z  V18_mean  V18_median     V18_z  V19_mean  V19_median     V19_z  V20_mean  V20_median     V20_z  V21_mean  V21_median     V21_z  V22_mean  V22_median     V22_z  V23_mean  V23_median     V23_z  V24_mean  V24_median     V24_z  V25_mean  V25_median     V25_z  V26_mean  V26_median     V26_z  V27_mean  V27_median     V27_z  V28_mean  V28_median     V28_z  Amount_log1p_mean  Amount_log1p_median  Amount_log1p_z  Time_sin_mean  Time_sin_median  Time_sin_z  Time_cos_mean  Time_cos_median  Time_cos_z
       0     3    0.0012 -15.327642 -13.247210 -7.519638  9.623988   9.213994  5.921649 -19.276289 -16.969412 -12.164753 6.905690   6.203314  4.916502 -12.493024 -10.760276 -8.628519 -4.182816  -4.192171 -3.094103 -14.450028 -14.077086 -11.332556 10.230109   8.986474  7.994958 -3.508174  -3.683242 -3.356802 -11.279043   -9.946409 -11.092350  7.091623    7.868726  6.663353 -11.984114  -12.324822 -11.286505  0.617411    0.461808  0.603078 -10.806113   -9.637468 -10.922850 -0.300764   -0.241095 -0.365234 -8.876604   -8.185039 -9.765728 -14.266568  -14.310254 -15.643327 -5.161660   -4.884467 -6.157438  2.222498    1.433108  2.660879  0.813919    1.476576  0.984864  1.956471    1.762232  2.681733 -0.836202   -1.065086 -1.136595 -0.808548   -0.951043 -1.220343  0.188623    0.134565  0.283188  0.797172    0.777812  1.505927 -0.176027   -0.222671 -0.397125  0.610633    1.527655  1.518020  0.201328    0.453699  0.644603           4.498406              4.51075        0.778601       0.807711         0.926665    1.527493       0.094148        -0.198085    0.552262
       1  2497    0.9988  -0.062695   0.035121  0.009034 -0.021089   0.061331 -0.007115   0.163678   0.331054   0.014615 0.073185   0.038151 -0.005907  -0.070402  -0.105509  0.010367  0.019996  -0.248266  0.003717   0.028388   0.028151   0.013615 -0.020380   0.029949 -0.009605 -0.026090  -0.092712  0.004033  -0.013977   -0.098944   0.013327  0.073306    0.028548 -0.008006  -0.002706    0.121075   0.013560 -0.001202   -0.026475 -0.000725   0.034919    0.083603   0.013123  0.041987    0.099468  0.000439  0.017192    0.069871  0.011733   0.017251   -0.048859   0.018795 -0.029607   -0.056276  0.007398  0.030970    0.006774 -0.003197 -0.032726   -0.051461 -0.001183 -0.009274   -0.038717 -0.003222 -0.039656   -0.014549  0.001366 -0.007767   -0.018148  0.001466  0.014371    0.071891 -0.000340  0.047369    0.094026 -0.001809  0.016559   -0.037499  0.000477  0.003394    0.002431 -0.001824  0.001956    0.015196 -0.000774           3.180305              3.15700       -0.000935      -0.157538        -0.255165   -0.001835      -0.293548        -0.610779   -0.000664

### 7.3 Interpretation limitation

Variables V1 through V28 are anonymised PCA features. Consequently, cluster labels describe statistical deviations rather than direct business concepts. The amount and temporal features permit limited operational interpretation, but the hidden semantics of the PCA variables prevent strong causal or behavioural naming.

## 8. Downstream Anomaly Analysis

{
  "threshold": 8.432275339515975,
  "quantile": 0.99,
  "top_n": 100,
  "flagged_count": 25,
  "flagged_fraction": 0.01,
  "fraud_rate_all": 0.002,
  "fraud_rate_top_n": 0.03,
  "fraud_rate_flagged": 0.08
}

The anomaly score represents unusualness relative to the assigned cluster. It is not a calibrated probability of fraud.

## 9. Post-hoc Fraud Composition

 cluster  size  fraud_count  fraud_rate  global_fraud_rate  fraud_lift
       0     3            2    0.666667              0.002  333.333333
       1  2497            3    0.001201              0.002    0.600721

Purity and raw fraud rates must be interpreted cautiously because the external fraud class is severely imbalanced.

## 10. Fairness and Sensitivity

The dataset does not contain explicit demographic attributes such as gender, race, age, or socio-economic category. A demographic fairness audit is therefore not possible. Fraud composition by cluster is an outcome audit and is not equivalent to a fairness audit.

### 10.1 Preprocessing sensitivity

 representation  StandardScaler  RobustScaler  PCA_of_Standard
 StandardScaler        1.000000      0.442663         0.598364
   RobustScaler        0.442663      1.000000         0.458847
PCA_of_Standard        0.598364      0.458847         1.000000

High disagreement between preprocessing alternatives weakens the claim that the discovered partition is unique or intrinsic.

## 11. Production Pipeline

The production pipeline validates input schema, checks row consistency, versions artifacts with SHA-256 hashes, serialises the scaler, PCA reducer, consensus centroids, cluster identifiers, data version, fit date, and metric scoreboard.

Consensus hierarchical clustering does not provide a native prediction method for unseen observations. Live assignment is therefore implemented as a documented nearest-centroid proxy in the fitted PCA space. This operational approximation must not be confused with reconstructing the original consensus procedure for each new record.

### 11.1 Registry metadata

{
  "model_type": "Nearest-centroid assignment proxy for Phase 3 consensus clusters",
  "fit_date_utc": "2026-07-17T19:06:19.579748+00:00",
  "cluster_ids": [
    0,
    1
  ],
  "cluster_sizes": [
    3,
    2497
  ],
  "pca_dimensions": 28,
  "training_sample_size": 2500,
  "distance_metric": "euclidean",
  "class_used_for_assignment": false,
  "artifact_hashes": {
    "phase1_arrays": "c836b06b9a0a44c0ac692cb6fe7cf16a49f346720507554cda3700597ebf1dcf",
    "assignments": "9a723c13e93987ee3f49dd17b2f164884337a5f29b038a58e438964b2a6a22a0",
    "scaler": "82c239afed05080e92cceb429326da2798bc4f08f4aab36d8c4683d30f4e7d5d",
    "reducer": "9b4311834dd031217ed30daacd03bf8cd5a9ebca0546372fda664f46a4b26631",
    "registry": "ea17051b9d99e7b31ca1c351fc837d8229184271776558dc4390e7439103aca0"
  },
  "metric_scoreboard": {
    "reports/phase2/final_comparison.csv": [
      {
        "algorithm": "KMeans",
        "parameters": "k=2",
        "n_records": 6000,
        "n_clusters": 2,
        "noise_fraction": 0.0,
        "runtime_seconds": 0.1641990419999999,
        "silhouette": 0.1807099228063333,
        "davies_bouldin": 2.221314555862618,
        "calinski_harabasz": 284.36391756604394,
        "ari": 0.0036781537324972,
        "nmi": 0.0005055702711922,
        "ami": -0.0001459688576425,
        "fowlkes_mallows": 0.9201063227034392,
        "homogeneity": 0.0052223315543302,
        "completeness": 0.0002656435173062,
        "v_measure": 0.0005055702711922,
        "purity": 0.998
      },
      {
        "algorithm": "Hierarchical",
        "parameters": "k=2,linkage=single",
        "n_records": 6000,
        "n_clusters": 2,
        "noise_fraction": 0.0,
        "runtime_seconds": 0.3509353750000006,
        "silhouette": 0.8892326041611816,
        "davies_bouldin": 0.077408343023204,
        "calinski_harabasz": 132.71336077206786,
        "ari": -0.0003072213362133,
        "nmi": 4.159802853790664e-05,
        "ami": -0.0002600086159431,
        "fowlkes_mallows": 0.9978346575911162,
        "homogeneity": 2.3129547387896395e-05,
        "completeness": 0.0002064213472209,
        "v_measure": 4.159802853790664e-05,
        "purity": 0.998
      },
      {
        "algorithm": "DBSCAN",
        "parameters": "eps=11.405267,min_samples=30",
        "n_records": 6000,
        "n_clusters": 1,
        "noise_fraction": 0.0041666666666666,
        "runtime_seconds": 0.6451932910000018,
        "silhouette": NaN,
        "davies_bouldin": NaN,
        "calinski_harabasz": NaN,
        "ari": 0.2670772327223387,
        "nmi": 0.1594299011906836,
        "ami": 0.1583838164908061,
        "fowlkes_mallows": 0.995494131663156,
        "homogeneity": 0.2288652175797255,
        "completeness": 0.1223194559502206,
        "v_measure": 0.1594299011906836,
        "purity": 0.998
      },
      {
        "algorithm": "GMM",
        "parameters": "k=8,covariance=full",
        "n_records": 6000,
        "n_clusters": 8,
        "noise_fraction": 0.0,
        "runtime_seconds": 1.3328284590000052,
        "silhouette": 0.0361752746535297,
        "davies_bouldin": 4.147336611969541,
        "calinski_harabasz": 134.7119948854929,
        "ari": 0.0014262658432302,
        "nmi": 0.0051390207385563,
        "ami": 0.0044224929609562,
        "fowlkes_mallows": 0.459179628025137,
        "homogeneity": 0.3193680304024034,
        "completeness": 0.002590351323766,
        "v_measure": 0.0051390207385563,
        "purity": 0.998
      }
    ],
    "reports/phase3/advanced_method_comparison.csv": [
      {
        "algorithm": "Best base: KMeans",
        "parameters": "base_id=0,k=2,dimension=5,seed=42.0,covariance=None",
        "requested_k": 2,
        "n_records": 2500,
        "n_clusters": 2,
        "silhouette": 0.8264262802538963,
        "davies_bouldin": 0.3539339428615566,
        "calinski_harabasz": 153.01838364327142,
        "ari": 0.4984469116037456,
        "nmi": 0.3709361204065082,
        "ami": 0.3702454332044394,
        "fowlkes_mallows": 0.998397116931514,
        "homogeneity": 0.3046353335816212,
        "completeness": 0.4741245040436679,
        "v_measure": 0.3709361204065082,
        "purity": 0.9984
      },
      {
        "algorithm": "Consensus",
        "parameters": "k=2,linkage=average",
        "requested_k": 2,
        "n_records": 2500,
        "n_clusters": 2,
        "silhouette": 0.8264262802538963,
        "davies_bouldin": 0.3539339428615566,
        "calinski_harabasz": 153.01838364327142,
        "ari": 0.4984469116037456,
        "nmi": 0.3709361204065082,
        "ami": 0.3702454332044394,
        "fowlkes_mallows": 0.998397116931514,
        "homogeneity": 0.3046353335816212,
        "completeness": 0.4741245040436679,
        "v_measure": 0.3709361204065082,
        "purity": 0.9984
      }
    ]
  }
}

### 11.2 Temporal drift monitoring

{
  "created_at_utc": "2026-07-17T19:06:22.456346+00:00",
  "reference_records": 226980,
  "current_records": 56746,
  "stable_features": 16,
  "warning_features": 9,
  "refit_features": 6,
  "maximum_psi": 8.611627938723117,
  "maximum_ks": 0.5780817702505449,
  "refit_recommended": true
}

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
