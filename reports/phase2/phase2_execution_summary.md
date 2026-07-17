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
| KMeans       | k=2                          |            2 |          0       |      0.18071 |          2.22131 |             284.364 |  0.00368 | 0.00051 | -0.00015 |    0.998 |           0.16263 |
| Hierarchical | k=2,linkage=single           |            2 |          0       |      0.88923 |          0.07741 |             132.713 | -0.00031 | 4e-05   | -0.00026 |    0.998 |           0.354   |
| DBSCAN       | eps=11.405267,min_samples=30 |            1 |          0.00417 |    nan       |        nan       |             nan     |  0.26708 | 0.15943 |  0.15838 |    0.998 |           0.59179 |
| GMM          | k=8,covariance=full          |            8 |          0       |      0.03618 |          4.14734 |             134.712 |  0.00143 | 0.00514 |  0.00442 |    0.998 |           1.17588 |

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
