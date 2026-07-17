# Distance-Metric Sensitivity Analysis

Agglomerative clustering with average linkage was held fixed while only the distance metric was changed.

The experiment used the same records, PCA representation, candidate k values, linkage criterion, and random sample for both metrics.

Analysis sample size: 2500

Euclidean distance is natural for standardised continuous features and emphasises larger coordinate differences.

Manhattan distance is less dominated by a small number of large coordinate deviations and can be more robust to heavy-tailed observations.

The fraud label was not used to select the distance metric or k. It was consulted only for post-hoc external metrics.

## Euclidean Selection

- Selected k: 2
- Matching-metric silhouette: 0.888051
- Davies-Bouldin: 0.077300
- Calinski-Harabasz: 131.019865
- External ARI: -0.000666

## Manhattan Selection

- Selected k: 2
- Matching-metric silhouette: 0.889609
- Davies-Bouldin: 0.077300
- Calinski-Harabasz: 131.019865
- External ARI: -0.000666

## Partition Agreement

- ARI between independently selected partitions: 1.000000
- Sensitivity level: low
- Minimum same-k ARI: 0.332533
- Maximum same-k ARI: 1.000000

## Interpretation

The two metrics recover highly similar partitions. The cluster structure is therefore relatively robust to replacing Euclidean distance with Manhattan distance.

Davies-Bouldin and Calinski-Harabasz are based on Euclidean geometry in their standard scikit-learn implementations. They are retained as common-space comparators, while matching-metric silhouette is the primary metric-specific criterion.

External fraud metrics are descriptive only. They did not participate in metric or hyperparameter selection.