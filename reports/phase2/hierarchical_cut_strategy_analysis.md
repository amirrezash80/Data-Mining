# Hierarchical Cutting Strategy Analysis

Two flat-clustering strategies were compared for single, complete, average, and Ward linkage.

The fixed-height strategy uses a documented fraction of the maximum dendrogram height. The same deterministic rule is applied without consulting the fraud label.

The maximum-silhouette strategy evaluates the candidate k range and selects the partition with the highest internal silhouette coefficient. External metrics are computed only afterward.

Best internal result:

- Linkage: single
- Strategy: fixed_height
- Requested k: nan
- Actual k: 2
- Silhouette: 0.888179
- Davies-Bouldin: 0.077451
- Calinski-Harabasz: 135.475219

Agreement between the two strategies:

- single: ARI=1.000000, fixed k=2, max-silhouette k=2
- complete: ARI=1.000000, fixed k=2, max-silhouette k=2
- average: ARI=0.666333, fixed k=3, max-silhouette k=2
- ward: ARI=0.013881, fixed k=4, max-silhouette k=2

Interpretation:

ARI values near one indicate that both cutting strategies recover nearly the same partition. Low ARI indicates that the flat clustering is sensitive to how the dendrogram is cut.

Single linkage may exhibit chaining, while Ward linkage typically produces tighter and more balanced clusters. Cophenetic correlation measures how faithfully each dendrogram preserves the original pairwise distances.

The fraud label was not used to choose the linkage, height, or number of clusters.