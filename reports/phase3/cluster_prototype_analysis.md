# Cluster Prototypes and Exemplars

The final consensus clusters were interpreted using four representative records per cluster:

- the exact medoid
- the point with the highest silhouette
- the most marginal boundary point
- the record nearest to the arithmetic centroid

The medoid is an actual observed record whose total Euclidean distance to all other records in its cluster is minimal.

This differs from a centroid exemplar. A centroid exemplar is the observed record nearest to the arithmetic mean, but it does not necessarily minimise total pairwise distance.

Number of clusters: 2

Clusters where medoid and centroid exemplar are the same record: 2

Clusters where they differ: 0

Mean reduction in total within-cluster distance from using the true medoid: 0.000000

Maximum reduction in total within-cluster distance: 0.000000

Boundary points with negative silhouette: 0

A negative boundary-point silhouette indicates that the selected record is, on average, closer to another cluster than to its assigned cluster.

The fraud label was not used to select any prototype. It is attached only for post-hoc inspection.

Because V1 through V28 are anonymised PCA variables, prototypes support statistical interpretation but cannot reveal direct business semantics for those variables.