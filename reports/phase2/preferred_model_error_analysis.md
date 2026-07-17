# Phase 2 Preferred-Clustering Error Analysis

Preferred algorithm: Hierarchical

Selected parameters: k=2,linkage=single

The preferred algorithm was selected using internal metrics only. Fraud labels were not used for algorithm selection.

Records with the lowest silhouette coefficients were inspected as potential boundary cases, probable misassignments, or weakly separated observations.

Analysed records: 100

Minimum silhouette: 0.000000

Mean silhouette among inspected records: 0.695325

Negative-silhouette fraction: 0.000000

Boundary fraction: 0.010000

Diagnostic categories:

{
  "weakly_separated": 99,
  "boundary_case": 1
}

Interpretation guidance:

- Negative silhouette indicates that a record is, on average, closer to another cluster than to its assigned cluster.
- Values near zero indicate boundary cases.
- A small centroid-distance margin indicates ambiguity between the assigned and nearest alternative cluster.
- Fraud composition is evaluated only after the unsupervised model has been selected.

Post-hoc fraud rate among inspected records: 0.060000
