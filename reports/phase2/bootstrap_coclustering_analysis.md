# Bootstrap Stability and Co-clustering Analysis

Bootstrap repetitions per k: 20

Fixed anchor records: 1500

A reference K-Means model was fitted for each candidate k. Each bootstrap repetition resampled the evaluation data with replacement, fitted a new K-Means model, and predicted cluster membership for the same fixed anchor records.

The fraud label was not used in resampling, model fitting, cluster matching, or stability selection.

## Selected K-Means k

- Selected k from the original silhouette search: 2
- Mean bootstrap ARI: 0.355487
- Mean cluster Jaccard: 0.667667
- Worst-cluster mean Jaccard: 0.404129

## Most Stable Candidate

- k: 6
- Mean cluster Jaccard: 0.682745
- Mean ARI: 0.644150

## Co-clustering Matrix Diagnostics

- Mean within-cluster probability: 0.974986
- Mean between-cluster probability: 0.619445
- Probability separation: 0.355541

High within-cluster probability combined with low between-cluster probability indicates a stable partition. Blurred blocks or high between-cluster probability indicate unstable boundaries.

Cluster-level Jaccard scores are obtained by matching each reference cluster to the bootstrap cluster with maximum membership-set Jaccard similarity.