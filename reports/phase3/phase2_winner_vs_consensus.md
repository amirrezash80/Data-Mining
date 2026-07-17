# Phase 3 Advanced-Track Comparison

The Phase 3 consensus clustering was compared directly with the recommended Phase 2 model.

Both methods were evaluated on the same records and in the same Phase 1 PCA space.

The fraud label was not used to fit either method or choose their hyperparameters. External metrics are post-hoc only.

## Phase 2 Winner

- Algorithm: Hierarchical
- Original recommendation parameters: k=2,linkage=single
- Selection source: reports/phase2/phase2_final_recommendation.json
- Number of clusters: 2
- Silhouette: 0.887794
- Davies-Bouldin: 0.077300
- Calinski-Harabasz: 131.019865
- External ARI: -0.000666

## Phase 3 Consensus

- Number of clusters: 2
- Silhouette: 0.826426
- Davies-Bouldin: 0.353934
- Calinski-Harabasz: 153.018384
- External ARI: 0.498447

## Changes Produced by Consensus

- Silhouette change: -0.061368
- Davies-Bouldin improvement: -0.276634
- Calinski-Harabasz change: 21.998519
- External ARI change: 0.499113
- ARI between the two partitions: -0.000600

## Conclusion

Consensus produces a mixed result: one internal metric improves while the others do not.

A failure of consensus to outperform the Phase 2 winner is still a valid scientific result. Consensus is intended to reconcile algorithm disagreement and can improve robustness even when geometric separation does not improve.