# Phase 2 Integrated Clustering Report

## 1. Objective

Phase 2 implements, tunes, and compares clustering algorithms from four families on the same Phase 1 representation:

- Partitioning: K-Means
- Hierarchical: Agglomerative clustering
- Density-based: DBSCAN
- Model-based: Gaussian Mixture Model

The fraud label is excluded from clustering, hyperparameter tuning, determining k, distance selection, and stability selection. It is used only after model selection for external evaluation.

## 2. Experimental Data

The analysis uses the PCA representation produced in Phase 1. Hyperparameter searches are conducted on a reproducible representative subset selected from the Phase 1 training partition without consulting the fraud label.

## 3. Determining the Number of Clusters

The following methods were applied:

- Elbow and Kneedle on K-Means inertia
- Average Silhouette
- Gap Statistic
- Davies-Bouldin and Calinski-Harabasz
- GMM BIC and AIC
- Bootstrap Jaccard stability

Summary:

    {
      "silhouette_selected_k": 2,
      "maximum_kmeans_silhouette": 0.1807099228063333,
      "kmeans_candidate_range": [
        2,
        8
      ],
      "gap_selected_k": 2,
      "gmm_bic_selected_k": 8,
      "gmm_bic_selected_covariance": "full",
      "minimum_gmm_bic": -57689.97935307604
    }

### 3.1 K-Means search

| k | inertia | silhouette | davies_bouldin | calinski_harabasz | runtime_seconds |
| --- | --- | --- | --- | --- | --- |
| 2.000000 | 166769.443281 | 0.180710 | 2.221315 | 284.363918 | 0.433233 |
| 3.000000 | 160269.100044 | 0.172688 | 1.831121 | 269.539993 | 0.284849 |
| 4.000000 | 155624.513003 | 0.066251 | 3.311503 | 244.675253 | 0.236943 |
| 5.000000 | 149556.328773 | 0.063597 | 2.884860 | 251.735894 | 0.269988 |
| 6.000000 | 143946.895063 | 0.068681 | 2.520130 | 255.914092 | 0.264779 |
| 7.000000 | 139675.376201 | 0.075250 | 2.422593 | 250.292920 | 0.319053 |
| 8.000000 | 136322.638226 | 0.075780 | 2.327733 | 240.829273 | 0.330827 |

### 3.2 Gap Statistic

| k | gap | standard_error | real_log_dispersion |
| --- | --- | --- | --- |
| 2.000000 | 3.802711 | 0.002599 | 12.024368 |
| 3.000000 | 3.784802 | 0.001602 | 11.984610 |
| 4.000000 | 3.772821 | 0.003470 | 11.955201 |
| 5.000000 | 3.777274 | 0.000482 | 11.928082 |
| 6.000000 | 3.806339 | 0.001378 | 11.877200 |
| 7.000000 | 3.818852 | 0.003617 | 11.847076 |
| 8.000000 | 3.821046 | 0.002979 | 11.830251 |

### 3.3 GMM information criteria

| k | covariance_type | bic | aic | silhouette | davies_bouldin | calinski_harabasz | converged | runtime_seconds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2.000000 | spherical | 444396.200366 | 444000.928995 | 0.441371 | 5.105948 | 107.523651 | True | 0.042794 |
| 3.000000 | spherical | 434193.343384 | 433597.086571 | 0.031612 | 5.403127 | 115.285279 | True | 0.082170 |
| 4.000000 | spherical | 429871.015640 | 429073.773385 | 0.026632 | 4.942724 | 128.998078 | True | 0.120463 |
| 5.000000 | spherical | 424498.158562 | 423499.930865 | 0.047497 | 4.647531 | 159.620538 | True | 0.221655 |
| 6.000000 | spherical | 421133.329081 | 419934.115941 | 0.010651 | 5.264796 | 140.033494 | True | 0.275331 |
| 7.000000 | spherical | 417392.154462 | 415991.955879 | -0.014738 | 4.866535 | 135.342314 | True | 0.162962 |
| 8.000000 | spherical | 413944.851855 | 412343.667831 | -0.002957 | 4.660760 | 144.603062 | True | 0.258076 |
| 2.000000 | diag | 419820.845546 | 419063.800380 | 0.326206 | 5.982776 | 100.244208 | True | 0.052262 |
| 3.000000 | diag | 407270.670209 | 406131.752702 | 0.048922 | 6.044471 | 104.036067 | True | 0.202156 |
| 4.000000 | diag | 402202.010505 | 400681.220657 | 0.028134 | 5.307150 | 119.498697 | True | 0.097634 |
| 5.000000 | diag | 397560.903924 | 395658.241736 | 0.020071 | 4.658774 | 145.423297 | True | 0.295053 |
| 6.000000 | diag | 393125.705606 | 390841.171077 | 0.020847 | 5.053041 | 133.215665 | True | 0.205555 |
| 7.000000 | diag | 392469.621477 | 389803.214607 | -0.015435 | 4.889128 | 123.662594 | True | 0.279528 |
| 8.000000 | diag | 385951.913164 | 382903.633954 | -0.006658 | 4.658804 | 134.555920 | True | 0.292230 |
| 2.000000 | tied | 473887.168420 | 470785.293091 | 0.180710 | 2.221315 | 284.363918 | True | 0.076741 |
| 3.000000 | tied | 463407.519249 | 460111.357993 | 0.045459 | 4.108116 | 201.985397 | True | 0.615334 |
| 4.000000 | tied | 458856.418610 | 455365.971426 | 0.067132 | 3.297552 | 241.040647 | True | 0.529360 |
| 5.000000 | tied | 453904.771300 | 450220.038188 | 0.089762 | 2.672263 | 198.833191 | True | 0.368897 |
| 6.000000 | tied | 440667.859717 | 436788.840678 | 0.069192 | 2.499437 | 252.462910 | True | 0.579973 |
| 7.000000 | tied | 436935.485018 | 432862.180051 | 0.073013 | 2.407009 | 246.911011 | True | 0.614416 |
| 8.000000 | tied | 440045.329861 | 435777.738967 | 0.061947 | 2.651397 | 209.243382 | True | 0.736284 |
| 2.000000 | full | 277759.962476 | 271938.084160 | 0.132242 | 6.043100 | 130.230290 | True | 0.239555 |
| 3.000000 | full | 56470.695017 | 47734.527785 | -0.000527 | 5.828279 | 121.586213 | True | 0.888937 |
| 4.000000 | full | 25726.945901 | 14076.489754 | 0.017735 | 4.990747 | 145.663436 | True | 1.023973 |
| 5.000000 | full | -30490.840396 | -45055.585459 | 0.033590 | 4.592479 | 171.036010 | True | 1.108237 |
| 6.000000 | full | -38215.385388 | -55694.419366 | 0.037666 | 4.226037 | 152.225885 | True | 0.987955 |
| 7.000000 | full | -46328.979737 | -66722.302631 | 0.032578 | 4.173295 | 140.946003 | True | 0.869679 |
| 8.000000 | full | -57689.979353 | -80997.591162 | 0.036175 | 4.147337 | 134.711995 | True | 1.074491 |

## 4. Hierarchical Clustering

Single, complete, average, and Ward linkage were evaluated. Cophenetic correlation was used to assess dendrogram fidelity. Two cutting strategies were compared:

- fixed-height cut
- cut selected by maximum Silhouette

| linkage | strategy | requested_k | actual_k | cut_height | cophenetic_correlation | silhouette | davies_bouldin | calinski_harabasz | ari | runtime_seconds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| single | fixed_height |  | 2.000000 | 34.478127 | 0.944904 | 0.888179 | 0.077451 | 135.475219 | -0.000750 | 0.032975 |
| single | maximum_silhouette | 2.000000 | 2.000000 | 41.516011 | 0.944904 | 0.888179 | 0.077451 | 135.475219 | -0.000750 | 0.032975 |
| complete | fixed_height |  | 2.000000 | 62.133381 | 0.778717 | 0.888179 | 0.077451 | 135.475219 | -0.000750 | 0.054998 |
| complete | maximum_silhouette | 2.000000 | 2.000000 | 74.406799 | 0.778717 | 0.888179 | 0.077451 | 135.475219 | -0.000750 | 0.054998 |
| average | fixed_height |  | 3.000000 | 43.216433 | 0.953854 | 0.858334 | 0.091098 | 114.303790 | -0.001200 | 0.054827 |
| average | maximum_silhouette | 2.000000 | 2.000000 | 55.174508 | 0.953854 | 0.888179 | 0.077451 | 135.475219 | -0.000750 | 0.054827 |
| ward | fixed_height |  | 4.000000 | 60.886684 | 0.274256 | 0.108171 | 1.327806 | 108.653100 | -0.002735 | 0.055549 |
| ward | maximum_silhouette | 2.000000 | 2.000000 | 78.697567 | 0.274256 | 0.888179 | 0.077451 | 135.475219 | -0.000750 | 0.055549 |

## 5. Internal and External Evaluation

Internal metrics determine the recommendation:

- Silhouette: higher is better
- Davies-Bouldin: lower is better
- Calinski-Harabasz: higher is better

External metrics are reported post-hoc:

- ARI
- NMI
- AMI
- Fowlkes-Mallows
- Homogeneity
- Completeness
- V-measure
- Purity

### 5.1 Final selected configurations

| algorithm | parameters | selected_k | n_clusters | noise_fraction | silhouette | davies_bouldin | calinski_harabasz | ari | nmi | ami | purity | runtime_seconds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| KMeans | k=2 | 2.000000 | 2.000000 | 0.000000 | 0.180710 | 2.221315 | 284.363918 | 0.003678 | 0.000506 | -0.000146 | 0.998000 | 0.162626 |
| Hierarchical | k=2,linkage=single | 2.000000 | 2.000000 | 0.000000 | 0.889233 | 0.077408 | 132.713361 | -0.000307 | 0.000042 | -0.000260 | 0.998000 | 0.354003 |
| DBSCAN | eps=11.405267,min_samples=30 | 1.000000 | 1.000000 | 0.004167 |  |  |  | 0.267077 | 0.159430 | 0.158384 | 0.998000 | 0.591787 |
| GMM | k=8,covariance=full | 8.000000 | 8.000000 | 0.000000 | 0.036175 | 4.147337 | 134.711995 | 0.001426 | 0.005139 | 0.004422 | 0.998000 | 1.175882 |

### 5.2 Internal ranking scoreboard

| final_order | algorithm | parameters | selected_k | n_clusters | noise_fraction | silhouette | davies_bouldin | calinski_harabasz | ari | nmi | ami | purity | runtime_seconds | seed_ari_mean | internal_rank_sum |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1.000000 | Hierarchical | k=2,linkage=single | 2.000000 | 2.000000 | 0.000000 | 0.889233 | 0.077408 | 132.713361 | -0.000307 | 0.000042 | -0.000260 | 0.998000 | 0.354003 |  | 5.000000 |
| 2.000000 | KMeans | k=2 | 2.000000 | 2.000000 | 0.000000 | 0.180710 | 2.221315 | 284.363918 | 0.003678 | 0.000506 | -0.000146 | 0.998000 | 0.162626 | 0.296726 | 5.000000 |
| 3.000000 | GMM | k=8,covariance=full | 8.000000 | 8.000000 | 0.000000 | 0.036175 | 4.147337 | 134.711995 | 0.001426 | 0.005139 | 0.004422 | 0.998000 | 1.175882 | 0.555697 | 8.000000 |

## 6. Stability Analysis

### 6.1 Seed stability

K-Means and GMM were rerun under multiple random seeds. Pairwise ARI between runs measures sensitivity to initialisation.

| algorithm | pair_id | pairwise_ari |
| --- | --- | --- |
| KMeans | 0.000000 | 0.112166 |
| KMeans | 1.000000 | 0.000850 |
| KMeans | 2.000000 | -0.004590 |
| KMeans | 3.000000 | 0.000850 |
| KMeans | 4.000000 | 0.000862 |
| KMeans | 5.000000 | 0.017855 |
| KMeans | 6.000000 | -0.009180 |
| KMeans | 7.000000 | 0.016904 |
| KMeans | 8.000000 | 0.009508 |
| KMeans | 9.000000 | 0.009508 |
| KMeans | 10.000000 | -0.010072 |
| KMeans | 11.000000 | 0.000850 |
| KMeans | 12.000000 | 0.020557 |
| KMeans | 13.000000 | 1.000000 |
| KMeans | 14.000000 | 0.000862 |
| KMeans | 15.000000 | 0.001014 |
| KMeans | 16.000000 | 0.001078 |
| KMeans | 17.000000 | -0.007058 |
| KMeans | 18.000000 | 0.001319 |
| KMeans | 19.000000 | 0.118225 |
| KMeans | 20.000000 | -0.009229 |
| KMeans | 21.000000 | 0.117767 |
| KMeans | 22.000000 | 0.117536 |
| KMeans | 23.000000 | 0.167419 |
| KMeans | 24.000000 | -0.003036 |
| KMeans | 25.000000 | 0.196120 |
| KMeans | 26.000000 | -0.000596 |
| KMeans | 27.000000 | -0.000596 |
| KMeans | 28.000000 | 0.314473 |
| KMeans | 29.000000 | 0.118225 |
| KMeans | 30.000000 | 0.211780 |
| KMeans | 31.000000 | 0.112166 |
| KMeans | 32.000000 | 0.117078 |
| KMeans | 33.000000 | 0.117539 |
| KMeans | 34.000000 | 0.118921 |
| KMeans | 35.000000 | 0.301593 |
| KMeans | 36.000000 | 0.119848 |
| KMeans | 37.000000 | 0.007363 |
| KMeans | 38.000000 | 0.998667 |
| KMeans | 39.000000 | 0.994008 |

### 6.2 Bootstrap stability

Bootstrap resampling was evaluated through ARI to a reference partition, cluster-membership Jaccard, and the co-clustering probability matrix.

| k | bootstrap_runs | ari_mean | ari_std | mean_jaccard | jaccard_std | minimum_jaccard | worst_cluster_mean_jaccard |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2.000000 | 20.000000 | 0.355487 | 0.485303 | 0.667667 | 0.252211 | 0.077333 | 0.404129 |
| 3.000000 | 20.000000 | 0.463588 | 0.450320 | 0.576291 | 0.223414 | 0.003896 | 0.301858 |
| 4.000000 | 20.000000 | 0.502063 | 0.288295 | 0.639160 | 0.150074 | 0.132231 | 0.292574 |
| 5.000000 | 20.000000 | 0.348175 | 0.177945 | 0.653562 | 0.091092 | 0.001529 | 0.392839 |
| 6.000000 | 20.000000 | 0.644150 | 0.290763 | 0.682745 | 0.119909 | 0.008547 | 0.090933 |
| 7.000000 | 20.000000 | 0.619225 | 0.218955 | 0.661689 | 0.105018 | 0.004425 | 0.122063 |
| 8.000000 | 20.000000 | 0.614862 | 0.142767 | 0.638100 | 0.096093 | 0.004608 | 0.083487 |

## 7. Algorithm Agreement

Pairwise ARI between final algorithms was calculated and visualised as a heatmap.

    {
      "mean_pairwise_algorithm_ari": 0.02803467564198787,
      "minimum_pairwise_algorithm_ari": -0.0003030895691132,
      "maximum_pairwise_algorithm_ari": 0.0851571609017775
    }

Low agreement indicates that alternative clustering assumptions recover different transaction structures. This disagreement is scientifically informative rather than necessarily an implementation error.

## 8. Per-Point Silhouette and Error Analysis

The lowest-silhouette records of the preferred clustering were inspected. Negative values indicate probable misassignment, while values near zero indicate boundary observations.

    {
      "preferred_algorithm": "Hierarchical",
      "analysed_record_count": 100,
      "minimum_silhouette": 0.0,
      "mean_silhouette": 0.6953249807921668,
      "negative_silhouette_count": 0,
      "negative_silhouette_fraction": 0.0,
      "boundary_count": 1,
      "boundary_fraction": 0.01,
      "diagnostic_category_counts": {
        "weakly_separated": 99,
        "boundary_case": 1
      },
      "fraud_rate_lowest_silhouette": 0.06,
      "amount_median_lowest_silhouette": 35.445,
      "amount_mean_lowest_silhouette": 533.3472999999999
    }

The full record-level analysis is stored in `lowest_silhouette_records.csv`.

## 9. Distance-Metric Sensitivity

Agglomerative clustering with average linkage was evaluated under Euclidean and Manhattan distances while holding the data, linkage, and candidate k values fixed.

| distance_metric | requested_k | actual_k | silhouette_matching_metric | silhouette_euclidean | silhouette_manhattan | davies_bouldin | calinski_harabasz | ari_external | runtime_seconds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| euclidean | 2.000000 | 2.000000 | 0.888051 | 0.888051 | 0.889609 | 0.077300 | 131.019865 | -0.000666 | 0.097131 |
| manhattan | 2.000000 | 2.000000 | 0.889609 | 0.888051 | 0.889609 | 0.077300 | 131.019865 | -0.000666 | 0.089531 |

The agreement between the two selected partitions is documented in `distance_metric_selected_agreement.json`.

## 10. Final Recommendation

The recommended Phase 2 clustering is Hierarchical with parameters k=2,linkage=single. The recommended number of clusters is k=2. This recommendation was determined only from internal clustering metrics. The selected configuration obtained Silhouette=0.88923, Davies-Bouldin=0.07741, and Calinski-Harabasz=132.71336. It achieved the strongest aggregate rank across Silhouette, Davies-Bouldin, and Calinski-Harabasz among the final candidate configurations. Its Silhouette coefficient was 0.889233, indicating strong geometric separation. Its bootstrap membership stability was classified as moderate.

### Reasons

- It achieved the strongest aggregate rank across Silhouette, Davies-Bouldin, and Calinski-Harabasz among the final candidate configurations.
- Its Silhouette coefficient was 0.889233, indicating strong geometric separation.
- Its bootstrap membership stability was classified as moderate.

### Limitations

- The external fraud label was not used for model selection; external scores are post-hoc descriptive evidence only.
- The fraud class is highly imbalanced, so purity can be misleading.
- A transaction profile cluster is not the same thing as a fraud class.
- Sensitivity to changing Euclidean distance to Manhattan distance was classified as low.
- Among the inspected lowest-silhouette records, the negative-silhouette fraction was 0.000000.
- Hierarchical partitions depend on linkage choice and dendrogram cutting strategy.

## 11. Answer to the Research Question

The Phase 2 experiments indicate whether transaction records admit stable profile clusters under classical clustering assumptions. The recommended partition represents transaction-profile structure rather than a direct reconstruction of the binary fraud label.

External fraud metrics and fraud composition indicate whether discovered profiles are enriched for fraud. Low external agreement does not automatically invalidate the clustering, because fraud may occur across multiple transaction profiles rather than form one isolated cluster.

## 12. Reproducibility

All reported values are read from persisted experiment outputs. Random seeds are fixed in `params_phase2.yaml`. The recommendation is regenerated by running:

    python src/phase2_completion_step5.py

## 13. Generated Artifacts

- `phase2_unified_performance_table.csv`
- `phase2_final_scoreboard.csv`
- `phase2_final_recommendation.json`
- `phase2_integrated_report.md`
- `phase2_integrated_report.pdf`
