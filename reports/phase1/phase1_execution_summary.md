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

