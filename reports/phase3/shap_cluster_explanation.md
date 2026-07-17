# SHAP-based Cluster Explanation

A shallow decision tree was trained to predict cluster membership.
SHAP values were then computed for a sampled subset of records.

For each cluster, the mean absolute SHAP value of each feature is
reported as the feature's influence on that cluster's assignment.

Model tree depth: 1
Number of leaves: 2
Training accuracy: 0.9892

SHAP computation status: skipped
SHAP explanation was not computed.
Reason: shap not installed