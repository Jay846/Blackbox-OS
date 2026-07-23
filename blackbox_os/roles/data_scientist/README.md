# Data Scientist Agent: Role & Skill Mappings

This directory manages the 77 data science skills divided across 12 functional categories, coordinated by 4 operational workflows (SOPs).

## Workflows (SOPs) & Skill Category Distribution

To keep LLM context library size small ($N \le 15$ per step), we isolate categories into specific sub-graphs:

### 1. Data Preparation & Ingestion Sub-Graph (`workflows/data_prep.py`)
*   **Missing Data (6 skills)**: Mean, median, KNN, MICE, forward fill, missing indicator flagging.
*   **Scaling & Encoding (7 skills)**: StandardScaler, MinMaxScaler, RobustScaler, One-Hot, Target, Ordinal encoding, Log transform.
*   **Imbalanced Data (6 skills)**: SMOTE, random undersampling, class weights, ADASYN, Tomek links, balanced bagging.

### 2. Feature Selection & Modeling Sub-Graph (`workflows/feature_model.py`)
*   **Feature Selection (8 skills)**: RFE, Mutual Information, Chi-Square, Lasso selection, Variance threshold, Correlation filter, Forward/Backward selection.
*   **Regularization (7 skills)**: L1/L2, ElasticNet, Dropout, Early stopping, Weight decay, Max norm.
*   **Ensembling (6 skills)**: RandomForest, GradientBoosting, Stacking, Bagging, Voting, Blending.

### 3. Optimization & Evaluation Sub-Graph (`workflows/opt_eval.py`)
*   **Hyperparameter Tuning (6 skills)**: Grid search, Random search, Bayesian optimization, Hyperband, PBT, Optuna runner.
*   **Resampling & Validation (8 skills)**: K-Fold, Stratified K-Fold, Leave-One-Out, Nested CV, Bootstrap, Holdout, Time-Series CV, Repeated K-Fold.
*   **Evaluation Metrics (7 skills)**: Precision-Recall, F1 score, ROC-AUC, Confusion matrix, Log loss, MCC, Calibration curve.

### 4. Compliance & Deployment Sub-Graph (`workflows/deploy_monitor.py`)
*   **Data Leakage/Integrity (5 skills)**: Train-test leakage scan, Target leakage detector, Duplicate row detector, Temporal leakage check, Group leakage check.
*   **Interpretability (5 skills)**: SHAP values, Permutation importance, PDP, LIME, Feature ranker.
*   **Deployment & Monitoring (6 skills)**: Model drift, Data drift, A/B test evaluator, Champion-Challenger, Model registry, Latency monitoring.

---

## Skill Definition Schema
All skill schemas are stored as JSON files under `roles/data_scientist/skills/` using a standardized structure:
```json
{
  "id": "skill_name",
  "category": "missing_data",
  "description": "Short description of what the skill does",
  "boundary_conditions": "When to use and when NOT to use",
  "example_invocation": "Code or parameter example"
}
```
