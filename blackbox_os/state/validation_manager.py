from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from blackbox_os.state.shared_state import SharedState

class ValidationContract(BaseModel):
    """
    Defines the acceptance thresholds and weight parameters for model approval.
    """
    w_metric: float = Field(default=1.0, description="Weight for performance metrics (F1/MCC)")
    w_leakage: float = Field(default=1.5, description="Penalty weight for target/group leakage")
    w_drift: float = Field(default=0.5, description="Penalty weight for training/validation drift")
    w_latency: float = Field(default=0.1, description="Penalty weight for execution latency")
    threshold: float = Field(default=0.6, description="Minimum validation score required for approval")

class ValidationResult(BaseModel):
    """
    Represents the structured outcome of the Validation Manager's audit.
    """
    approved: bool = Field(..., description="Whether the validation contract is fully satisfied")
    val_score: float = Field(..., description="The computed weighted validation score (W_val)")
    target_node: Optional[str] = Field(default=None, description="Graph node to loopback to in case of rejection")
    feedback: str = Field(..., description="Detailed diagnostic feedback for the agent workflow")

class ValidationManager:
    """
    Audits the current shared state against a ValidationContract, computes
    a weighted validation score, logs decisions, and decides loopback paths.
    """
    @staticmethod
    def evaluate(state: SharedState, contract: Optional[ValidationContract] = None) -> ValidationResult:
        if contract is None:
            contract = ValidationContract()

        # Dynamic contract adjustments based on dataset properties
        if state.dataset_path:
            import os
            if os.path.exists(state.dataset_path):
                try:
                    import pandas as pd
                    df = pd.read_csv(state.dataset_path)
                    num_samples = len(df)
                    missing_ratio = df.isnull().mean().mean()
                    
                    # Compute minority ratio for classification targets
                    minority_ratio = 0.5
                    if state.target_column and state.target_column in df.columns:
                        class_counts = df[state.target_column].value_counts()
                        if len(class_counts) == 2:
                            minority_ratio = class_counts.min() / len(df)
                            
                    # Apply dynamic threshold scaling rules
                    if num_samples < 100:
                        contract.w_latency = 0.02
                        contract.w_drift = 1.0  # scale up drift penalty for small datasets
                    
                    if minority_ratio < 0.3:
                        contract.w_metric = 1.5
                        contract.threshold = 0.70  # stricter validation for imbalanced targets
                        
                except Exception:
                    pass

        # Apply Multiple-Testing Correction (Holm-Bonferroni adaptation) to prevent overfitting
        if state.validation_history:
            import numpy as np
            trial_count = len(state.validation_history) + 1
            adjustment = 0.04 * np.log(trial_count)
            contract.threshold = min(0.95, contract.threshold + adjustment)



        # 1. Extract performance metric (M_metric)
        # We check for 'mcc' or 'f1_score' or 'accuracy' in order of preference
        m_metric = 0.0
        for metric_name in ["mcc", "f1_score", "accuracy"]:
            if metric_name in state.metrics:
                m_metric = state.metrics[metric_name]
                break
        
        # 2. Extract leakage penalty (L_leakage)
        # Use score if available, otherwise fallback to boolean flag
        l_leakage = 0.0
        if "leakage_score" in state.metrics:
            l_leakage = state.metrics["leakage_score"]
        elif state.target_leakage_detected:
            l_leakage = 1.0

        # 3. Extract drift penalty (D_drift)
        d_drift = 0.0
        if "drift_score" in state.metrics:
            d_drift = state.metrics["drift_score"]
        elif state.data_drift_detected:
            d_drift = 1.0

        # 4. Extract latency penalty (tau_latency)
        # If raw latency is present, normalize it (assume max penalty at 5000ms)
        tau_latency = 0.0
        if "latency_ms" in state.metrics:
            tau_latency = min(1.0, state.metrics["latency_ms"] / 5000.0)
        elif "latency_score" in state.metrics:
            tau_latency = state.metrics["latency_score"]

        # 5. Compute Weighted Validation Score (W_val)
        w_val = (
            (contract.w_metric * m_metric)
            - (contract.w_leakage * l_leakage)
            - (contract.w_drift * d_drift)
            - (contract.w_latency * tau_latency)
        )

        approved = True
        target_node = None
        feedback = "Validation passed. Model matches institutional criteria."

        # Rule A: Zero-tolerance for target leakage
        if state.target_leakage_detected or l_leakage > 0.05:
            approved = False
            target_node = "preprocessing"
            feedback = (
                f"REJECTED: Target leakage detected (Leakage: {l_leakage:.2f}). "
                "The model is training on future data or the target column itself. "
                "Action: Loop back to Stage 4 (Preprocessing & Features), rerun leakage scans, and remove contaminated columns."
            )
        # Rule B: Zero-tolerance for extreme drift
        elif state.data_drift_detected or d_drift > 0.25:
            approved = False
            target_node = "preprocessing"
            feedback = (
                f"REJECTED: Data drift score exceeds tolerance limit (Drift: {d_drift:.2f}). "
                "Feature distributions between training and validation sets have diverged. "
                "Action: Loop back to Stage 4 (Preprocessing & Features) to adjust scaling/robustness or check Stage 5 Validation split."
            )
        # Rule C: Overall score threshold check
        elif w_val < contract.threshold:
            approved = False
            # Check if failure is primarily due to low performance vs latency
            if m_metric < 0.70:
                target_node = "tuning_evaluation"
                feedback = (
                    f"REJECTED: Low weighted score (W_val: {w_val:.2f}, Metric: {m_metric:.2f}). "
                    "Performance metric is below the acceptable threshold. "
                    "Action: Loop back to Stage 7 (Tuning & Evaluation) or Stage 6 (Model Building) to modify hyperparameters or try ensembling."
                )
            else:
                target_node = "model_building"
                feedback = (
                    f"REJECTED: Low weighted score (W_val: {w_val:.2f}) due to resource constraints (Latency: {tau_latency:.2f}). "
                    "Model is too slow for production. "
                    "Action: Loop back to Stage 6 (Model Building) to select a lighter estimator or simplify features."
                )

        # 6. Log decision to the Shared State history
        log_entry = {
            "val_score": w_val,
            "approved": approved,
            "target_node": target_node,
            "feedback": feedback,
            "inputs": {
                "m_metric": m_metric,
                "l_leakage": l_leakage,
                "d_drift": d_drift,
                "tau_latency": tau_latency
            }
        }
        state.validation_history.append(log_entry)
        state.execution_history.append(f"validation_manager: {'APPROVED' if approved else 'REJECTED'}")

        return ValidationResult(
            approved=approved,
            val_score=w_val,
            target_node=target_node,
            feedback=feedback
        )
