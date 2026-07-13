import pytest
from blackbox_os.state.shared_state import SharedState
from blackbox_os.state.validation_manager import ValidationManager, ValidationContract

def test_validation_success():
    # Setup state representing a high-performing model with no issues
    state = SharedState(
        active_role="data_scientist",
        active_workflow="SOP 3",
        metrics={"mcc": 0.85, "latency_ms": 150.0},
        target_leakage_detected=False,
        data_drift_detected=False
    )
    
    result = ValidationManager.evaluate(state)
    
    assert result.approved is True
    assert result.val_score > 0.60
    assert result.target_node is None
    assert "passed" in result.feedback
    
    # Check that logs were committed to SharedState
    assert len(state.validation_history) == 1
    assert state.validation_history[0]["approved"] is True
    assert "validation_manager: APPROVED" in state.execution_history

def test_validation_leakage_rejection():
    # Setup state with high metric but target leakage flagged
    state = SharedState(
        active_role="data_scientist",
        active_workflow="SOP 3",
        metrics={"f1_score": 0.99, "latency_ms": 50.0},
        target_leakage_detected=True,
        data_drift_detected=False
    )
    
    result = ValidationManager.evaluate(state)
    
    assert result.approved is False
    assert result.target_node == "preprocessing"
    assert "leakage" in result.feedback.lower()
    
    # Check history logging
    assert state.validation_history[0]["approved"] is False
    assert "validation_manager: REJECTED" in state.execution_history

def test_validation_drift_rejection():
    # Setup state with high metric but data drift detected
    state = SharedState(
        active_role="data_scientist",
        active_workflow="SOP 3",
        metrics={"f1_score": 0.88, "drift_score": 0.35},
        target_leakage_detected=False,
        data_drift_detected=True
    )
    
    result = ValidationManager.evaluate(state)
    
    assert result.approved is False
    assert result.target_node == "preprocessing"
    assert "drift" in result.feedback.lower()

def test_validation_low_performance_rejection():
    # Setup state with poor model performance (e.g. low F1 score)
    state = SharedState(
        active_role="data_scientist",
        active_workflow="SOP 3",
        metrics={"f1_score": 0.52},
        target_leakage_detected=False,
        data_drift_detected=False
    )
    
    result = ValidationManager.evaluate(state)
    
    assert result.approved is False
    assert result.target_node == "tuning_evaluation"
    assert "low weighted score" in result.feedback.lower()

def test_validation_high_latency_rejection():
    # Setup state with good performance but extreme latency
    state = SharedState(
        active_role="data_scientist",
        active_workflow="SOP 3",
        metrics={"f1_score": 0.91, "latency_ms": 4800.0},
        target_leakage_detected=False,
        data_drift_detected=False
    )
    
    # Using default threshold 0.60
    # W_val = 1.0 * 0.91 - 0.1 * (4800 / 5000) = 0.91 - 0.096 = 0.814 (still passes)
    # Let's adjust contract to make latency weight much heavier to force rejection
    custom_contract = ValidationContract(w_latency=0.5, threshold=0.80)
    # W_val = 1.0 * 0.91 - 0.5 * (4800 / 5000) = 0.91 - 0.48 = 0.43 (should fail)
    
    result = ValidationManager.evaluate(state, contract=custom_contract)
    
    assert result.approved is False
    assert result.target_node == "model_building"
    assert "latency" in result.feedback.lower()
