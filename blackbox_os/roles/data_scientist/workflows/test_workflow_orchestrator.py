import pytest
from typing import Dict, Any, List
from blackbox_os.state.shared_state import SharedState
from blackbox_os.state.validation_manager import ValidationManager
from blackbox_os.roles.data_scientist.workflows.workflow_orchestrator import DataScientistOrchestrator, ASTGuardrail

def test_skill_isolation():
    """
    Verify that all 9 stages have a restricted skill library with size <= 15.
    """
    orchestrator = DataScientistOrchestrator()
    
    for stage_name in [f"Stage {i}" for i in range(1, 10)]:
        skills = orchestrator.get_stage_skills(stage_name)
        assert len(skills) <= 15, f"{stage_name} has {len(skills)} skills, exceeding 15!"
        assert len(skills) > 0, f"{stage_name} has 0 skills!"
        
        for skill in skills:
            assert skill["designation"] == "Data Scientist"

def test_ast_guardrails():
    """
    Verify that the AST guardrail blocks invalid syntax and unsafe exec/eval functions.
    """
    valid_code = "print('Hello World')\nimport pandas as pd"
    empty_code = ""
    syntax_error_code = "for i in range(10)"  # unclosed syntax
    unsafe_eval = "eval('1 + 1')"
    unsafe_exec = "exec('import sys')"
    
    assert ASTGuardrail.validate_code(valid_code) is True
    assert ASTGuardrail.validate_code(empty_code) is False
    assert ASTGuardrail.validate_code(syntax_error_code) is False
    assert ASTGuardrail.validate_code(unsafe_eval) is False
    assert ASTGuardrail.validate_code(unsafe_exec) is False

def test_standard_success():
    """
    Test a standard, clean execution where validation passes on the first try.
    """
    orchestrator = DataScientistOrchestrator()
    state = SharedState()
    
    stages_visited = []

    def mock_runner(stage: str, skills: List[Dict[str, Any]], shared_state: SharedState):
        stages_visited.append(stage)
        assert len(skills) <= 15
        
        if stage == "Stage 9":
            shared_state.metrics = {"mcc": 0.85, "latency_ms": 120.0}
            shared_state.target_leakage_detected = False
            shared_state.data_drift_detected = False

    result_state = orchestrator.run(state, max_loopbacks=3, agent_runner=mock_runner)
    
    assert result_state["validation_approved"] is True
    assert result_state["loopback_count"] == 0
    assert "deploy" in result_state["logs"][-1].lower()
    assert stages_visited == [f"Stage {i}" for i in range(1, 10)]

def test_loopback_leakage():
    """
    Test target leakage detection triggering a loopback to Data Ingestion & Integrity (Stage 2).
    """
    orchestrator = DataScientistOrchestrator()
    state = SharedState()
    
    run_count = 0

    def mock_runner(stage: str, skills: List[Dict[str, Any]], shared_state: SharedState):
        nonlocal run_count
        if stage == "Stage 9":
            run_count += 1
            if run_count == 1:
                shared_state.target_leakage_detected = True
                shared_state.metrics = {"f1_score": 0.99}
            else:
                shared_state.target_leakage_detected = False
                shared_state.metrics = {"f1_score": 0.85}

    result_state = orchestrator.run(state, max_loopbacks=3, agent_runner=mock_runner)
    
    assert result_state["validation_approved"] is True
    assert result_state["loopback_count"] == 1
    exec_history = result_state["shared_state"].execution_history
    assert exec_history.count("stage_2_completed") == 2
    assert "loopback retry #1 to preprocessing" in "".join(result_state["logs"]).lower()

def test_loopback_low_performance():
    """
    Test poor model performance triggering a loopback to Tuning & Evaluation (Stage 8).
    """
    orchestrator = DataScientistOrchestrator()
    state = SharedState()
    
    run_count = 0

    def mock_runner(stage: str, skills: List[Dict[str, Any]], shared_state: SharedState):
        nonlocal run_count
        if stage == "Stage 9":
            run_count += 1
            if run_count == 1:
                shared_state.metrics = {"f1_score": 0.52}
            else:
                shared_state.metrics = {"f1_score": 0.81}

    result_state = orchestrator.run(state, max_loopbacks=3, agent_runner=mock_runner)
    
    assert result_state["validation_approved"] is True
    assert result_state["loopback_count"] == 1
    exec_history = result_state["shared_state"].execution_history
    assert exec_history.count("stage_8_completed") == 2
    assert exec_history.count("stage_2_completed") == 1

def test_human_remediation_abort():
    """
    Verify that if the user submits 'abort', execution fails.
    """
    orchestrator = DataScientistOrchestrator()
    state = SharedState()
    
    def mock_runner(stage: str, skills: List[Dict[str, Any]], shared_state: SharedState):
        if stage == "Stage 9":
            shared_state.target_leakage_detected = True
            shared_state.metrics = {"f1_score": 0.90}
            
            # Setup abort instruction
            shared_state.human_override = "abort"

    result_state = orchestrator.run(state, max_loopbacks=2, agent_runner=mock_runner)
    
    assert result_state["validation_approved"] is False
    assert result_state["human_abort"] is True
    assert "user chose to abort the run" in "".join(result_state["logs"]).lower()

def test_human_remediation_override():
    """
    Verify that if the user submits instructions, the loopback count resets and retry is scheduled.
    """
    orchestrator = DataScientistOrchestrator()
    state = SharedState()
    
    stage_9_runs = 0

    def mock_runner(stage: str, skills: List[Dict[str, Any]], shared_state: SharedState):
        nonlocal stage_9_runs
        if stage == "Stage 9":
            stage_9_runs += 1
            if stage_9_runs <= 3:
                # Keep failing to trigger human remediation (max_loopbacks=2)
                shared_state.target_leakage_detected = True
                shared_state.metrics = {"f1_score": 0.90}
                if stage_9_runs == 3:
                    # Provide remediation override instruction on the 3rd run
                    shared_state.human_override = "drop leaked_return column and re-run"
            else:
                # Succeeded after override
                shared_state.target_leakage_detected = False
                shared_state.metrics = {"f1_score": 0.85}

    result_state = orchestrator.run(state, max_loopbacks=2, agent_runner=mock_runner)
    
    assert result_state["validation_approved"] is True
    assert result_state["human_abort"] is False
    # Verified it hit human remediation and successfully recovered
    assert "executing user override instructions" in "".join(result_state["logs"]).lower()
    assert stage_9_runs == 4


def test_validation_manager_with_real_dataframe(tmp_path):
    """
    Verify that ValidationManager dynamically loads a CSV, counts samples,
    and scales contract thresholds/weights correctly.
    """
    import pandas as pd
    import os
    from blackbox_os.state.validation_manager import ValidationManager, ValidationContract
    
    # 1. Create a mock CSV dataset
    dataset_file = os.path.join(tmp_path, "mock_data.csv")
    data = {
        "feature_a": [1.0] * 50,
        "feature_b": [2.0] * 50,
        "target": [0] * 45 + [1] * 5  # 10% minority ratio (imbalanced)
    }
    df = pd.DataFrame(data)
    df.to_csv(dataset_file, index=False)
    
    state = SharedState()
    state.dataset_path = dataset_file
    state.target_column = "target"
    state.metrics = {"f1_score": 0.40} # Fail threshold check (1.5 * 0.40 = 0.60 < 0.70)
    
    # Evaluate with default contract parameters
    contract = ValidationContract()
    
    # Run evaluation
    res = ValidationManager.evaluate(state, contract)
    
    # The minority ratio is 10% (< 30%), so threshold should be scaled to 0.70.
    assert contract.threshold == 0.70
    assert res.approved is False
    assert contract.w_metric == 1.5
    
    # If we run with a balanced target (e.g. 25 zeros, 25 ones)
    data_balanced = {
        "feature_a": [1.0] * 50,
        "feature_b": [2.0] * 50,
        "target": [0] * 25 + [1] * 25
    }
    df_bal = pd.DataFrame(data_balanced)
    df_bal.to_csv(dataset_file, index=False)
    
    # Set metrics to pass balanced threshold (1.0 * 0.68 = 0.68 >= 0.60)
    state.metrics = {"f1_score": 0.68}
    
    import numpy as np
    contract_bal = ValidationContract()
    res_bal = ValidationManager.evaluate(state, contract_bal)
    
    # Balanced should fall back to default threshold (0.60) + Holm-Bonferroni adjustment for trial 2
    expected_threshold = 0.60 + 0.04 * np.log(2)
    assert np.isclose(contract_bal.threshold, expected_threshold)
    assert res_bal.approved is True



