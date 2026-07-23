import pytest
from typing import Dict, Any, List
from blackbox_os.state.shared_state import SharedState
from blackbox_os.roles.quant_researcher.workflows.workflow_orchestrator import QuantResearcherOrchestrator

def test_qr_skill_isolation():
    orchestrator = QuantResearcherOrchestrator()
    for stage_name in [f"Stage {i}" for i in range(1, 10)]:
        skills = orchestrator.get_stage_skills(stage_name)
        assert len(skills) <= 15, f"{stage_name} has {len(skills)} skills, exceeding 15!"
        assert len(skills) > 0, f"{stage_name} has 0 skills!"
        for skill in skills:
            assert skill["designation"] == "Quant Researcher"

def test_qr_standard_success():
    orchestrator = QuantResearcherOrchestrator()
    state = SharedState()
    stages_visited = []

    def mock_runner(stage: str, skills: List[Dict[str, Any]], shared_state: SharedState):
        stages_visited.append(stage)
        if stage == "Stage 9":
            shared_state.metrics = {"mcc": 0.85}

    result = orchestrator.run(state, max_loopbacks=3, agent_runner=mock_runner)
    assert result["validation_approved"] is True
    assert result["loopback_count"] == 0
    assert stages_visited == [f"Stage {i}" for i in range(1, 10)]

def test_qr_loopback_and_hitl():
    orchestrator = QuantResearcherOrchestrator()
    state = SharedState()
    runs = 0

    def mock_runner(stage: str, skills: List[Dict[str, Any]], shared_state: SharedState):
        nonlocal runs
        if stage == "Stage 9":
            runs += 1
            if runs == 1:
                shared_state.metrics = {"f1_score": 0.50}
            elif runs == 2:
                shared_state.metrics = {"f1_score": 0.85}
                shared_state.target_leakage_detected = True
                shared_state.human_override = "force pass override"
            else:
                shared_state.metrics = {"f1_score": 0.85}
                shared_state.target_leakage_detected = False

    result = orchestrator.run(state, max_loopbacks=1, agent_runner=mock_runner)
    assert result["validation_approved"] is True
    assert "remediation" in "".join(result["logs"]).lower()
