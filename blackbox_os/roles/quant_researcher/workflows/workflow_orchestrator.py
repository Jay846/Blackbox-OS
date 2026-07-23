import os
import json
from typing import Dict, Any, List, Optional, Callable, TypedDict
from langgraph.graph import StateGraph, END

from blackbox_os.state.shared_state import SharedState
from blackbox_os.state.validation_manager import ValidationManager
from blackbox_os.roles.data_scientist.workflows.workflow_orchestrator import ASTGuardrail, OrchestratorState

class QuantResearcherOrchestrator:
    """
    LangGraph-based Orchestrator for Quant Researcher workflow partitioning the 99 skills
    into 9 stages (N = 11 skills per stage) and integrating validation loops.
    """
    def __init__(self, skills_file_path: Optional[str] = None):
        if skills_file_path is None:
            skills_file_path = os.path.abspath(
                os.path.join(
                    os.path.dirname(__file__),
                    "../../../roles_based_skills/quant_researcher_skills.txt"
                )
            )
        self.skills_file_path = skills_file_path
        self.all_skills = self._load_skills()
        self.agent_runner: Optional[Callable[[str, List[Dict[str, Any]], SharedState], None]] = None

        # Partition 99 skills into 9 stages programmatically for strict context isolation
        self.skills_by_stage = {}
        sorted_skills = sorted(self.all_skills, key=lambda x: x["id"])
        
        for i in range(9):
            stage_name = f"Stage {i+1}"
            chunk = sorted_skills[i*11 : (i+1)*11]
            self.skills_by_stage[stage_name] = chunk

        self.graph = self._build_graph()

    def _load_skills(self) -> List[Dict[str, Any]]:
        if os.path.exists(self.skills_file_path):
            try:
                with open(self.skills_file_path, "r") as f:
                    return json.load(f)
            except Exception:
                pass
        return []

    def get_stage_skills(self, stage_name: str) -> List[Dict[str, Any]]:
        return self.skills_by_stage.get(stage_name, [])

    # Graph Nodes
    def stage_1_node(self, state: OrchestratorState) -> OrchestratorState:
        state["logs"].append("[Quant Researcher Stage 1] Idea Generation...")
        state["shared_state"].active_role = "quant_researcher"
        state["shared_state"].active_workflow = "Stage 1"
        if self.agent_runner:
            self.agent_runner("Stage 1", self.get_stage_skills("Stage 1"), state["shared_state"])
        state["shared_state"].execution_history.append("stage_1_completed")
        return state

    def stage_2_node(self, state: OrchestratorState) -> OrchestratorState:
        state["logs"].append("[Quant Researcher Stage 2] Data Ingestion...")
        state["shared_state"].active_workflow = "Stage 2"
        if self.agent_runner:
            self.agent_runner("Stage 2", self.get_stage_skills("Stage 2"), state["shared_state"])
        state["shared_state"].execution_history.append("stage_2_completed")
        return state

    def stage_3_node(self, state: OrchestratorState) -> OrchestratorState:
        state["logs"].append("[Quant Researcher Stage 3] Data Quality & Cleaning...")
        state["shared_state"].active_workflow = "Stage 3"
        if self.agent_runner:
            self.agent_runner("Stage 3", self.get_stage_skills("Stage 3"), state["shared_state"])
        state["shared_state"].execution_history.append("stage_3_completed")
        return state

    def stage_4_node(self, state: OrchestratorState) -> OrchestratorState:
        state["logs"].append("[Quant Researcher Stage 4] Feature Engineering...")
        state["shared_state"].active_workflow = "Stage 4"
        if self.agent_runner:
            self.agent_runner("Stage 4", self.get_stage_skills("Stage 4"), state["shared_state"])
        state["shared_state"].execution_history.append("stage_4_completed")
        return state

    def stage_5_node(self, state: OrchestratorState) -> OrchestratorState:
        state["logs"].append("[Quant Researcher Stage 5] Feature Screening...")
        state["shared_state"].active_workflow = "Stage 5"
        if self.agent_runner:
            self.agent_runner("Stage 5", self.get_stage_skills("Stage 5"), state["shared_state"])
        state["shared_state"].execution_history.append("stage_5_completed")
        return state

    def stage_6_node(self, state: OrchestratorState) -> OrchestratorState:
        state["logs"].append("[Quant Researcher Stage 6] Backtest Design...")
        state["shared_state"].active_workflow = "Stage 6"
        if self.agent_runner:
            self.agent_runner("Stage 6", self.get_stage_skills("Stage 6"), state["shared_state"])
        state["shared_state"].execution_history.append("stage_6_completed")
        return state

    def stage_7_node(self, state: OrchestratorState) -> OrchestratorState:
        state["logs"].append("[Quant Researcher Stage 7] Simulation & Stress Testing...")
        state["shared_state"].active_workflow = "Stage 7"
        if self.agent_runner:
            self.agent_runner("Stage 7", self.get_stage_skills("Stage 7"), state["shared_state"])
        state["shared_state"].execution_history.append("stage_7_completed")
        return state

    def stage_8_node(self, state: OrchestratorState) -> OrchestratorState:
        state["logs"].append("[Quant Researcher Stage 8] Performance Attribution...")
        state["shared_state"].active_workflow = "Stage 8"
        if self.agent_runner:
            self.agent_runner("Stage 8", self.get_stage_skills("Stage 8"), state["shared_state"])
        state["shared_state"].execution_history.append("stage_8_completed")
        return state

    def stage_9_node(self, state: OrchestratorState) -> OrchestratorState:
        state["logs"].append("[Quant Researcher Stage 9] Handoff & Documentation...")
        state["shared_state"].active_workflow = "Stage 9"
        if self.agent_runner:
            self.agent_runner("Stage 9", self.get_stage_skills("Stage 9"), state["shared_state"])
            
        # Run Compliance Audit
        result = ValidationManager.evaluate(state["shared_state"])
        state["validation_approved"] = result.approved
        state["validation_target_node"] = result.target_node
        state["logs"].append(f"[Quant Researcher Stage 9] Audit result: Approved={result.approved}, TargetNode={result.target_node}")
        
        if not result.approved:
            state["loopback_count"] += 1
            if state["loopback_count"] <= state["max_loopbacks"]:
                state["logs"].append(f"[Quant Researcher Stage 9] Loopback retry #{state['loopback_count']} to {result.target_node}")
            else:
                state["logs"].append(f"[Quant Researcher Stage 9] Max loopback retries ({state['max_loopbacks']}) reached.")
                
        state["shared_state"].execution_history.append("stage_9_completed")
        return state

    def human_remediation_node(self, state: OrchestratorState) -> OrchestratorState:
        state["logs"].append("[Quant Researcher Human Remediation] Reviewing validation failures...")
        override = state["shared_state"].human_override
        
        if override and override.lower() == "abort":
            state["logs"].append("[Quant Researcher Human Remediation] User chose to abort the run.")
            state["human_abort"] = True
        elif override:
            state["logs"].append(f"[Quant Researcher Human Remediation] Executing override: {override}")
            state["loopback_count"] = 0
            state["human_abort"] = False
        else:
            state["logs"].append("[Quant Researcher Human Remediation] No override found. Aborting.")
            state["human_abort"] = True
            
        return state

    # Conditional Routing
    def route_after_validation(self, state: OrchestratorState) -> str:
        if state["validation_approved"]:
            state["logs"].append("[Quant Researcher Routing] Validation successful. Proceeding to deploy.")
            return "deploy"

        if state["loopback_count"] > state["max_loopbacks"]:
            return "human_remediation"

        target = state["validation_target_node"]
        if target == "preprocessing":
            return "stage_2"
        elif target == "model_building":
            return "stage_6"
        elif target == "tuning_evaluation":
            return "stage_7"
        else:
            return "stage_2"

    def route_after_remediation(self, state: OrchestratorState) -> str:
        if state["human_abort"]:
            return "fail"
            
        target = state["validation_target_node"]
        if target == "preprocessing":
            return "stage_2"
        elif target == "model_building":
            return "stage_6"
        elif target == "tuning_evaluation":
            return "stage_7"
        else:
            return "stage_2"

    def _build_graph(self) -> StateGraph:
        workflow = StateGraph(OrchestratorState)
        
        workflow.add_node("stage_1", self.stage_1_node)
        workflow.add_node("stage_2", self.stage_2_node)
        workflow.add_node("stage_3", self.stage_3_node)
        workflow.add_node("stage_4", self.stage_4_node)
        workflow.add_node("stage_5", self.stage_5_node)
        workflow.add_node("stage_6", self.stage_6_node)
        workflow.add_node("stage_7", self.stage_7_node)
        workflow.add_node("stage_8", self.stage_8_node)
        workflow.add_node("stage_9", self.stage_9_node)
        workflow.add_node("human_remediation", self.human_remediation_node)
        
        workflow.set_entry_point("stage_1")
        workflow.add_edge("stage_1", "stage_2")
        workflow.add_edge("stage_2", "stage_3")
        workflow.add_edge("stage_3", "stage_4")
        workflow.add_edge("stage_4", "stage_5")
        workflow.add_edge("stage_5", "stage_6")
        workflow.add_edge("stage_6", "stage_7")
        workflow.add_edge("stage_7", "stage_8")
        workflow.add_edge("stage_8", "stage_9")
        
        workflow.add_conditional_edges(
            "stage_9",
            self.route_after_validation,
            {
                "stage_2": "stage_2",
                "stage_6": "stage_6",
                "stage_7": "stage_7",
                "human_remediation": "human_remediation",
                "deploy": END
            }
        )
        
        workflow.add_conditional_edges(
            "human_remediation",
            self.route_after_remediation,
            {
                "stage_2": "stage_2",
                "stage_6": "stage_6",
                "stage_7": "stage_7",
                "fail": END
            }
        )
        
        return workflow.compile()

    def run(
        self, 
        initial_shared_state: SharedState, 
        max_loopbacks: int = 3, 
        agent_runner: Optional[Callable[[str, List[Dict[str, Any]], SharedState], None]] = None
    ) -> Dict[str, Any]:
        self.agent_runner = agent_runner
        
        initial_state: OrchestratorState = {
            "shared_state": initial_shared_state,
            "loopback_count": 0,
            "max_loopbacks": max_loopbacks,
            "logs": [],
            "validation_approved": False,
            "validation_target_node": None,
            "human_abort": False
        }
        
        return self.graph.invoke(initial_state)
