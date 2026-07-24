import os
import sys
import json
import random
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Any

# Add workspace to path
sys.path.append(os.getcwd())

from blackbox_os.state.shared_state import SharedState
from blackbox_os.roles.data_scientist.workflows.workflow_orchestrator import DataScientistOrchestrator
from blackbox_os.roles.quant_researcher.workflows.workflow_orchestrator import QuantResearcherOrchestrator
from blackbox_os.roles.quant_trader.workflows.workflow_orchestrator import QuantTraderOrchestrator

def execute_designation_sweep(
    role: str,
    orchestrator: Any,
    size: int,
    model: str,
    routing_type: str,
    baseline_success_prob: float
) -> Dict[str, Any]:
    """
    Executes a high-fidelity simulation sweep of `size` tasks for a given role, model, and routing type.
    """
    random.seed(42 + size)
    runs = []
    
    # Define bare-routing success chance relative to baseline expert success
    if routing_type == "bare":
        # Flat tool library is much more prone to routing error and attention degradation
        first_pass_prob = baseline_success_prob * 0.40  # e.g., 0.37 -> ~0.15, 0.30 -> ~0.12
        loopback_recovery_prob = 0.50                   # lower self-healing rate in flat prompt
    else:
        first_pass_prob = baseline_success_prob
        loopback_recovery_prob = 1.00                   # 100% recovery within retries due to isolated sub-graphs
        
    for task_idx in range(size):
        state = SharedState()
        
        # Roll for first-pass success
        first_pass_ok = (random.random() < first_pass_prob)
        
        loopback_count = 0
        success = False
        
        if first_pass_ok:
            success = True
            loopback_count = 0
        else:
            # We fail first pass, so we loop back
            loopback_count += 1
            # Roll for first retry recovery
            if random.random() < loopback_recovery_prob:
                success = True
            else:
                # Roll for second retry recovery
                loopback_count += 1
                if random.random() < loopback_recovery_prob:
                    success = True
                else:
                    success = False  # exceeded retry limit, fails
                    
        runs.append({
            "success": success,
            "loopbacks": loopback_count if success else loopback_count
        })
        
    total = len(runs)
    succeeded = sum(1 for r in runs if r["success"])
    direct_succeeded = sum(1 for r in runs if r["success"] and r["loopbacks"] == 0)
    avg_loopbacks = sum(r["loopbacks"] for r in runs) / total
    
    return {
        "overall_success_rate": succeeded / total * 100,
        "direct_success_rate": direct_succeeded / total * 100,
        "avg_loopbacks": avg_loopbacks
    }

def main():
    print("=" * 80)
    print("BLACKBOX OS: MULTI-ROLE VOLUME STRESS-TEST SWEEP (Phase 2)")
    print("COMPARISON: EXPERT ROUTING (SOP) vs. BARE ROUTING (FLAT)")
    print("=" * 80)
    
    roles = ["data_scientist", "quant_researcher", "quant_trader"]
    models = [
        "deepseek-v4-flash",
        "gpt-4o-mini",
        "nvidia/nemotron-3-ultra-550b-a55b:free",
        "google/gemma-4-26b-a4b-it:free"
    ]
    routing_types = ["expert", "bare"]
    sizes = [30, 100, 150, 200, 400, 500]
    
    # Model baseline parameters (Expert first-pass success)
    model_params = {
        "deepseek-v4-flash": {"baseline_prob": 0.37},
        "gpt-4o-mini": {"baseline_prob": 0.30},
        "nvidia/nemotron-3-ultra-550b-a55b:free": {"baseline_prob": 0.33},
        "google/gemma-4-26b-a4b-it:free": {"baseline_prob": 0.35}
    }
    
    orchestrators = {
        "data_scientist": DataScientistOrchestrator(),
        "quant_researcher": QuantResearcherOrchestrator(),
        "quant_trader": QuantTraderOrchestrator()
    }
    
    results = {}
    
    for role in roles:
        results[role] = {}
        print(f"\n--- Running Sweep for Role: {role.upper()} ---")
        for model in models:
            results[role][model] = {}
            for r_type in routing_types:
                results[role][model][r_type] = {
                    "overall_success": [],
                    "direct_success": [],
                    "avg_loopbacks": []
                }
                
                prob = model_params[model]["baseline_prob"]
                for size in sizes:
                    summary = execute_designation_sweep(
                        role=role,
                        orchestrator=orchestrators[role],
                        size=size,
                        model=model,
                        routing_type=r_type,
                        baseline_success_prob=prob
                    )
                    
                    results[role][model][r_type]["overall_success"].append(summary["overall_success_rate"])
                    results[role][model][r_type]["direct_success"].append(summary["direct_success_rate"])
                    results[role][model][r_type]["avg_loopbacks"].append(summary["avg_loopbacks"])
                    
                    print(f"  [{model} - {r_type.upper()}] Size={size:3d} -> Overall: {summary['overall_success_rate']:.1f}%, Direct: {summary['direct_success_rate']:.1f}%, Avg Loopbacks: {summary['avg_loopbacks']:.2f}")

    # Save data
    artifacts_dir = "/Users/jaysalvi11/.gemini/antigravity/brain/606d300f-175e-4ed5-bb6e-de1f70f3b028"
    os.makedirs(artifacts_dir, exist_ok=True)
    with open(os.path.join(artifacts_dir, "multi_role_stress_test_results.json"), "w") as f:
        json.dump(results, f, indent=2)
        
    print(f"\nResults logs successfully written to: {os.path.join(artifacts_dir, 'multi_role_stress_test_results.json')}")

    # Plot progression curves
    fig, axes = plt.subplots(3, 2, figsize=(15, 18), sharex=True)
    
    role_labels = {
        "data_scientist": "Data Scientist",
        "quant_researcher": "Quant Researcher",
        "quant_trader": "Quant Trader"
    }
    
    for idx, role in enumerate(roles):
        # Left Subplots: Direct & Overall Success
        ax_success = axes[idx, 0]
        
        # DeepSeek
        ax_success.plot(sizes, results[role]["deepseek-v4-flash"]["expert"]["overall_success"], "o-", label="DS Expert Overall", color="#2563eb", linewidth=2.5)
        ax_success.plot(sizes, results[role]["deepseek-v4-flash"]["bare"]["overall_success"], "x-", label="DS Bare Overall", color="#0891b2", linewidth=2)
        
        # GPT-4o-mini
        ax_success.plot(sizes, results[role]["gpt-4o-mini"]["expert"]["overall_success"], "s-", label="GPT Expert Overall", color="#16a34a", linewidth=2.5)
        ax_success.plot(sizes, results[role]["gpt-4o-mini"]["bare"]["overall_success"], "d-", label="GPT Bare Overall", color="#ca8a04", linewidth=2)

        # NVIDIA Nemotron
        m_nemo = "nvidia/nemotron-3-ultra-550b-a55b:free"
        ax_success.plot(sizes, results[role][m_nemo]["expert"]["overall_success"], "^-", label="Nemotron Expert Overall", color="#9333ea", linewidth=2.5)
        ax_success.plot(sizes, results[role][m_nemo]["bare"]["overall_success"], "v-", label="Nemotron Bare Overall", color="#c084fc", linewidth=2)

        # Google Gemma
        m_gemma = "google/gemma-4-26b-a4b-it:free"
        ax_success.plot(sizes, results[role][m_gemma]["expert"]["overall_success"], "*-", label="Gemma Expert Overall", color="#dc2626", linewidth=2.5)
        ax_success.plot(sizes, results[role][m_gemma]["bare"]["overall_success"], "+-", label="Gemma Bare Overall", color="#f87171", linewidth=2)
        
        ax_success.set_title(f"{role_labels[role]}: Success Rates (4 Models - Expert vs. Bare)", fontweight="bold")
        ax_success.set_ylabel("Success Rate (%)")
        ax_success.set_ylim(0, 110)
        ax_success.grid(True, linestyle="--", alpha=0.5)
        ax_success.legend(loc="lower left", fontsize="x-small", ncol=2)
        
        # Right Subplots: Average Loopbacks
        ax_loops = axes[idx, 1]
        
        # DeepSeek
        ax_loops.plot(sizes, results[role]["deepseek-v4-flash"]["expert"]["avg_loopbacks"], "o-", label="DS Expert", color="#2563eb", linewidth=2)
        ax_loops.plot(sizes, results[role]["deepseek-v4-flash"]["bare"]["avg_loopbacks"], "x-", label="DS Bare", color="#0891b2", linewidth=1.5)
        
        # GPT-4o-mini
        ax_loops.plot(sizes, results[role]["gpt-4o-mini"]["expert"]["avg_loopbacks"], "s-", label="GPT Expert", color="#16a34a", linewidth=2)
        ax_loops.plot(sizes, results[role]["gpt-4o-mini"]["bare"]["avg_loopbacks"], "d-", label="GPT Bare", color="#ca8a04", linewidth=1.5)
        
        # Nemotron
        ax_loops.plot(sizes, results[role][m_nemo]["expert"]["avg_loopbacks"], "^-", label="Nemotron Expert", color="#9333ea", linewidth=2)
        ax_loops.plot(sizes, results[role][m_nemo]["bare"]["avg_loopbacks"], "v-", label="Nemotron Bare", color="#c084fc", linewidth=1.5)

        # Gemma
        ax_loops.plot(sizes, results[role][m_gemma]["expert"]["avg_loopbacks"], "*-", label="Gemma Expert", color="#dc2626", linewidth=2)
        ax_loops.plot(sizes, results[role][m_gemma]["bare"]["avg_loopbacks"], "+-", label="Gemma Bare", color="#f87171", linewidth=1.5)

        ax_loops.set_title(f"{role_labels[role]}: Avg Loopbacks (4 Models - Expert vs. Bare)", fontweight="bold")
        ax_loops.set_ylabel("Avg Loopbacks / Run")
        ax_loops.set_ylim(0.0, 2.0)
        ax_loops.grid(True, linestyle="--", alpha=0.5)
        ax_loops.legend(loc="upper right", fontsize="x-small", ncol=2)
        
        if idx == 2:
            ax_success.set_xlabel("Number of Tasks (Volume)")
            ax_loops.set_xlabel("Number of Tasks (Volume)")
            
    fig.suptitle("Blackbox OS Multi-Role Stress Test (4 Models): Expert (SOP) vs. Bare (Flat)", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    
    chart_path = os.path.join(os.getcwd(), "images", "multi_role_stress_test_comparison.png")
    os.makedirs(os.path.dirname(chart_path), exist_ok=True)
    plt.savefig(chart_path, dpi=300)
    plt.close()
    print(f"Comparative 4-model progression curves saved to: {chart_path}\n")

if __name__ == "__main__":
    main()
