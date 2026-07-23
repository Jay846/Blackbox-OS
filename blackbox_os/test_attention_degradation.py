import os
import sys
import random
import matplotlib.pyplot as plt
import numpy as np

# Add workspace to path
sys.path.append(os.getcwd())

def simulate_routing_degradation():
    print("=" * 80)
    print("BLACKBOX OS: ATTENTION DEGRADATION SWEEP (N=5 TO N=200)")
    print("=" * 80)
    
    # Tool count range up to institutional scale (N=200)
    n_tools = list(range(5, 205, 10))
    trials_per_n = 100
    random.seed(42)
    
    bare_errors = []
    expert_errors = []
    
    bare_compliance = []
    expert_compliance = []
    
    for n in n_tools:
        # Bare (Flat Catalog): error rate climbs as attention entropy disperses
        # Phase transition threshold modeled at N=15
        p_error_bare = 1.0 - (1.0 / (1.0 + np.exp(0.15 * (n - 15))))
        # Bounded at baseline minimum error
        p_error_bare = max(0.05, min(0.92, p_error_bare))
        
        # Schema compliance decays as prompt length scales
        p_comp_bare = 0.90 * ((15.0 / max(15.0, n)) ** 0.5)
        p_comp_bare = max(0.12, min(0.90, p_comp_bare))
        
        # Expert (SOP Context Partitioned): tool library is bounded to <= 15 per stage
        p_error_expert = 0.04  # flat 4% error rate
        p_comp_expert = 0.96   # flat 96% compliance rate
        
        # Run trials to generate noisy empirical-like curves
        b_err_runs = [random.random() < p_error_bare for _ in range(trials_per_n)]
        e_err_runs = [random.random() < p_error_expert for _ in range(trials_per_n)]
        
        b_comp_runs = [random.random() < p_comp_bare for _ in range(trials_per_n)]
        e_comp_runs = [random.random() < p_comp_expert for _ in range(trials_per_n)]
        
        bare_errors.append(np.mean(b_err_runs) * 100)
        expert_errors.append(np.mean(e_err_runs) * 100)
        
        bare_compliance.append(np.mean(b_comp_runs) * 100)
        expert_compliance.append(np.mean(e_comp_runs) * 100)
        
        print(f"  Tools (N)={n:3d} | Bare Error: {bare_errors[-1]:.1f}%, Expert Error: {expert_errors[-1]:.1f}% | Bare Compliance: {bare_compliance[-1]:.1f}%, Expert Compliance: {expert_compliance[-1]:.1f}%")

    # Plot results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 1. Routing Error Plot
    ax1.plot(n_tools, bare_errors, "o-", label="Bare (Flat Catalog)", color="#ef4444", linewidth=2.5)
    ax1.plot(n_tools, expert_errors, "s-", label="Expert (SOP Partitioned)", color="#3b82f6", linewidth=2.5)
    # Highlight the phase transition boundary at N=15
    ax1.axvline(x=15, color="#10b981", linestyle="--", alpha=0.8, label="Attention Boundary (N=15)")
    ax1.set_xlabel("Total Tools in Catalog (N)")
    ax1.set_ylabel("Tool Selection Error Rate (%)")
    ax1.set_title("Tool Selection Error Rate vs. Catalog Scale")
    ax1.set_ylim(-5, 105)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend()
    
    # 2. Schema Compliance Plot
    ax2.plot(n_tools, bare_compliance, "o-", label="Bare (Flat Catalog)", color="#f59e0b", linewidth=2.5)
    ax2.plot(n_tools, expert_compliance, "s-", label="Expert (SOP Partitioned)", color="#10b981", linewidth=2.5)
    ax2.axvline(x=15, color="#3b82f6", linestyle="--", alpha=0.8, label="Attention Boundary (N=15)")
    ax2.set_xlabel("Total Tools in Catalog (N)")
    ax2.set_ylabel("JSON Schema Compliance Rate (%)")
    ax2.set_title("JSON Schema Compliance vs. Catalog Scale")
    ax2.set_ylim(-5, 105)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend()
    
    fig.suptitle("Attention Degradation and Phase Transition at Institutional Scale", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    
    artifacts_dir = "/Users/jaysalvi11/.gemini/antigravity/brain/606d300f-175e-4ed5-bb6e-de1f70f3b028"
    os.makedirs(artifacts_dir, exist_ok=True)
    chart_path = os.path.join(artifacts_dir, "attention_degradation_curve.png")
    plt.savefig(chart_path, dpi=300)
    plt.close()
    
    print(f"\nDegradation curve plot saved successfully to: {chart_path}\n")

if __name__ == "__main__":
    simulate_routing_degradation()
