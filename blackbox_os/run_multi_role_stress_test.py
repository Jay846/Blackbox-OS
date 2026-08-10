import os
import sys
import json
import random
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Any

# ── Dynamic Workspace Root Detection ─────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
root_candidate = SCRIPT_DIR
while root_candidate != os.path.dirname(root_candidate):
    if os.path.exists(os.path.join(root_candidate, "blackbox_os")):
        WORKSPACE_ROOT = root_candidate
        break
    root_candidate = os.path.dirname(root_candidate)
else:
    WORKSPACE_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "..", ".."))

if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

# ── Empirical Baseline Grounding (Calibrated from Table 6 Live API Sweep) ─────
# Maps empirical (Direct Success %, Loopback Recovery %) per model & mode
EMPIRICAL_BASELINES = {
    "deepseek-v4-flash": {
        "expert": {"direct_p": 1.00, "rec_p": 1.00},
        "bare":   {"direct_p": 0.533, "rec_p": 0.214}
    },
    "gpt-4o-mini": {
        "expert": {"direct_p": 1.00, "rec_p": 1.00},
        "bare":   {"direct_p": 0.467, "rec_p": 0.000}
    },
    "nvidia/nemotron-3-ultra-550b-a55b:free": {
        "expert": {"direct_p": 0.600, "rec_p": 0.833},
        "bare":   {"direct_p": 0.167, "rec_p": 0.360}
    },
    "openai/gpt-oss-20b:free": {
        "expert": {"direct_p": 0.867, "rec_p": 0.500},
        "bare":   {"direct_p": 0.667, "rec_p": 0.300}
    }
}

# Skill catalog sizes per role
ROLE_CATALOG_SIZES = {
    "data_scientist": 77,
    "quant_researcher": 99,
    "quant_trader": 100
}

def execute_designation_sweep(
    role: str,
    size: int,
    model: str,
    routing_type: str,
    max_loopbacks: int = 2
) -> Dict[str, Any]:
    """
    Simulates a volume workflow sweep grounded in Table 6 empirical baselines,
    accounting for scale-induced catalog pressure in Bare mode.
    """
    random.seed(42 + size + len(role))
    runs = []
    
    baseline = EMPIRICAL_BASELINES[model][routing_type]
    direct_p = baseline["direct_p"]
    rec_p = baseline["rec_p"]

    # Catalog scale penalty for Bare mode as total role catalog expands (77 -> 99 -> 100)
    if routing_type == "bare":
        scale_ratio = ROLE_CATALOG_SIZES[role] / 77.0
        direct_p = max(0.05, direct_p / (scale_ratio ** 0.3))
        rec_p = max(0.00, rec_p / (scale_ratio ** 0.3))

    for _ in range(size):
        # Pass 1: Direct Attempt
        if random.random() < direct_p:
            runs.append({"success": True, "loopbacks": 0})
            continue

        # Fail Pass 1 -> Loopbacks
        recovered = False
        loopbacks_used = 0
        for attempt in range(1, max_loopbacks + 1):
            loopbacks_used = attempt
            if random.random() < rec_p:
                recovered = True
                break

        runs.append({
            "success": recovered,
            "loopbacks": loopbacks_used
        })

    total = len(runs)
    succeeded = sum(1 for r in runs if r["success"])
    direct_succeeded = sum(1 for r in runs if r["success"] and r["loopbacks"] == 0)
    avg_loopbacks = sum(r["loopbacks"] for r in runs) / total

    return {
        "overall_success_rate": (succeeded / total) * 100,
        "direct_success_rate": (direct_succeeded / total) * 100,
        "avg_loopbacks": avg_loopbacks
    }

def main():
    print("=" * 80)
    print("BLACKBOX OS: MULTI-ROLE VOLUME STRESS-TEST SWEEP (Simulation Phase)")
    print("EMPIRICALLY CALIBRATED: EXPERT ROUTING (SOP) vs. BARE ROUTING (FLAT)")
    print("=" * 80)

    roles = ["data_scientist", "quant_researcher", "quant_trader"]
    models = list(EMPIRICAL_BASELINES.keys())
    routing_types = ["expert", "bare"]
    sizes = [30, 100, 150, 200, 400, 500]

    results = {}

    for role in roles:
        results[role] = {}
        print(f"\n--- Running Volume Sweep for Role: {role.upper()} ({ROLE_CATALOG_SIZES[role]} Skills) ---")
        for model in models:
            results[role][model] = {}
            for r_type in routing_types:
                results[role][model][r_type] = {
                    "overall_success": [],
                    "direct_success": [],
                    "avg_loopbacks": []
                }
                for size in sizes:
                    summary = execute_designation_sweep(
                        role=role,
                        size=size,
                        model=model,
                        routing_type=r_type
                    )

                    results[role][model][r_type]["overall_success"].append(summary["overall_success_rate"])
                    results[role][model][r_type]["direct_success"].append(summary["direct_success_rate"])
                    results[role][model][r_type]["avg_loopbacks"].append(summary["avg_loopbacks"])

                    print(f"  [{model} - {r_type.upper()}] Tasks={size:3d} -> E2E: {summary['overall_success_rate']:.1f}%, Direct: {summary['direct_success_rate']:.1f}%, Avg Loopbacks: {summary['avg_loopbacks']:.2f}")

    # Output JSON relative to WORKSPACE_ROOT
    out_dir = os.path.join(WORKSPACE_ROOT, "blackbox_os", "roles", "data_scientist", "workflows")
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "multi_role_stress_test_results.json")

    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults logs successfully written to: {out_file}")

    # Plot progression curves
    fig, axes = plt.subplots(3, 2, figsize=(15, 18), sharex=True)

    role_labels = {
        "data_scientist": "Data Scientist (77 Skills)",
        "quant_researcher": "Quant Researcher (99 Skills)",
        "quant_trader": "Quant Trader (100 Skills)"
    }

    colors = {
        "deepseek-v4-flash": ("#2563eb", "#0891b2"),
        "gpt-4o-mini": ("#16a34a", "#ca8a04"),
        "nvidia/nemotron-3-ultra-550b-a55b:free": ("#9333ea", "#c084fc"),
        "openai/gpt-oss-20b:free": ("#dc2626", "#f87171")
    }

    for idx, role in enumerate(roles):
        ax_success = axes[idx, 0]
        ax_loops = axes[idx, 1]

        for model in models:
            c_exp, c_bare = colors[model]
            m_short = model.split("/")[-1].replace(":free", "")

            # Left: Success Rates
            ax_success.plot(sizes, results[role][model]["expert"]["overall_success"], "o-", label=f"{m_short} Expert", color=c_exp, linewidth=2.5)
            ax_success.plot(sizes, results[role][model]["bare"]["overall_success"], "x--", label=f"{m_short} Bare", color=c_bare, linewidth=2)

            # Right: Average Loopbacks
            ax_loops.plot(sizes, results[role][model]["expert"]["avg_loopbacks"], "o-", label=f"{m_short} Expert", color=c_exp, linewidth=2)
            ax_loops.plot(sizes, results[role][model]["bare"]["avg_loopbacks"], "x--", label=f"{m_short} Bare", color=c_bare, linewidth=1.5)

        ax_success.set_title(f"{role_labels[role]}: E2E Success Rate (Expert vs. Bare)", fontweight="bold")
        ax_success.set_ylabel("Success Rate (%)")
        ax_success.set_ylim(0, 110)
        ax_success.grid(True, linestyle="--", alpha=0.5)
        ax_success.legend(loc="lower left", fontsize="x-small", ncol=2)

        ax_loops.set_title(f"{role_labels[role]}: Avg Loopbacks / Run (Expert vs. Bare)", fontweight="bold")
        ax_loops.set_ylabel("Avg Loopbacks / Run")
        ax_loops.set_ylim(0.0, 2.2)
        ax_loops.grid(True, linestyle="--", alpha=0.5)
        ax_loops.legend(loc="upper left", fontsize="x-small", ncol=2)

        if idx == 2:
            ax_success.set_xlabel("Volume Stress (Total Executed Tasks)")
            ax_loops.set_xlabel("Volume Stress (Total Executed Tasks)")

    fig.suptitle("Blackbox OS Multi-Role Volume Stress Test: Expert (SOP) vs. Bare (Flat)", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    chart_path = os.path.join(out_dir, "multi_role_stress_test_comparison.png")
    plt.savefig(chart_path, dpi=300)
    plt.close()
    print(f"Comparative progression curves saved to: {chart_path}\n")

if __name__ == "__main__":
    main()
