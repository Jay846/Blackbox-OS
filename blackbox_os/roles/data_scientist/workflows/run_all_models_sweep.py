"""
run_all_models_sweep.py
=======================
Batch runner: sweeps all 7 benchmark models sequentially through
run_sandbox_experiment.py for both clean (Table 3) and noisy (Table 4)
query conditions.

Usage (from blackbox_os/ directory):
    python3 roles/data_scientist/workflows/run_all_models_sweep.py
    python3 roles/data_scientist/workflows/run_all_models_sweep.py --dry-run
    python3 roles/data_scientist/workflows/run_all_models_sweep.py --clean-only
    python3 roles/data_scientist/workflows/run_all_models_sweep.py --noise-only
"""

import os
import sys
import json
import time
import subprocess
import argparse
from datetime import datetime

# ── Model Registry ────────────────────────────────────────────────────────────
# All 7 benchmark models routed through OpenRouter
MODELS = [
    {"label": "DeepSeek V4 Flash",    "model": "deepseek-chat",        "provider": "openrouter"},
    {"label": "DeepSeek V4 Pro",      "model": "deepseek-v4-pro",      "provider": "openrouter"},
    {"label": "GPT-4o-mini",          "model": "gpt-4o-mini",          "provider": "openrouter"},
    {"label": "GPT-OSS 20B",          "model": "gpt-oss-20b",          "provider": "openrouter"},
    {"label": "Google Gemma 26B",     "model": "google-gemma-26b",     "provider": "openrouter"},
    {"label": "Google Gemma 31B",     "model": "google-gemma-31b",     "provider": "openrouter"},
    {"label": "NVIDIA Nemotron 550B", "model": "nvidia-nemotron-550b", "provider": "openrouter"},
]

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
RUNNER      = os.path.join(SCRIPT_DIR, "run_sandbox_experiment.py")

# ── Helpers ───────────────────────────────────────────────────────────────────
def model_result_path(model_slug: str, noise: bool) -> str:
    """Return the output JSON path that run_sandbox_experiment.py writes."""
    clean_slug = model_slug.replace("/", "_").replace("-", "_")
    prefix = "results_sandbox_expanded_noise_sweep" if noise else "results_sandbox_sweep"
    # run_sandbox_experiment.py writes relative to CWD (blackbox_os/)
    return os.path.join(
        "blackbox_os", "roles", "data_scientist", "workflows",
        f"{prefix}_{clean_slug}.json"
    )

def run_one(entry: dict, noise: bool, dry_run: bool) -> dict:
    label    = entry["label"]
    model    = entry["model"]
    provider = entry["provider"]
    mode     = "NOISY" if noise else "CLEAN"

    print(f"\n{'='*75}")
    print(f"  ▶  {label}  |  Mode: {mode}  |  Provider: {provider}")
    print(f"{'='*75}")

    cmd = [
        sys.executable, RUNNER,
        "--model",    model,
        "--provider", provider,
    ]
    if dry_run:
        cmd.append("--dry-run")
    if noise:
        cmd.append("--noise")

    env = os.environ.copy()          # passes OPENROUTER_API_KEY / DEEPSEEK_API_KEY

    start = time.time()
    proc  = subprocess.run(cmd, env=env, cwd=os.path.join(SCRIPT_DIR, "..", "..", ".."))
    elapsed = time.time() - start

    status = "✅ OK" if proc.returncode == 0 else "❌ FAILED"
    print(f"\n  {status} — {label} [{mode}] finished in {elapsed:.1f}s")

    # Read results JSON written by the runner
    result_path = model_result_path(model, noise)
    data = {}
    if os.path.exists(result_path):
        try:
            with open(result_path) as f:
                data = json.load(f).get("results", {})
        except Exception as e:
            print(f"  ⚠ Could not read result file: {e}")

    return {"label": label, "model": model, "mode": mode, "data": data, "ok": proc.returncode == 0}

def print_summary_table(all_results: list, sizes: list, condition: str):
    """Print a combined E2E accuracy summary table for all models."""
    SIZES = [str(s) for s in sizes]
    col_w = 22

    header  = f"{'Model':<25}" + "".join(f"  N={s:<6}" for s in SIZES)
    divider = "-" * len(header)

    print(f"\n  Condition: {condition.upper()}")
    print(f"  {'Model':<25}" + "".join(f"  {'N='+s:<8}" for s in SIZES))
    print("  " + divider)

    for r in all_results:
        row = f"  {r['label']:<25}"
        for s in sizes:
            try:
                e2e = r["data"][str(s)][condition]["e2e_acc"]
                row += f"  {e2e:>5.1f}%  "
            except Exception:
                row += f"  {'N/A':<8}"
        print(row)

def print_master_table(clean_results: list, noise_results: list, sizes: list):
    conditions = [("expert", "Expert SOP (Prompt Math)"), ("expert_sandbox", "Expert Sandbox (Python)")]
    print("\n\n" + "="*75)
    print("  MASTER SUMMARY TABLE — ALL MODELS")
    print("="*75)

    for cond_key, cond_label in conditions:
        print(f"\n── {cond_label} ──")
        if clean_results:
            print("  [ CLEAN QUERIES ]")
            print_summary_table(clean_results, sizes, cond_key)
        if noise_results:
            print("  [ NOISY QUERIES ]")
            print_summary_table(noise_results, sizes, cond_key)

    print("\n" + "="*75)
    print("  Full logs saved to:")
    print("  blackbox_os/roles/data_scientist/workflows/results_sandbox_*.json")
    print("="*75)

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Batch sweep all 7 models through run_sandbox_experiment.py")
    parser.add_argument("--dry-run",    action="store_true", help="Only run N=60 and N=200 (quick test)")
    parser.add_argument("--clean-only", action="store_true", help="Only run clean query sweep (Table 2)")
    parser.add_argument("--noise-only", action="store_true", help="Only run noisy query sweep (Table 4)")
    args = parser.parse_args()

    run_clean = not args.noise_only
    run_noise = not args.clean_only
    sizes     = [60, 200] if args.dry_run else [60, 200, 500, 1000]

    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        print("⚠  WARNING: OPENROUTER_API_KEY not set — API calls will fail.")
    else:
        print(f"✅ OPENROUTER_API_KEY detected ({api_key[:12]}...)")

    print(f"\n🚀 Blackbox OS — Multi-Model Benchmark Sweep")
    print(f"   Models    : {len(MODELS)}")
    print(f"   Conditions: {'Clean + Noisy' if run_clean and run_noise else 'Clean only' if run_clean else 'Noisy only'}")
    print(f"   Sizes     : {sizes}")
    print(f"   Dry-run   : {args.dry_run}")
    print(f"   Started   : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    clean_results = []
    noise_results = []

    for i, entry in enumerate(MODELS, 1):
        print(f"\n[{i}/{len(MODELS)}] Processing: {entry['label']}")

        if run_clean:
            result = run_one(entry, noise=False, dry_run=args.dry_run)
            clean_results.append(result)

        if run_noise:
            result = run_one(entry, noise=True, dry_run=args.dry_run)
            noise_results.append(result)

    # Print combined master summary table
    print_master_table(clean_results, noise_results, sizes)
    print(f"\n  Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()
