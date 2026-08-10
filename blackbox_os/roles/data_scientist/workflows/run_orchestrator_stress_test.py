#!/usr/bin/env python3
"""
Data Scientist Orchestrator Stress Test (Production V2)
Expert (partitioned SOP, ≤15 tools/stage) vs Bare (flat full catalog N=77)
30 live tasks × 2 modes × N models
"""

import os
import sys
import json
import csv
import time
import re
import ast
import random
from typing import Dict, Any, List, Optional, Tuple

# ── Workspace Root ───────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
root = SCRIPT_DIR
while root != os.path.dirname(root):
    if os.path.exists(os.path.join(root, "blackbox_os")):
        WORKSPACE_ROOT = root
        break
    root = os.path.dirname(root)
else:
    WORKSPACE_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "..", ".."))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODEL_MAP = {
    "deepseek-v4-flash": "deepseek/deepseek-chat",
    "gpt-4o-mini": "openai/gpt-4o-mini",
    "nvidia/nemotron-3-ultra-550b-a55b:free": "nvidia/nemotron-3-ultra-550b-a55b:free",
    "openai/gpt-oss-20b:free": "openai/gpt-oss-20b:free",
}

# ── Stage tool partitions (Expert mode) ──────────────────────────────────────
STAGE_TOOLS = {
    "audit": [
        {"id": "lookahead_bias_audit", "desc": "Detect lookahead/target leakage in feature formulas."},
        {"id": "data_drift_monitor", "desc": "Check feature drift against warning threshold."},
        {"id": "missing_value_report", "desc": "Report missing-value rates per column."},
        {"id": "schema_validator", "desc": "Validate CSV schema and dtypes."},
        {"id": "correlation_scan", "desc": "Scan pairwise correlations for leakage proxies."},
        {"id": "temporal_split_check", "desc": "Verify train/test temporal ordering."},
        {"id": "duplicate_row_audit", "desc": "Detect duplicate rows."},
        {"id": "outlier_summary", "desc": "Summarize outlier rates."},
        {"id": "class_balance_report", "desc": "Report class imbalance statistics."},
        {"id": "feature_cardinality_check", "desc": "Check categorical cardinality."},
    ],
    "remediate": [
        {"id": "mean_imputer", "desc": "Impute numeric columns with mean."},
        {"id": "median_imputer", "desc": "Impute numeric columns with median."},
        {"id": "knn_imputer", "desc": "KNN imputation with missingness indicators."},
        {"id": "mice_imputer", "desc": "Multiple imputation by chained equations."},
        {"id": "smote_balancer", "desc": "Apply SMOTE for class imbalance."},
        {"id": "drop_leaky_columns", "desc": "Drop columns flagged as leaky."},
        {"id": "standard_scaler", "desc": "StandardScaler on numeric features."},
        {"id": "robust_scaler", "desc": "RobustScaler on numeric features."},
        {"id": "minmax_scaler", "desc": "MinMax scaling."},
        {"id": "target_encoder", "desc": "Target encoding for categoricals."},
        {"id": "ordinal_encoder", "desc": "Ordinal encoding for categoricals."},
        {"id": "variance_filter", "desc": "Drop low-variance features."},
    ],
    "model": [
        {"id": "random_forest_fit", "desc": "Fit Random Forest classifier/regressor."},
        {"id": "gradient_boosting_fit", "desc": "Fit Gradient Boosting model."},
        {"id": "ridge_fit", "desc": "Fit Ridge regressor."},
        {"id": "lasso_fit", "desc": "Fit Lasso with feature selection."},
        {"id": "elasticnet_fit", "desc": "Fit ElasticNet."},
        {"id": "stacking_fit", "desc": "Fit stacking ensemble."},
        {"id": "voting_fit", "desc": "Fit voting classifier."},
        {"id": "bagging_fit", "desc": "Fit bagging ensemble."},
        {"id": "xgboost_fit", "desc": "Fit XGBoost model."},
        {"id": "lightgbm_fit", "desc": "Fit LightGBM model."},
        {"id": "mutual_info_select", "desc": "Mutual-information feature selection."},
        {"id": "rfe_select", "desc": "Recursive feature elimination."},
    ],
    "evaluate": [
        {"id": "grid_search_cv", "desc": "Grid search hyperparameter tuning."},
        {"id": "random_search_cv", "desc": "Random search hyperparameter tuning."},
        {"id": "optuna_tune", "desc": "Optuna hyperparameter study."},
        {"id": "bayesian_tune", "desc": "Bayesian optimization tuner."},
        {"id": "metrics_report", "desc": "Compute F1, ROC-AUC, log-loss, MCC."},
        {"id": "calibration_curve", "desc": "Generate calibration curve."},
        {"id": "shap_explain", "desc": "SHAP feature attribution."},
        {"id": "lime_explain", "desc": "LIME local explanations."},
        {"id": "permutation_importance", "desc": "Permutation feature importance."},
        {"id": "latency_check", "desc": "Measure inference latency."},
        {"id": "champion_challenger", "desc": "Champion/challenger model comparison."},
        {"id": "model_registry", "desc": "Register model version."},
        {"id": "drift_monitor_setup", "desc": "Attach post-deploy drift monitor."},
        {"id": "compliance_log", "desc": "Write compliance audit log."},
    ],
}

VALID_BY_STAGE = {stage: {t["id"] for t in tools} for stage, tools in STAGE_TOOLS.items()}

def locate_filler_file(filename: str) -> str:
    cwd = os.getcwd()
    candidate_paths = [
        os.path.join(WORKSPACE_ROOT, "skill_experiment", filename),
        os.path.join(WORKSPACE_ROOT, "blackbox_os", "skill_experiment", filename),
        os.path.join(WORKSPACE_ROOT, filename),
        os.path.join(SCRIPT_DIR, "skill_experiment", filename),
        os.path.join(SCRIPT_DIR, "..", "skill_experiment", filename),
        os.path.join(cwd, "skill_experiment", filename),
        os.path.join(cwd, filename),
    ]
    for path in candidate_paths:
        norm = os.path.abspath(path)
        if os.path.exists(norm):
            return norm
    raise FileNotFoundError(f"Could not locate '{filename}'.")

def build_bare_catalog(target_size: int = 77) -> List[Dict[str, str]]:
    """Builds full N=77 flat catalog by padding stage tools with background fillers."""
    base_tools = []
    seen = set()
    for tools in STAGE_TOOLS.values():
        for t in tools:
            if t["id"] not in seen:
                base_tools.append(t)
                seen.add(t["id"])
    
    needed = target_size - len(base_tools)
    if needed <= 0:
        return base_tools
    
    try:
        fillers = json.load(open(locate_filler_file("fillers_v4.json")))
        for f in fillers:
            if f["id"] not in seen:
                base_tools.append({"id": f["id"], "desc": f.get("concept", "Data processing tool.")})
                seen.add(f["id"])
                if len(base_tools) >= target_size:
                    break
    except Exception as e:
        print(f"Warning: Could not load background fillers for Bare N=77 ({e}). Using base {len(base_tools)} tools.")
    
    return base_tools

BARE_CATALOG_N77 = build_bare_catalog(77)
STAGES = ["audit", "remediate", "model", "evaluate"]

# ── 30 User Goals ────────────────────────────────────────────────────────────
USER_GOALS = [
    "Clean returns.csv with mean imputation, standard scale, train Random Forest, and deploy.",
    "Examine features.csv for target leakage. Impute with KNN, RobustScale, fit linear regression, deploy.",
    "Scan for data drift. Median impute, min-max scale, train gradient boosting, run compliance.",
    "Handle class imbalance with SMOTE, scale data, fit stacking classifier, run interpretability.",
    "Impute with MICE, Target Encode categoricals, fit Ridge, verify temporal leakage.",
    "Mutual-information feature selection, Lasso model, check lookahead bias.",
    "RFE on columns, voting classifier, SHAP audit.",
    "Variance filter, ElasticNet, data drift check, register champion.",
    "Stacking with RF+GB, Ridge regularization, scan target leakage.",
    "Bagging classifier, ordinal encode, select top features, verify calibration.",
    "Grid Search CV for random forest, report log-loss and F1.",
    "Random Search for Gradient Boosting, precision-recall and MCC.",
    "Optuna tune neural weight decay, stratified K-Fold, confusion matrix.",
    "Bayesian tune XGBoost, time-series CV verification.",
    "LightGBM fit, nested CV, ROC-AUC, calibration curve.",
    "Audit features.csv for leakage, permutation importance, drift monitor, deploy.",
    "Check temporal leakage, LIME explain, latency metrics, register model.",
    "Lookahead bias scan, partial dependence, drift verify, latency alerts.",
    "Group leakage audit, SHAP top features, champion-challenger, registry.",
    "Target contamination verify, rank importance, model drift monitor, compliance log.",
    "Impute missing, Lasso select, Random Search, audit temporal leakage, deploy.",
    "SMOTE balance, stacking classifier, Optuna, data drift check, register.",
    "RobustScaler, Gradient Boosting, Grid Search, lookahead scan, latency verify.",
    "MICE impute, Target Encode, Ridge, target leakage check, SHAP.",
    "Mutual Information select, Voting classifier, time-series validation, drift monitor.",
    "SMOTE, Random Forest, Optuna, lookahead audit, deploy champion.",
    "Mean impute, RFE, ElasticNet, precision-recall, drift monitor.",
    "Standard scale, Bagging, stratified CV, SHAP, register champion.",
    "Log transform, variance threshold, gradient boosting, F1, latency verify.",
    "Median impute, stacking, Bayesian tune, lookahead check, deploy version.",
]

def generate_task_meta(task_idx: int) -> Dict[str, Any]:
    return {
        "task_idx": task_idx,
        "goal": USER_GOALS[task_idx],
    }

# ── LLM Helpers ──────────────────────────────────────────────────────────────
def clean_json(raw: str) -> Optional[Dict]:
    if not raw:
        return None
    cleaned = re.sub(r"<think>[\s\S]*?</think>", "", raw).strip()
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start != -1 and end > start:
        cand = cleaned[start:end + 1]
        try:
            return json.loads(cand)
        except Exception:
            try:
                return ast.literal_eval(cand)
            except Exception:
                pass
    return None

def query_llm(system_prompt: str, user_prompt: str, model_name: str) -> str:
    import urllib.request
    api_key = os.environ.get("OPENROUTER_API_KEY", "Your_API_Key")
    if not api_key:
        return '{"error": "OPENROUTER_API_KEY not set"}'
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": "https://github.com/google/antigravity",
    }
    target = OPENROUTER_MODEL_MAP.get(model_name, model_name)
    payload = {
        "model": target,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.0,
        "max_tokens": 2048,
    }
    for attempt in range(4):
        try:
            req = urllib.request.Request(OPENROUTER_URL, method="POST")
            for k, v in headers.items():
                req.add_header(k, v)
            with urllib.request.urlopen(req, data=json.dumps(payload).encode(), timeout=45) as resp:
                return json.loads(resp.read().decode())["choices"][0]["message"]["content"]
        except Exception as e:
            if "429" in str(e) and attempt < 3:
                time.sleep(2 * (attempt + 1))
                continue
            return f'{{"error": "{str(e)}"}}'
    return '{"error": "max retries"}'

def select_skill(model_name: str, stage: str, goal: str, tools: List[Dict], history: List[str]) -> Optional[str]:
    listing = "\n".join(f"{t['id']}: {t['desc']}" for t in tools)
    hist = " | ".join(history[-6:]) if history else "None"
    system = (
        "You are a multi-step data-science orchestrator.\n"
        f"Current stage: {stage}.\n"
        "Select the single most appropriate skill_id for this stage given the user goal.\n"
        "Return ONLY JSON: {\"chosen_skill_id\": \"skill_id\"}\n\n"
        f"Available skills:\n{listing}"
    )
    user = f"User goal: {goal}\nPrior stages: {hist}\nSelect the skill for stage '{stage}'."
    raw = query_llm(system, user, model_name)
    parsed = clean_json(raw)
    if not parsed:
        return None
    return parsed.get("chosen_skill_id") or parsed.get("skill_id")

# ── Stage Validation ──────────────────────────────────────────────────────────
def stage_success(stage: str, chosen: Optional[str]) -> bool:
    """Validates that chosen skill belongs to the expected stage partition."""
    if not chosen or chosen not in VALID_BY_STAGE[stage]:
        return False
    return True

# ── One Task Execution ───────────────────────────────────────────────────────
def run_one_task(model_name: str, mode: str, meta: Dict[str, Any], max_loopbacks: int = 2) -> Dict[str, Any]:
    history = []
    loopbacks = 0
    stage_logs = []

    for stage in STAGES:
        tools = STAGE_TOOLS[stage] if mode == "expert" else BARE_CATALOG_N77
        attempts = 0
        ok = False
        chosen = None

        while attempts <= max_loopbacks and not ok:
            chosen = select_skill(model_name, stage, meta["goal"], tools, history)
            ok = stage_success(stage, chosen)
            
            stage_logs.append({
                "stage": stage,
                "attempt": attempts,
                "chosen": chosen,
                "success": ok,
                "n_tools_exposed": len(tools),
            })
            
            if not ok:
                if attempts < max_loopbacks:
                    loopbacks += 1
                attempts += 1
                time.sleep(0.25)
            else:
                history.append(f"{stage}:{chosen}")
                break

        if not ok:
            return {
                "success": False,
                "direct_success": False,
                "loopbacks": loopbacks,
                "failed_stage": stage,
                "stage_logs": stage_logs,
            }

    return {
        "success": True,
        "direct_success": (loopbacks == 0),
        "loopbacks": loopbacks,
        "failed_stage": None,
        "stage_logs": stage_logs,
    }

# ── Main Sweep ───────────────────────────────────────────────────────────────
def run_stress_sweep(models: List[str], n_tasks: int = 30, max_loopbacks: int = 2):
    print("=" * 82)
    print("ORCHESTRATOR STRESS TEST — Expert (partitioned ≤15) vs Bare (flat N=77)")
    print(f"Tasks={n_tasks} | Stages={STAGES} | Max loopbacks/stage={max_loopbacks}")
    print("=" * 82)

    all_results = {}

    for model in models:
        print(f"\n######## MODEL: {model} ########")
        all_results[model] = {"expert": [], "bare": []}

        for mode in ["expert", "bare"]:
            print(f"\n── Mode: {mode.upper()} ──")
            for i in range(n_tasks):
                meta = generate_task_meta(i)
                res = run_one_task(model, mode, meta, max_loopbacks=max_loopbacks)
                all_results[model][mode].append({**res, "task_idx": i, "goal": meta["goal"]})
                
                icon = "✓" if res["success"] else "✗"
                kind = "direct" if res["direct_success"] else ("recovered" if res["success"] else "fail")
                print(f"  [{mode}] {i+1:02d}/{n_tasks} {icon}  {kind:9s}  loopbacks={res['loopbacks']}")
                time.sleep(0.2)

    # ── Corrected Summary Reporting Block ─────────────────────────────────────
    print("\n" + "=" * 82)
    print(f"{'Model':<28} {'Mode':<7} {'E2E %':>7} {'Direct %':>9} {'Loopback Rec %':>15} {'Avg Loop/Run':>13}")
    print("-" * 82)
    summary = {}
    
    for model in models:
        summary[model] = {}
        for mode in ["expert", "bare"]:
            runs = all_results[model][mode]
            n = len(runs)
            
            n_e2e = sum(1 for r in runs if r["success"])
            n_direct = sum(1 for r in runs if r["direct_success"])
            n_initial_failures = n - n_direct
            n_recovered = n_e2e - n_direct
            
            e2e_pct = (n_e2e / n) * 100
            direct_pct = (n_direct / n) * 100
            
            # Corrected Loopback Recovery Rate Formula
            loopback_rec_pct = (n_recovered / n_initial_failures * 100) if n_initial_failures > 0 else 100.0
            avg_loop = sum(r["loopbacks"] for r in runs) / n

            summary[model][mode] = {
                "e2e": e2e_pct,
                "direct": direct_pct,
                "loopback_recovery": loopback_rec_pct,
                "avg_loopbacks": avg_loop
            }
            print(f"{model:<28} {mode:<7} {e2e_pct:7.1f} {direct_pct:9.1f} {loopback_rec_pct:15.1f} {avg_loop:13.2f}")
            
    print("=" * 82)

    out_dir = os.path.join(WORKSPACE_ROOT, "blackbox_os", "roles", "data_scientist", "workflows")
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "results_orchestrator_expert_vs_bare.json")
    with open(out_file, "w") as f:
        json.dump({"summary": summary, "runs": all_results}, f, indent=2)
    print(f"Saved → {out_file}")
    return summary

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=[
        "deepseek-v4-flash",
        "gpt-4o-mini",
        "nvidia/nemotron-3-ultra-550b-a55b:free",
        "openai/gpt-oss-20b:free",
    ])
    parser.add_argument("--tasks", type=int, default=30)
    parser.add_argument("--max-loopbacks", type=int, default=2)
    args = parser.parse_args()

    if not os.environ.get("OPENROUTER_API_KEY"):
        print("ERROR: set OPENROUTER_API_KEY")
        sys.exit(1)

    run_stress_sweep(args.models, n_tasks=args.tasks, max_loopbacks=args.max_loopbacks)
