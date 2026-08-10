#!/usr/bin/env python3
"""
Empirical attention / scale degradation
- Live tool selection error vs catalog size N
- Live JSON schema compliance vs N
- Bare (flat catalog of size N) vs Expert (partition ≤15 tools)

Usage:
  export OPENROUTER_API_KEY=...
  python test_attention_degradation.py --models deepseek-v4-flash gpt-4o-mini --trials 3
"""

import os
import sys
import json
import random
import time
import re
import ast
import argparse
from typing import List, Dict, Any, Optional, Tuple

import matplotlib.pyplot as plt

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

# ── Targets ──────────────────────────────────────────────────────────────────
TARGETS = [
    {
        "id": "lookahead_bias_audit",
        "desc": "Audit feature formulas for lookahead/target leakage. Output leakage_detected bool.",
        "query": "Audit features.csv for lookahead bias. Check if any formula references future timestamps.",
        "schema_keys": {"chosen_skill_id": str, "leakage_detected": bool},
        "partition": ["lookahead_bias_audit", "data_drift_monitor", "schema_validator", "temporal_split_check", "correlation_scan"],
    },
    {
        "id": "trade_ev_calculator",
        "desc": "Compute trade expected value after fee from JSON trades. Output ev float.",
        "query": "Load trade_performance.json and calculate net expected value deducting a fee of 5.0.",
        "schema_keys": {"chosen_skill_id": str, "ev": float},
        "partition": ["trade_ev_calculator", "kelly_position_size", "metrics_report", "risk_reward_ratio", "pnl_summary"],
    },
    {
        "id": "kelly_position_size",
        "desc": "Compute Kelly fraction and dollar amount from trade PnL. Output fraction and amount.",
        "query": "Analyze BTC trades in fills.csv and calculate Kelly fraction for a $50,000 bankroll.",
        "schema_keys": {"chosen_skill_id": str, "fraction": float, "amount": float},
        "partition": ["kelly_position_size", "trade_ev_calculator", "position_sizer", "volatility_target", "pnl_summary"],
    },
    {
        "id": "atr_dynamic_stop",
        "desc": "Compute ATR trailing stop distance from OHLC. Output stop_distance float.",
        "query": "Read prices.csv, compute 14-period ATR, and find stop distance with multiplier 2.0.",
        "schema_keys": {"chosen_skill_id": str, "stop_distance": float},
        "partition": ["atr_dynamic_stop", "volatility_stop", "price_channel", "risk_reward_ratio", "metrics_report"],
    },
    {
        "id": "data_drift_monitor",
        "desc": "Check feature drift against warning threshold. Output data_drift_detected bool.",
        "query": "Analyze drift.json and check if implied_vol exceeds the warning threshold.",
        "schema_keys": {"chosen_skill_id": str, "data_drift_detected": bool},
        "partition": ["data_drift_monitor", "lookahead_bias_audit", "schema_validator", "correlation_scan", "missing_value_report"],
    },
]
TARGET_IDS = {t["id"] for t in TARGETS}

# Extra names used only to fill Expert partitions if not in fillers
PARTITION_STUBS = {
    "schema_validator": "Validate CSV schema and dtypes.",
    "temporal_split_check": "Verify train/test temporal ordering.",
    "correlation_scan": "Scan correlations for leakage proxies.",
    "metrics_report": "Compute standard classification/regression metrics.",
    "risk_reward_ratio": "Compute risk reward ratio from trades.",
    "pnl_summary": "Summarize realized PnL statistics.",
    "position_sizer": "Generic position sizing helper.",
    "volatility_target": "Volatility targeting position scaler.",
    "volatility_stop": "Volatility-based stop helper.",
    "price_channel": "Price channel breakout helper.",
    "missing_value_report": "Report missing value rates.",
}

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

def load_fillers() -> List[Dict[str, str]]:
    fillers, seen = [], set(TARGET_IDS)
    for fn in ["fillers_v4.json", "hq_fillers.json", "additional_fillers.json"]:
        try:
            data = json.load(open(locate_filler_file(fn)))
        except Exception:
            continue
        for item in data:
            fid = item.get("id")
            if not fid or fid in seen:
                continue
            desc = item.get("expert_desc") or item.get("concept") or item.get("desc") or ""
            if item.get("disambiguator"):
                desc = f"{desc} {item['disambiguator']}".strip()
            if not desc:
                continue
            fillers.append({"id": fid, "desc": desc})
            seen.add(fid)
    print(f"Loaded {len(fillers)} fillers.")
    return fillers

def seeded_sample(items: list, k: int, seed: int) -> list:
    a = list(items)
    random.Random(seed).shuffle(a)
    return a[:k]

def clean_json(raw: str) -> Optional[Dict]:
    if not raw:
        return None
    cleaned = re.sub(r"<think>[\s\S]*?</think>", "", raw).strip()
    s, e = cleaned.find("{"), cleaned.rfind("}")
    if s == -1 or e <= s:
        return None
    cand = cleaned[s : e + 1]
    try:
        return json.loads(cand)
    except Exception:
        try:
            return ast.literal_eval(cand)
        except Exception:
            return None

def query_llm(system: str, user: str, model_name: str) -> str:
    import urllib.request
    key = os.environ.get("OPENROUTER_API_KEY", "Your_API_Key")
    if not key:
        return '{"error":"OPENROUTER_API_KEY not set"}'
    target = OPENROUTER_MODEL_MAP.get(model_name, model_name)
    payload = {
        "model": target,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0.0,
        "max_tokens": 2048,
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {key}",
        "HTTP-Referer": "https://github.com/google/antigravity",
    }
    for attempt in range(4):
        try:
            req = urllib.request.Request(OPENROUTER_URL, method="POST")
            for k, v in headers.items():
                req.add_header(k, v)
            with urllib.request.urlopen(req, data=json.dumps(payload).encode(), timeout=45) as resp:
                return json.loads(resp.read().decode())["choices"][0]["message"]["content"]
        except Exception as ex:
            if "429" in str(ex) and attempt < 3:
                time.sleep(2 * (attempt + 1))
                continue
            return f'{{"error":"{ex}"}}'
    return '{"error":"retries"}'

def build_bare_catalog(fillers: List[Dict], N: int, seed: int) -> List[Dict[str, str]]:
    n_t = len(TARGETS)
    dist = seeded_sample(fillers, max(0, N - n_t), seed=seed)
    tools = [{"id": t["id"], "desc": t["desc"]} for t in TARGETS] + dist
    random.Random(seed + 3).shuffle(tools)
    return tools[:N] if len(tools) > N else tools

def build_expert_partition(target: Dict, fillers: List[Dict], max_tools: int = 15) -> List[Dict[str, str]]:
    """Small SOP-style list: true target + partition ids resolved from fillers/stubs."""
    by_id = {f["id"]: f for f in fillers}
    tools = []
    seen = set()
    for pid in target["partition"]:
        if pid in seen:
            continue
        if pid == target["id"]:
            tools.append({"id": target["id"], "desc": target["desc"]})
        elif pid in by_id:
            tools.append(by_id[pid])
        elif pid in PARTITION_STUBS:
            tools.append({"id": pid, "desc": PARTITION_STUBS[pid]})
        else:
            continue
        seen.add(pid)
        if len(tools) >= max_tools:
            break
    # ensure target present
    if target["id"] not in seen:
        tools.insert(0, {"id": target["id"], "desc": target["desc"]})
    return tools[:max_tools]

def format_tools(tools: List[Dict[str, str]]) -> str:
    return "\n".join(f"{t['id']}: {t['desc']}" for t in tools)

def schema_ok(parsed: Optional[Dict], required: Dict[str, type]) -> bool:
    if not isinstance(parsed, dict):
        return False
    if "error" in parsed and len(parsed) == 1:
        return False
    for k, typ in required.items():
        if k not in parsed:
            return False
        v = parsed[k]
        if typ is float:
            if not isinstance(v, (int, float)):
                return False
        elif typ is bool:
            if not isinstance(v, bool):
                return False
        elif typ is str:
            if not isinstance(v, str) or not v:
                return False
    return True

def run_one(
    model_name: str,
    target: Dict,
    tools: List[Dict[str, str]],
) -> Tuple[bool, bool, Optional[str]]:
    """
    Returns (selection_correct, schema_compliant, chosen_id)
    Request both routing and schema fields so schema pressure is real.
    """
    listing = format_tools(tools)
    key_list = ", ".join(target["schema_keys"].keys())
    system = (
        "You are a tool-routing and structured-output component.\n"
        "1) Select the single best skill_id for the user query from the list.\n"
        "2) Return ONLY one JSON object containing ALL of these keys: "
        f"{key_list}.\n"
        "Use plausible typed values for non-id fields "
        "(bool true/false, floats as numbers).\n"
        "No markdown, no extra keys required beyond the schema keys.\n\n"
        f"Available skills:\n{listing}"
    )
    user = target["query"]
    raw = query_llm(system, user, model_name)
    parsed = clean_json(raw)
    chosen = None
    if parsed:
        chosen = parsed.get("chosen_skill_id") or parsed.get("skill_id")
    sel_ok = chosen == target["id"]
    sch_ok = schema_ok(parsed, target["schema_keys"])
    return sel_ok, sch_ok, chosen

def evaluate_model(
    model_name: str,
    fillers: List[Dict],
    sizes: List[int],
    trials: int,
    expert_max_tools: int = 15,
) -> Dict[str, Any]:
    result = {
        "N": [],
        "bare_selection_error": [],
        "expert_selection_error": [],
        "bare_schema_compliance": [],
        "expert_schema_compliance": [],
    }
    print(f"\n######## {model_name} ########")
    for N in sizes:
        bare_sel_err = bare_sch_ok = 0
        exp_sel_err = exp_sch_ok = 0
        total = 0

        for trial in range(trials):
            bare_catalog = build_bare_catalog(fillers, N, seed=10_000 + N * 31 + trial)
            for target in TARGETS:
                # Bare
                sel_ok, sch_ok, _ = run_one(model_name, target, bare_catalog)
                bare_sel_err += int(not sel_ok)
                bare_sch_ok += int(sch_ok)

                # Expert partition (size capped, independent of N except N can be smaller)
                part = build_expert_partition(target, fillers, max_tools=min(expert_max_tools, max(3, N)))
                sel_ok_e, sch_ok_e, _ = run_one(model_name, target, part)
                exp_sel_err += int(not sel_ok_e)
                exp_sch_ok += int(sch_ok_e)

                total += 1
                time.sleep(0.15)

        result["N"].append(N)
        result["bare_selection_error"].append(100.0 * bare_sel_err / total)
        result["expert_selection_error"].append(100.0 * exp_sel_err / total)
        result["bare_schema_compliance"].append(100.0 * bare_sch_ok / total)
        result["expert_schema_compliance"].append(100.0 * exp_sch_ok / total)

        print(
            f"N={N:<4}  "
            f"Bare err {result['bare_selection_error'][-1]:5.1f}%  "
            f"Expert err {result['expert_selection_error'][-1]:5.1f}%  |  "
            f"Bare schema {result['bare_schema_compliance'][-1]:5.1f}%  "
            f"Expert schema {result['expert_schema_compliance'][-1]:5.1f}%"
        )
    return result

def plot_results(all_results: Dict[str, Any], out_path: str):
    # Average across models for a clean main figure; also save per-model in JSON
    sizes = next(iter(all_results.values()))["N"]
    bare_err, exp_err, bare_sch, exp_sch = [], [], [], []
    for i in range(len(sizes)):
        bare_err.append(sum(all_results[m]["bare_selection_error"][i] for m in all_results) / len(all_results))
        exp_err.append(sum(all_results[m]["expert_selection_error"][i] for m in all_results) / len(all_results))
        bare_sch.append(sum(all_results[m]["bare_schema_compliance"][i] for m in all_results) / len(all_results))
        exp_sch.append(sum(all_results[m]["expert_schema_compliance"][i] for m in all_results) / len(all_results))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(sizes, bare_err, "o--", color="#ef4444", lw=2, label="Bare (flat catalog)")
    ax1.plot(sizes, exp_err, "s-", color="#2563eb", lw=2.3, label="Expert (partition ≤15)")
    ax1.set_xlabel("Catalog size N")
    ax1.set_ylabel("Selection error rate (%)")
    ax1.set_title("Tool selection error vs N")
    ax1.grid(True, ls="--", alpha=0.4)
    ax1.legend(fontsize=9)

    ax2.plot(sizes, bare_sch, "o--", color="#f59e0b", lw=2, label="Bare")
    ax2.plot(sizes, exp_sch, "s-", color="#10b981", lw=2.3, label="Expert")
    ax2.set_xlabel("Catalog size N")
    ax2.set_ylabel("Schema compliance (%)")
    ax2.set_title("JSON schema compliance vs N")
    ax2.set_ylim(0, 105)
    ax2.grid(True, ls="--", alpha=0.4)
    ax2.legend(fontsize=9)

    fig.suptitle("Empirical scale degradation: Bare vs Expert", fontweight="bold")
    fig.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["deepseek-v4-flash", "gpt-4o-mini"])
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--sizes", nargs="+", type=int, default=[5, 15, 30, 60, 100, 150, 200])
    args = parser.parse_args()

    if not os.environ.get("OPENROUTER_API_KEY"):
        print("ERROR: set OPENROUTER_API_KEY")
        sys.exit(1)

    print("=" * 78)
    print("EMPIRICAL ATTENTION / SCALE DEGRADATION (LIVE)")
    print(f"Models={args.models}  trials/target={args.trials}  sizes={args.sizes}")
    print("=" * 78)

    fillers = load_fillers()
    all_results = {}
    for m in args.models:
        all_results[m] = evaluate_model(m, fillers, args.sizes, args.trials)

    out_dir = os.path.join(WORKSPACE_ROOT, "blackbox_os", "roles", "data_scientist", "workflows")
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, "attention_degradation_results.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    img_path = os.path.join(out_dir, "attention_degradation_curve.png")
    plot_results(all_results, img_path)
    print(f"\nSaved JSON → {json_path}")
    print(f"Saved figure → {img_path}")

if __name__ == "__main__":
    main()
