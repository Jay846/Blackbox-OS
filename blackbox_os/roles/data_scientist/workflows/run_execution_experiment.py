import json
import urllib.request
import math
import random
from typing import Any, Dict, List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import os

DEEPSEEK_API_KEY = os.environ.get('DEEPSEEK_API_KEY', 'YOUR_API_KEY_HERE')
DEEPSEEK_URL = 'https://api.deepseek.com/chat/completions'
OPENROUTER_URL = 'https://openrouter.ai/api/v1/chat/completions'

# Global defaults
MODEL = 'deepseek-chat'
PROVIDER = 'deepseek'

# 1. Target Execution Skills and Queries
TARGET_SKILLS = [
    {
        "id": "kelly_position_size",
        "bare_desc": "Compute Kelly fraction and dollar amount based on edge, payoff, and bankroll.",
        "expert_desc": (
            "Computes the optimal leverage and bet size to maximize logarithmic growth rate. "
            "Formula: f = (win_rate * payoff - (1 - win_rate)) / payoff. Dollar amount = f * bankroll. "
            "Edge case: If win_rate <= 0 or payoff <= 0, return fraction = 0.0, amount = 0.0. "
            "Output schema: {\"fraction\": float, \"amount\": float}. Do not confuse with kelly_fractional_sizing."
        ),
        "queries": [
            {
                "text": "Calculate the Kelly fraction and amount for win_rate=0.55, payoff=1.8, and bankroll=10000.",
                "ground_truth": {"fraction": 0.30, "amount": 3000.0}
            },
            {
                "text": "Find my bet sizing using the Kelly criterion for win_rate=0.60, payoff=1.5, and bankroll=50000.",
                "ground_truth": {"fraction": 0.333333, "amount": 16666.67}
            }
        ]
    },
    {
        "id": "atr_dynamic_stop",
        "bare_desc": "Compute stop distance from ATR and multiplier.",
        "expert_desc": (
            "Calculates the trailing stop distance based on ATR. Formula: stop_distance = atr * multiplier. "
            "Edge case: If atr <= 0 or multiplier <= 0, return stop_distance = 0.0. "
            "Output schema: {\"stop_distance\": float}. Do not confuse with standard fixed stop losses."
        ),
        "queries": [
            {
                "text": "Compute stop distance from atr=15.2 and multiplier=2.0.",
                "ground_truth": {"stop_distance": 30.4}
            },
            {
                "text": "Determine trailing stop distance for atr=2.5 and multiplier=3.0.",
                "ground_truth": {"stop_distance": 7.5}
            }
        ]
    },
    {
        "id": "trade_ev_calculator",
        "bare_desc": "Compute net expected value of a trade with fee.",
        "expert_desc": (
            "Computes expected value including transaction fee. "
            "Formula: ev = (win_prob * (win_return - fee)) + ((1 - win_prob) * (loss_return - fee)). "
            "Output schema: {\"ev\": float}. Do not confuse with raw risk/reward ratio."
        ),
        "queries": [
            {
                "text": "Compute expected value with win_prob=0.6, win_return=200.0, loss_return=-100.0, and fee=5.0.",
                "ground_truth": {"ev": 75.0}
            },
            {
                "text": "Calculate trade net EV for win_prob=0.4, win_return=300.0, loss_return=-50.0, and fee=10.0.",
                "ground_truth": {"ev": 80.0}
            }
        ]
    },
    {
        "id": "lookahead_bias_audit",
        "bare_desc": "Detect lookahead bias in columns.",
        "expert_desc": (
            "Audits column formulas for leakage/lookahead bias. Criteria: If a column uses information "
            "from the target variable or future timestamps, leakage_detected is true. "
            "Output schema: {\"leakage_detected\": bool, \"reason\": str}. Do not confuse with data drift checks."
        ),
        "queries": [
            {
                "text": "Detect lookahead leakage in column_name='total_revenue_offset' calculated as 'target * 5 + 10'.",
                "ground_truth": {"leakage_detected": True, "contains_reason": "target"}
            },
            {
                "text": "Check column 'lagged_price_1' calculated as 'price_t minus 1' for lookahead bias.",
                "ground_truth": {"leakage_detected": False}
            }
        ]
    },
    {
        "id": "standard_scaler_apply",
        "bare_desc": "Compute StandardScaler parameters.",
        "expert_desc": (
            "Computes population StandardScaler parameters. Formula: mean = sum(x)/N, std = sqrt(sum((x - mean)^2)/N). "
            "Output schema: {\"mean\": float, \"std\": float}. Do not confuse with sample standard deviation (ddof=1)."
        ),
        "queries": [
            {
                "text": "Compute population mean and std for values=[12, 17, 33, 44, 51].",
                "ground_truth": {"mean": 31.4, "std": 15.027973915}
            },
            {
                "text": "Apply StandardScaler on list [10, 20, 30, 40, 50] to find mean and std.",
                "ground_truth": {"mean": 30.0, "std": 14.1421356}
            }
        ]
    },
    {
        "id": "data_drift_monitor",
        "bare_desc": "Given KS score, check for drift.",
        "expert_desc": (
            "Evaluates KS score against threshold. If ks_score > threshold, data_drift_detected is true. "
            "Output schema: {\"data_drift_detected\": bool}. Do not confuse with target leakage detector."
        ),
        "queries": [
            {
                "text": "Evaluate if data drift is detected for ks_score=0.35 and threshold=0.25.",
                "ground_truth": {"data_drift_detected": True}
            },
            {
                "text": "Check for drift given ks_score=0.12 and threshold=0.20.",
                "ground_truth": {"data_drift_detected": False}
            }
        ]
    }
]

# Helper to shuffle fillers deterministically
def seeded_shuffle(arr: list, seed: int) -> list:
    a = arr.copy()
    r = random.Random(seed)
    r.shuffle(a)
    return a

def build_library(size: int, fillers: List[Dict[str, Any]], seed: int) -> List[Dict[str, Any]]:
    # Filter target skills out of fillers to avoid duplicate IDs
    target_ids = {t["id"] for t in TARGET_SKILLS}
    unique_fillers = [f for f in fillers if f.get("id") not in target_ids]
    
    n_fillers = max(0, size - len(TARGET_SKILLS))
    shuffled_fillers = seeded_shuffle(unique_fillers, seed)[:n_fillers]
    
    # Mix targets and fillers, then shuffle again
    return seeded_shuffle(TARGET_SKILLS + shuffled_fillers, seed + 1)

def format_skill_list(library: List[Dict[str, Any]], condition: str) -> str:
    lines = []
    for s in library:
        if s.get("id") in {t["id"] for t in TARGET_SKILLS}:
            # Use defined target descriptions
            desc = s["expert_desc"] if condition == "expert" else s["bare_desc"]
            lines.append(f"{s['id']}: {desc}")
        else:
            # Filler skills
            concept = s.get("concept", "Tool option.")
            if condition == "expert" and "disambiguator" in s:
                desc = f"{concept} {s['disambiguator']} Example: \"{s.get('example', '')}\""
            else:
                desc = concept
            lines.append(f"{s['id']}: {desc}")
    return "\n".join(lines)

def call_llm(system: str, user: str, retries: int = 4) -> str:
    req_data = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ],
        "temperature": 0,
        "max_tokens": 150
    }
    
    if PROVIDER == 'openrouter':
        url = OPENROUTER_URL
        api_key = os.environ.get("OPENROUTER_API_KEY")
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {api_key}',
            'HTTP-Referer': 'https://github.com/google/antigravity',
            'X-Title': 'Antigravity Research Benchmark'
        }
    else:
        url = DEEPSEEK_URL
        api_key = DEEPSEEK_API_KEY
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {api_key}'
        }

    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(req_data).encode('utf-8'),
                headers=headers,
                method='POST'
            )
            with urllib.request.urlopen(req, timeout=15) as res:
                body = res.read().decode('utf-8')
                return json.loads(body)['choices'][0]['message']['content'].strip()
        except Exception as e:
            if attempt == retries - 1:
                return f"Error: {e}"
            import time
            time.sleep(1.5 * (attempt + 1))
    return "Error: Request failed after retries"

def parse_model_response(raw: str, library: List[Dict[str, Any]]) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    # 1. Extract JSON payload for execution first
    parsed_json = None
    clean_raw = raw
    if "```json" in clean_raw:
        clean_raw = clean_raw.split("```json", 1)[1]
    if "```" in clean_raw:
        clean_raw = clean_raw.split("```", 1)[0]
    clean_raw = clean_raw.strip()
    
    start = clean_raw.find("{")
    end = clean_raw.rfind("}")
    if start != -1 and end != -1:
        try:
            parsed_json = json.loads(clean_raw[start:end+1])
        except Exception:
            pass

    # 2. Identify which tool was chosen
    chosen_tool = None
    normalized_raw = raw.lower()
    
    # Try finding exact matches for tool IDs in the raw text
    for s in library:
        tid = s["id"]
        if tid in normalized_raw:
            chosen_tool = tid
            break
            
    # Fallback inference based on JSON keys if not found in text
    if chosen_tool is None and parsed_json is not None:
        keys = list(parsed_json.keys())
        if any(k in keys for k in ["fraction", "kelly_fraction", "amount", "kelly_amount"]):
            chosen_tool = "kelly_position_size"
        elif "stop_distance" in keys:
            chosen_tool = "atr_dynamic_stop"
        elif any(k in keys for k in ["ev", "expected_value", "net_ev", "gross_ev"]):
            chosen_tool = "trade_ev_calculator"
        elif any(k in keys for k in ["leakage_detected", "lookahead_detected", "lookahead_flag"]):
            chosen_tool = "lookahead_bias_audit"
        elif any(k in keys for k in ["mean", "std", "population_mean", "population_std"]):
            chosen_tool = "standard_scaler_apply"
        elif any(k in keys for k in ["data_drift_detected", "drift_detected"]):
            chosen_tool = "data_drift_monitor"

    return chosen_tool, parsed_json

def grade_execution(chosen_tool: Optional[str], target_id: str, parsed: Optional[Dict[str, Any]], ground_truth: Dict[str, Any]) -> Tuple[bool, bool, bool, bool]:
    # Selection success
    sel_ok = (chosen_tool == target_id)
    
    # Schema compliance
    schema_keys = [k for k in ground_truth.keys() if k != "contains_reason"]
    schema_ok = False
    if parsed is not None:
        schema_ok = all(k in parsed for k in schema_keys)
        
    # Execution correctness
    exec_ok = False
    if sel_ok and parsed is not None:
        exec_ok = True
        for k, gt_v in ground_truth.items():
            if k == "contains_reason":
                reason_val = parsed.get("reason", "").lower()
                if gt_v not in reason_val:
                    exec_ok = False
            else:
                p_v = parsed.get(k)
                if isinstance(gt_v, float) and isinstance(p_v, (int, float)):
                    if abs(p_v - gt_v) > 0.05:  # tolerance window
                        exec_ok = False
                elif p_v != gt_v:
                    exec_ok = False
                    
    # End-to-end success
    e2e_ok = sel_ok and exec_ok and schema_ok
    
    return sel_ok, exec_ok, schema_ok, e2e_ok

def run_experiment(dry_run: bool = False):
    print("=" * 75)
    print(f"PHASE 3 CORE EXECUTION BENCHMARK SWEEP (Model: {MODEL})")
    print("=" * 75)
    
    # Load fillers from skill_experiment
    try:
        fillers = json.load(open("skill_experiment/fillers_v4.json"))
        # Also load additional and hq fillers to expand selection pool
        try:
            fillers += json.load(open("skill_experiment/hq_fillers.json"))
            fillers += json.load(open("skill_experiment/additional_fillers.json"))
        except Exception:
            pass
    except Exception as e:
        print("Failed to load fillers from skill_experiment:", e)
        return
        
    # Remove duplicates
    seen_ids = set()
    unique_fillers = []
    for f in fillers:
        if f.get("id") and f["id"] not in seen_ids:
            seen_ids.add(f["id"])
            unique_fillers.append(f)
            
    print(f"Loaded {len(unique_fillers)} filler skills from database.")
    
    SIZES = [60, 200] if dry_run else [60, 200, 500, 1000]
    CONDITIONS = ["bare", "expert"]
    
    # Formulate evaluation tasks pool
    eval_tasks = []
    for target in TARGET_SKILLS:
        for q in target["queries"]:
            eval_tasks.append({
                "target_id": target["id"],
                "query": q["text"],
                "ground_truth": q["ground_truth"]
            })
            
    total_calls = len(SIZES) * len(CONDITIONS) * len(eval_tasks)
    print(f"Total scheduled queries: {total_calls} ({len(eval_tasks)} queries x {len(SIZES)} sizes x 2 conditions)")
    print("=" * 75)
    
    results = {}
    logs = []
    done = 0
    
    # Threading executor
    executor = ThreadPoolExecutor(max_workers=8)
    
    for size in SIZES:
        results[size] = {}
        library = build_library(size, unique_fillers, size * 17 + 11)
        print(f"\n── Size={size} (Total tools: {len(library)}) ──")
        
        for condition in CONDITIONS:
            futures = []
            
            # Prepare task list prompt
            listing = format_skill_list(library, condition)
            system_prompt = (
                "You are an intelligent multi-agent routing and execution component in an enterprise OS.\n"
                "Review the request. First, select the most appropriate skill ID from the list below.\n"
                "Second, execute the task details using the chosen skill logic and output the results.\n\n"
                "Response MUST be a JSON object containing the calculations, plus any output variables. "
                "The response MUST also explicitly print or include the chosen skill ID. "
                "Respond ONLY with valid JSON.\n\n"
                f"Available skills:\n{listing}"
            )
            
            for task in eval_tasks:
                user_prompt = task["query"]
                
                # Submit to pool
                future = executor.submit(call_llm, system_prompt, user_prompt)
                futures.append((task, future))
                
            # Await results
            sel_correct = 0
            exec_correct = 0
            schema_correct = 0
            e2e_correct = 0
            
            for task, future in futures:
                raw_response = future.result()
                chosen_tool, parsed_json = parse_model_response(raw_response, library)
                sel_ok, exec_ok, schema_ok, e2e_ok = grade_execution(
                    chosen_tool, task["target_id"], parsed_json, task["ground_truth"]
                )
                
                if sel_ok: sel_correct += 1
                if exec_ok: exec_correct += 1
                if schema_ok: schema_correct += 1
                if e2e_ok: e2e_correct += 1
                
                done += 1
                logs.append({
                    "size": size,
                    "condition": condition,
                    "target_id": task["target_id"],
                    "query": task["query"],
                    "raw_response": raw_response,
                    "chosen_tool": chosen_tool,
                    "parsed_json": parsed_json,
                    "selection_ok": sel_ok,
                    "execution_ok": exec_ok,
                    "schema_ok": schema_ok,
                    "e2e_ok": e2e_ok
                })
                
                pct = (done / total_calls) * 100
                icon = "✓" if e2e_ok else "✗"
                if done % 5 == 0 or done == total_calls:
                    print(f"  [{done}/{total_calls} {pct:.1f}%] {condition} {icon} | Query: \"{task['query'][:35]}...\" -> Selected: {chosen_tool} (Want: {task['target_id']})")

            results[size][condition] = {
                "total": len(eval_tasks),
                "selection_acc": (sel_correct / len(eval_tasks)) * 100,
                "execution_acc": (exec_correct / len(eval_tasks)) * 100,
                "schema_acc": (schema_correct / len(eval_tasks)) * 100,
                "e2e_acc": (e2e_correct / len(eval_tasks)) * 100
            }
            
            print(f"  → Result [{condition}]: Selection Acc: {results[size][condition]['selection_acc']:.1f}% | Execution Acc: {results[size][condition]['execution_acc']:.1f}% | E2E Success: {results[size][condition]['e2e_acc']:.1f}%")

    # Save results
    model_clean = MODEL.replace("/", "_").replace("-", "_")
    out_file = f"blackbox_os/roles/data_scientist/workflows/results_execution_sweep_{model_clean}.json"
    with open(out_file, "w") as f:
        json.dump({"results": results, "logs": logs}, f, indent=2)
        
    print("\n" + "=" * 75)
    print(f"Experiment completed. Detailed logs written to: {out_file}")
    print("=" * 75)
    
    # Print comparison table
    print(f"{'Size':<6} | {'Condition':<10} | {'Selection Acc':<15} | {'Execution Acc':<15} | {'E2E Success':<12}")
    print("-" * 75)
    for size in SIZES:
        for cond in CONDITIONS:
            res = results[size][cond]
            print(f"{size:<6} | {cond:<10} | {res['selection_acc']:<15.1f}% | {res['execution_acc']:<15.1f}% | {res['e2e_acc']:<12.1f}%")
    print("=" * 75)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--model", type=str, default="deepseek-chat")
    parser.add_argument("--provider", type=str, default="deepseek")
    args = parser.parse_args()
    
    MODEL = args.model
    PROVIDER = args.provider
    
    run_experiment(dry_run=args.dry_run)
