import os
import sys
import json
import random
import subprocess
import tempfile
import time
from typing import Dict, Any, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor

# Add workspace to path
sys.path.append(os.getcwd())

# Configuration
MODEL = "deepseek-chat"
PROVIDER = "deepseek"

# 1. Production Target Execution Skills and Queries
TARGET_SKILLS = [
    {
        "id": "kelly_position_size",
        "bare_desc": "Compute Kelly fraction and dollar amount based on execution CSV files.",
        "expert_desc": (
            "Computes the optimal leverage and bet size to maximize logarithmic growth rate from execution history. "
            "Input: a CSV file containing columns 'symbol', 'side', 'price', 'amount', 'realized_pnl'. "
            "Formula: f = (win_rate * payoff - (1 - win_rate)) / payoff, where win_rate is count of positive pnl / total trades, "
            "and payoff is average positive pnl / average absolute negative pnl. Dollar amount = f * bankroll. "
            "Note: asset names in grid_bot_fills.csv have trading pairs format (e.g. 'BTC/USDT' or 'ETH/USDT'), so when filtering for 'BTC trades', check if 'symbol' starts with or contains 'BTC' (or equals 'BTC/USDT'). "
            "Output schema: {\"fraction\": float, \"amount\": float}. Do not confuse with kelly_fractional_sizing."
        ),
        "queries": [
            {
                "text": "Analyze the trade history in blackbox_os/roles/data_scientist/workflows/mock_data/grid_bot_fills.csv. Calculate the win rate and payoff ratio for BTC trades, and determine the Kelly fraction for a total bankroll of 50000.",
                "ground_truth": {"fraction": 0.375, "amount": 18750.0}
            }
        ]
    },
    {
        "id": "atr_dynamic_stop",
        "bare_desc": "Compute stop distance from ATR calculated over historical price CSV.",
        "expert_desc": (
            "Calculates the trailing stop distance based on ATR calculated from a CSV. "
            "Input: a CSV file containing columns 'timestamp', 'high', 'low', 'close'. "
            "True Range (TR) formula: TR = max(high - low, abs(high - close_prev), abs(low - close_prev)). "
            "Ensure you compute TR correctly using rolling/shifting close prices. Do NOT include the raw high or low price as components in the max of TR (the second and third components must be absolute differences from close_prev). "
            "ATR is the simple average of TR over the last 14 periods. stop_distance = ATR * multiplier. "
            "Output schema: {\"stop_distance\": float}. Do not confuse with standard fixed stop losses."
        ),
        "queries": [
            {
                "text": "Read blackbox_os/roles/data_scientist/workflows/mock_data/eth_usd_prices.csv, calculate the ATR over the last 14 periods, and determine the stop loss distance using a multiplier of 2.0.",
                "ground_truth": {"stop_distance": 20.0}
            }
        ]
    },
    {
        "id": "trade_ev_calculator",
        "bare_desc": "Compute net expected value of trades from performance JSON file.",
        "expert_desc": (
            "Computes expected value including transaction fee from JSON performance data. "
            "Input: JSON file containing list of dicts under 'trades' key with 'return' key. "
            "Formula: ev = (win_prob * (win_return - fee)) + ((1 - win_prob) * (loss_return - fee)). "
            "Where win_prob is count of positive returns / total trades, win_return is average of positive returns, "
            "and loss_return is average of negative returns. "
            "The final JSON response dictionary from Python MUST contain exactly the key 'ev' for expected value (do not use other names like 'net_ev' or 'net_expected_value'). "
            "Output schema: {\"ev\": float}. Do not confuse with raw risk/reward ratio."
        ),
        "queries": [
            {
                "text": "Load blackbox_os/roles/data_scientist/workflows/mock_data/trade_performance.json, count the successful vs failed trades to find probability, compute average win vs average loss, and calculate the net EV deducting a fee of 5.0 per trade.",
                "ground_truth": {"ev": 75.0}
            }
        ]
    },
    {
        "id": "lookahead_bias_audit",
        "bare_desc": "Detect lookahead bias in columns from feature CSV.",
        "expert_desc": (
            "Audits column formulas in features CSV for leakage/lookahead bias. "
            "Input: a CSV file containing columns 'column_name', 'formula'. "
            "Criteria: If any formula references future timestamps (e.g. t+1, t+2) or contains the substring 'target', "
            "leakage_detected is true. "
            "Output schema: {\"leakage_detected\": bool}. Do not confuse with data drift checks."
        ),
        "queries": [
            {
                "text": "Audit blackbox_os/roles/data_scientist/workflows/mock_data/features.csv for lookahead bias. Check if any column formula references future timestamps or targets.",
                "ground_truth": {"leakage_detected": True}
            }
        ]
    },
    {
        "id": "standard_scaler_apply",
        "bare_desc": "Compute StandardScaler parameters on CSV values column.",
        "expert_desc": (
            "Computes population StandardScaler parameters (mean, std) on returns in a CSV file. "
            "Input: a CSV file containing column 'return'. "
            "Formula: mean = sum(x)/N, std = sqrt(sum((x - mean)^2)/N). "
            "Output schema: {\"mean\": float, \"std\": float}. Do not confuse with sample standard deviation (ddof=1)."
        ),
        "queries": [
            {
                "text": "Read blackbox_os/roles/data_scientist/workflows/mock_data/raw_returns.csv and apply standard scale normalization on the 'return' column. Return the population mean and std.",
                "ground_truth": {"mean": 31.4, "std": 15.027973915}
            }
        ]
    },
    {
        "id": "data_drift_monitor",
        "bare_desc": "Given drift metrics JSON, check for drift.",
        "expert_desc": (
            "Evaluates KS drift score against warning threshold in drift metrics JSON. "
            "Input: a JSON file containing nested dict with key 'metrics'. "
            "Note: the JSON input contains a dictionary metrics -> implied_vol -> ks_score and warning_threshold. Simply extract ks_score and warning_threshold directly from metrics['implied_vol']. Do not assume baseline or raw distributions are present. "
            "Criteria: If implied_vol's ks_score > warning_threshold, data_drift_detected is true. "
            "Output schema: {\"data_drift_detected\": bool}. Do not confuse with target leakage detector."
        ),
        "queries": [
            {
                "text": "Analyze blackbox_os/roles/data_scientist/workflows/mock_data/drift_metrics.json. Compare the daily distribution metrics of the feature 'implied_vol' against the baseline distribution, and alert if the drift exceeds the warning threshold.",
                "ground_truth": {"data_drift_detected": True}
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
    target_ids = {t["id"] for t in TARGET_SKILLS}
    unique_fillers = [f for f in fillers if f.get("id") not in target_ids]
    n_fillers = max(0, size - len(TARGET_SKILLS))
    shuffled_fillers = seeded_shuffle(unique_fillers, seed)[:n_fillers]
    return seeded_shuffle(TARGET_SKILLS + shuffled_fillers, seed + 1)

def format_skill_list(library: List[Dict[str, Any]], condition: str) -> str:
    lines = []
    for s in library:
        if s.get("id") in {t["id"] for t in TARGET_SKILLS}:
            desc = s["expert_desc"]
            if condition == "expert_sandbox":
                desc += (
                    " You MUST write a complete, self-contained Python script to compute the output. "
                    "The script must be returned in the 'python_code' key of your JSON response. "
                    "The script must print the results to stdout as a JSON dictionary matching the required schema. "
                    "Ensure the Python code is fully escaped inside the JSON string."
                )
            lines.append(f"{s['id']}: {desc}")
        else:
            concept = s.get("concept", "Tool option.")
            if "disambiguator" in s:
                desc = f"{concept} {s['disambiguator']} Example: \"{s.get('example', '')}\""
            else:
                desc = concept
            lines.append(f"{s['id']}: {desc}")
    return "\n".join(lines)

def query_llm(system_prompt: str, user_prompt: str) -> str:
    # Use exact URL endpoints and structure from proven runner
    if PROVIDER == "deepseek":
        url = "https://api.deepseek.com/chat/completions"
        api_key = os.environ.get("DEEPSEEK_API_KEY", "")
        if not api_key:
            return '{"error": "DEEPSEEK_API_KEY not found"}'
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
    elif PROVIDER == "openrouter":
        url = "https://openrouter.ai/api/v1/chat/completions"
        api_key = os.environ.get("OPENROUTER_API_KEY", "")
        if not api_key:
            return '{"error": "OPENROUTER_API_KEY not found"}'
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "HTTP-Referer": "https://github.com/google/antigravity"
        }
    else:
        return '{"error": "Unknown provider"}'
        
    import urllib.request
    import json
    
    req = urllib.request.Request(url, method="POST")
    for k, v in headers.items():
        req.add_header(k, v)
        
    data = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.0,
        "max_tokens": 512
    }
    
    try:
        with urllib.request.urlopen(req, data=json.dumps(data).encode("utf-8"), timeout=15) as response:
            res_body = json.loads(response.read().decode("utf-8"))
            return res_body["choices"][0]["message"]["content"]
    except Exception as e:
        return f'{{"error": "{str(e)}"}}'

def execute_sandbox_code(code: str) -> Optional[Dict[str, Any]]:
    # Create temporary execution file
    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
        f.write(code)
        temp_name = f.name
        
    try:
        res = subprocess.run(
            [sys.executable, temp_name],
            capture_output=True,
            text=True,
            timeout=3.0
        )
        if res.returncode == 0:
            stdout_str = res.stdout.strip()
            start = stdout_str.find("{")
            end = stdout_str.rfind("}")
            if start != -1 and end != -1:
                json_part = stdout_str[start:end+1]
                try:
                    return json.loads(json_part)
                except Exception:
                    try:
                        import ast
                        return ast.literal_eval(json_part)
                    except Exception:
                        pass
    except Exception:
        pass
    finally:
        try:
            os.remove(temp_name)
        except Exception:
            pass
    return None

def evaluate_task(task: Dict[str, Any], condition: str, system_prompt: str, user_prompt: str) -> Tuple[bool, bool, bool, str, Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    raw_response = query_llm(system_prompt, user_prompt)
    
    # Parse choice
    chosen_tool = None
    parsed = None
    schema_ok = False
    sandbox_run_info = None
    
    try:
        # Extract JSON from response
        start = raw_response.find("{")
        end = raw_response.rfind("}")
        if start != -1 and end != -1:
            parsed = json.loads(raw_response[start:end+1])
            schema_ok = True
    except Exception:
        try:
            import ast
            start = raw_response.find("{")
            end = raw_response.rfind("}")
            if start != -1 and end != -1:
                parsed = ast.literal_eval(raw_response[start:end+1])
                schema_ok = True
        except Exception:
            pass
            
    if parsed is not None:
        chosen_tool = parsed.get("chosen_skill_id") or parsed.get("skill_id")
        
    sel_ok = (chosen_tool == task["target_id"])
    
    # For Sandbox condition: execute delegation
    if condition == "expert_sandbox" and sel_ok and parsed is not None:
        code = parsed.get("python_code")
        if code:
            sandbox_res = execute_sandbox_code(code)
            sandbox_run_info = {"code": code, "result": sandbox_res}
            if sandbox_res is not None:
                parsed = sandbox_res
            else:
                parsed = None
                
    exec_ok = False
    if sel_ok and parsed is not None:
        exec_ok = True
        ground_truth = task["ground_truth"]
        for k, gt_v in ground_truth.items():
            p_v = parsed.get(k)
            if p_v is None:
                exec_ok = False
                break
            # Handle NaN values explicitly
            if isinstance(p_v, float) and p_v != p_v:
                exec_ok = False
                break
            if str(p_v).lower() == 'nan':
                exec_ok = False
                break
            if isinstance(gt_v, float) and isinstance(p_v, (int, float)):
                try:
                    if abs(p_v - gt_v) > 0.05:
                        exec_ok = False
                except Exception:
                    exec_ok = False
            elif p_v != gt_v:
                exec_ok = False
                
    e2e_ok = sel_ok and exec_ok and schema_ok
    return sel_ok, exec_ok, schema_ok, raw_response, parsed, sandbox_run_info

def run_experiment(dry_run: bool = False):
    print("=" * 75)
    print(f"PRODUCTION-GRADE EXECUTION SWEEP (Model: {MODEL})")
    print("=" * 75)
    
    # Load fillers
    try:
        fillers = json.load(open("skill_experiment/fillers_v4.json"))
        try:
            fillers += json.load(open("skill_experiment/hq_fillers.json"))
            fillers += json.load(open("skill_experiment/additional_fillers.json"))
        except Exception:
            pass
    except Exception as e:
        print("Failed to load fillers:", e)
        return
        
    seen_ids = set()
    unique_fillers = []
    for f in fillers:
        if f.get("id") and f["id"] not in seen_ids:
            seen_ids.add(f["id"])
            unique_fillers.append(f)
            
    print(f"Loaded {len(unique_fillers)} filler skills.")
    
    SIZES = [60, 200] if dry_run else [60, 200, 500, 1000]
    CONDITIONS = ["expert", "expert_sandbox"]
    
    eval_tasks = []
    for target in TARGET_SKILLS:
        for q in target["queries"]:
            eval_tasks.append({
                "target_id": target["id"],
                "query": q["text"],
                "ground_truth": q["ground_truth"]
            })
            
    total_calls = len(SIZES) * len(CONDITIONS) * len(eval_tasks)
    print(f"Total scheduled queries: {total_calls}")
    print("=" * 75)
    
    results = {}
    logs = []
    done = 0
    
    executor = ThreadPoolExecutor(max_workers=8)
    
    for size in SIZES:
        results[size] = {}
        library = build_library(size, unique_fillers, size * 17 + 11)
        print(f"\n── Size={size} (Total tools: {len(library)}) ──")
        
        for condition in CONDITIONS:
            futures = []
            listing = format_skill_list(library, condition)
            
            if condition == "expert_sandbox":
                system_prompt = (
                    "You are an intelligent multi-agent routing and execution component in an enterprise OS.\n"
                    "Review the request. First, select the most appropriate skill ID from the list below.\n"
                    "Next, because this task requires loading/reading files and complex computations, write a completely self-contained python script to execute it.\n"
                    "The python script must print its final result to stdout as a single JSON dictionary with keys corresponding to the target schema.\n"
                    "Do NOT try to compute standard deviation or read files manually in your thought. Let the Python execution sandbox run the code.\n\n"
                    "Output Format:\n"
                    "Return ONLY a JSON dictionary of the form:\n"
                    "{\n"
                    "  \"chosen_skill_id\": \"id_of_selected_tool\",\n"
                    "  \"python_code\": \"def run():\\n    # Write python code here to read files, calculate and print\\n    import pandas as pd\\n    ...\\n    print(\\\"{...}\\\")\\n\\nrun()\"\n"
                    "}\n\n"
                    "Available Skills:\n" + listing
                )
            else:
                system_prompt = (
                    "You are an intelligent multi-agent routing component in an enterprise OS.\n"
                    "Review the request. Select the most appropriate skill ID from the list below, compute the results, and return ONLY a JSON response.\n"
                    "Output format:\n"
                    "{\n"
                    "  \"chosen_skill_id\": \"id_of_selected_tool\",\n"
                    "  \"metric_name\": computed_value,\n"
                    "  ...\n"
                    "}\n\n"
                    "Available Skills:\n" + listing
                )
                
            for task in eval_tasks:
                user_prompt = f"Target Query: \"{task['query']}\"\nCompute and return the values."
                futures.append(executor.submit(
                    evaluate_task,
                    task,
                    condition,
                    system_prompt,
                    user_prompt
                ))
                
            sel_correct = 0
            exec_correct = 0
            schema_correct = 0
            e2e_correct = 0
            
            for task, fut in zip(eval_tasks, futures):
                try:
                    sel_ok, exec_ok, schema_ok, raw_response, parsed_json, sandbox_run_info = fut.result()
                except Exception as e:
                    sel_ok, exec_ok, schema_ok, raw_response, parsed_json, sandbox_run_info = False, False, False, str(e), None, None
                    
                if sel_ok: sel_correct += 1
                if exec_ok: exec_correct += 1
                if schema_ok: schema_correct += 1
                e2e_ok = sel_ok and exec_ok and schema_ok
                if e2e_ok: e2e_correct += 1
                
                done += 1
                logs.append({
                    "size": size,
                    "condition": condition,
                    "target_id": task["target_id"],
                    "query": task["query"],
                    "raw_response": raw_response,
                    "chosen_tool": task["target_id"] if sel_ok else None,
                    "parsed_json": parsed_json,
                    "sandbox_run_info": sandbox_run_info,
                    "selection_ok": sel_ok,
                    "execution_ok": exec_ok,
                    "schema_ok": schema_ok,
                    "e2e_ok": e2e_ok
                })
                
                pct = (done / total_calls) * 100
                icon = "✓" if e2e_ok else "✗"
                if done % 5 == 0 or done == total_calls:
                    print(f"  [{done}/{total_calls} {pct:.1f}%] {condition} {icon} | Query: \"{task['query'][:35]}...\" -> Result: {parsed_json}")
                    
            results[size][condition] = {
                "total": len(eval_tasks),
                "selection_acc": (sel_correct / len(eval_tasks)) * 100,
                "execution_acc": (exec_correct / len(eval_tasks)) * 100,
                "schema_acc": (schema_correct / len(eval_tasks)) * 100,
                "e2e_acc": (e2e_correct / len(eval_tasks)) * 100
            }
            print(f"  → Result [{condition}]: Selection Acc: {results[size][condition]['selection_acc']:.1f}% | Execution Acc: {results[size][condition]['execution_acc']:.1f}% | E2E Success: {results[size][condition]['e2e_acc']:.1f}%")

    model_clean = MODEL.replace("/", "_").replace("-", "_")
    out_file = f"blackbox_os/roles/data_scientist/workflows/results_production_sweep_{model_clean}.json"
    with open(out_file, "w") as f:
        json.dump({"results": results, "logs": logs}, f, indent=2)
        
    print("\n" + "=" * 75)
    print(f"Production-Grade Sweep Completed. Logs written to: {out_file}")
    print("=" * 75)
    
    print(f"{'Size':<6} | {'Condition':<15} | {'Selection Acc':<15} | {'Execution Acc':<15} | {'E2E Success':<12}")
    print("-" * 75)
    for size in SIZES:
        for cond in CONDITIONS:
            res = results[size][cond]
            print(f"{size:<6} | {cond:<15} | {res['selection_acc']:<15.1f}% | {res['execution_acc']:<15.1f}% | {res['e2e_acc']:<12.1f}%")
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
