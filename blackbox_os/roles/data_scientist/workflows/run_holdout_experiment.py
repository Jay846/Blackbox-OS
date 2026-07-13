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

# 1. Hold-Out Target Execution Skills and Unseen Queries (5 per skill = 30 total)
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
                "text": "Check the trades inside blackbox_os/roles/data_scientist/workflows/mock_data/grid_bot_fills.csv. Filter only BTC/USDT fills, calculate the percentage of winning trades and positive/negative payoff ratio, then compute Kelly bet size for 50000 total capital.",
                "ground_truth": {"fraction": 0.375, "amount": 18750.0}
            },
            {
                "text": "Determine the Kelly positioning size for a total portfolio value of 50000 using the fill logs in blackbox_os/roles/data_scientist/workflows/mock_data/grid_bot_fills.csv. We only care about BTC trades.",
                "ground_truth": {"fraction": 0.375, "amount": 18750.0}
            },
            {
                "text": "Extract the trading fills for BTC from the log grid_bot_fills.csv in mock_data. Calculate the win rate and the average win over absolute average loss, then return the Kelly sizing fraction and total amount for a 50000 bankroll.",
                "ground_truth": {"fraction": 0.375, "amount": 18750.0}
            },
            {
                "text": "Using blackbox_os/roles/data_scientist/workflows/mock_data/grid_bot_fills.csv, find out the BTC trading performance. What is the win rate and payoff ratio? What would the Kelly allocation fraction and capital size be for a bankroll of 50000?",
                "ground_truth": {"fraction": 0.375, "amount": 18750.0}
            },
            {
                "text": "Calculate the optimal Kelly fraction and investment amount for a bankroll of 50000 using trade execution records in blackbox_os/roles/data_scientist/workflows/mock_data/grid_bot_fills.csv. Restrict analysis to BTC.",
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
                "text": "Using the price feed in blackbox_os/roles/data_scientist/workflows/mock_data/eth_usd_prices.csv, calculate the Average True Range (ATR) over 14 rows, and set a trailing stop distance with a factor of 2.0.",
                "ground_truth": {"stop_distance": 20.0}
            },
            {
                "text": "Determine the stop distance (factor 2.0) from the ATR of the last 14 candlesticks in blackbox_os/roles/data_scientist/workflows/mock_data/eth_usd_prices.csv.",
                "ground_truth": {"stop_distance": 20.0}
            },
            {
                "text": "Compute the ATR (14 periods) for eth_usd_prices.csv under mock_data. Using a multiplier of 2.0, what is the trailing stop loss distance?",
                "ground_truth": {"stop_distance": 20.0}
            },
            {
                "text": "Given the historical price dataset in blackbox_os/roles/data_scientist/workflows/mock_data/eth_usd_prices.csv, calculate the ATR value over a rolling window of 14 points and multiply it by 2.0 to get the stop distance.",
                "ground_truth": {"stop_distance": 20.0}
            },
            {
                "text": "Find the latest 14 prices in blackbox_os/roles/data_scientist/workflows/mock_data/eth_usd_prices.csv, compute the average true range (ATR), and scale it by 2.0 to calculate the dynamic stop loss distance.",
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
                "text": "Load the JSON performance file trade_performance.json from blackbox_os/roles/data_scientist/workflows/mock_data/. Calculate the expected value (EV) of the strategy, factoring in a 5.0 fee per trade.",
                "ground_truth": {"ev": 75.0}
            },
            {
                "text": "Using trade_performance.json in mock_data, find the win-rate, average win size, and average loss size. Compute the net EV subtracting a commission fee of 5.0.",
                "ground_truth": {"ev": 75.0}
            },
            {
                "text": "Extract trade list from blackbox_os/roles/data_scientist/workflows/mock_data/trade_performance.json. Compute the expected payoff (EV) of a trade after deducting a flat fee of 5.0.",
                "ground_truth": {"ev": 75.0}
            },
            {
                "text": "Find the trade performance metrics in mock_data/trade_performance.json. Count wins and losses to get probabilities, average them, and return the net EV after a 5.0 fee per trade.",
                "ground_truth": {"ev": 75.0}
            },
            {
                "text": "Calculate the net expected value (EV) for trades listed in blackbox_os/roles/data_scientist/workflows/mock_data/trade_performance.json with a flat 5.0 per-trade cost.",
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
                "text": "Look through the features CSV file blackbox_os/roles/data_scientist/workflows/mock_data/features.csv and scan for lookahead bias. Return true if there is any column leaking future data.",
                "ground_truth": {"leakage_detected": True}
            },
            {
                "text": "Audit mock_data/features.csv for lookahead bias. Scan the formulas for references to future index values (t+1, etc.) or targets.",
                "ground_truth": {"leakage_detected": True}
            },
            {
                "text": "Inspect the formulas in blackbox_os/roles/data_scientist/workflows/mock_data/features.csv to see if they reference future information or containing 'target', which would indicate data leakage.",
                "ground_truth": {"leakage_detected": True}
            },
            {
                "text": "Determine if any feature formula in blackbox_os/roles/data_scientist/workflows/mock_data/features.csv suffers from data leakage (references future offsets or targets).",
                "ground_truth": {"leakage_detected": True}
            },
            {
                "text": "Check for target leakage or lookahead bias in blackbox_os/roles/data_scientist/workflows/mock_data/features.csv. Look for formulas referencing future periods or 'target'.",
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
                "text": "Read the raw returns dataset in blackbox_os/roles/data_scientist/workflows/mock_data/raw_returns.csv and run a population standard scaling normalization on the 'return' column. Output the population mean and std.",
                "ground_truth": {"mean": 31.4, "std": 15.027973915}
            },
            {
                "text": "Calculate the population mean and population standard deviation (ddof=0) for standard scaling the 'return' column in mock_data/raw_returns.csv.",
                "ground_truth": {"mean": 31.4, "std": 15.027973915}
            },
            {
                "text": "Compute the z-score normalization parameters (mean and population std) for the returns listed in blackbox_os/roles/data_scientist/workflows/mock_data/raw_returns.csv.",
                "ground_truth": {"mean": 31.4, "std": 15.027973915}
            },
            {
                "text": "Determine the population mean and standard deviation of the returns column in blackbox_os/roles/data_scientist/workflows/mock_data/raw_returns.csv.",
                "ground_truth": {"mean": 31.4, "std": 15.027973915}
            },
            {
                "text": "Retrieve the returns from blackbox_os/roles/data_scientist/workflows/mock_data/raw_returns.csv, apply standard scaling, and return the population mean and standard deviation.",
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
                "text": "Load the data drift statistics file blackbox_os/roles/data_scientist/workflows/mock_data/drift_metrics.json. Compare the implied_vol's KS metric against its warning threshold to see if drift is detected.",
                "ground_truth": {"data_drift_detected": True}
            },
            {
                "text": "Audit drift_metrics.json in mock_data for feature drift on implied_vol. Check if the KS test score exceeds the warning threshold.",
                "ground_truth": {"data_drift_detected": True}
            },
            {
                "text": "Read the JSON drift metrics in blackbox_os/roles/data_scientist/workflows/mock_data/drift_metrics.json, look up implied_vol drift status, and determine if it exceeds the alert threshold.",
                "ground_truth": {"data_drift_detected": True}
            },
            {
                "text": "Using blackbox_os/roles/data_scientist/workflows/mock_data/drift_metrics.json, check if the Kolmogorov-Smirnov drift score for implied_vol is greater than its warning threshold.",
                "ground_truth": {"data_drift_detected": True}
            },
            {
                "text": "Determine if there is data drift for implied_vol using the KS score and threshold in blackbox_os/roles/data_scientist/workflows/mock_data/drift_metrics.json.",
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
            "Authorization": f"Bearer {api_key}"
        }
        
    data = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.0,
        "max_tokens": 800
    }
    
    import urllib.request
    req = urllib.request.Request(url, data=json.dumps(data).encode("utf-8"), headers=headers)
    try:
        with urllib.request.urlopen(req) as response:
            res_body = response.read().decode("utf-8")
            res_json = json.loads(res_body)
            return res_json["choices"][0]["message"]["content"]
    except Exception as e:
        return json.dumps({"error": str(e)})

def execute_sandbox_code(code: str) -> Optional[Dict[str, Any]]:
    # Create temp directory
    with tempfile.TemporaryDirectory() as tmpdir:
        # Recreate mock_data folder links inside temp directory if needed
        # But we can just execute python inside cwd
        temp_file = os.path.join(os.getcwd(), f"temp_sandbox_{random.randint(0, 1000000)}.py")
        with open(temp_file, "w") as f:
            f.write(code)
            
        try:
            res = subprocess.run(
                [sys.executable, temp_file],
                capture_output=True,
                text=True,
                timeout=5
            )
            stdout = res.stdout.strip()
            # Clean up temp file
            if os.path.exists(temp_file):
                os.remove(temp_file)
                
            # Attempt to parse json/dict from stdout
            start = stdout.find("{")
            end = stdout.rfind("}")
            if start != -1 and end != -1:
                json_part = stdout[start:end+1]
                try:
                    return json.loads(json_part)
                except Exception:
                    try:
                        import ast
                        return ast.literal_eval(json_part)
                    except Exception:
                        pass
            return None
        except Exception:
            if os.path.exists(temp_file):
                os.remove(temp_file)
            return None

def evaluate_task(task: Dict[str, Any], condition: str, system_prompt: str, user_prompt: str) -> Tuple[bool, bool, bool, str, Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    raw_response = query_llm(system_prompt, user_prompt)
    
    parsed = None
    schema_ok = False
    chosen_tool = None
    
    if raw_response:
        try:
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
    sandbox_run_info = None
    
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
    print(f"GENERALIZATION SWEEP ON UNSEEN HOLDOUT SET (Model: {MODEL})")
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
    
    # Run sizes 60 and 500 for generalization test
    SIZES = [60] if dry_run else [60, 500]
    CONDITIONS = ["expert", "expert_sandbox"]
    
    eval_tasks = []
    for target in TARGET_SKILLS:
        for idx, query_obj in enumerate(target["queries"]):
            eval_tasks.append({
                "target_id": target["id"],
                "query": query_obj["text"],
                "ground_truth": query_obj["ground_truth"],
                "query_idx": idx
            })
            
    total_calls = len(SIZES) * len(CONDITIONS) * len(eval_tasks)
    print(f"Total scheduled holdout queries: {total_calls}")
    print("=" * 75)
    
    results = {}
    logs = []
    done = 0
    
    for size in SIZES:
        results[size] = {}
        library = build_library(size, unique_fillers, seed=9999)
        
        for condition in CONDITIONS:
            listing = format_skill_list(library, condition)
            futures = []
            executor = ThreadPoolExecutor(max_workers=8)
            
            if condition == "expert":
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
    out_file = f"blackbox_os/roles/data_scientist/workflows/results_holdout_sweep_{model_clean}.json"
    with open(out_file, "w") as f:
        json.dump({"results": results, "logs": logs}, f, indent=2)
        
    print("\n" + "=" * 75)
    print(f"Generalization Hold-Out Sweep Completed. Logs written to: {out_file}")
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
