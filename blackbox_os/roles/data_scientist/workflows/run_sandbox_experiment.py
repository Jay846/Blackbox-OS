import json
import urllib.request
import math
import random
import os
import sys
import subprocess
import tempfile
from typing import Any, Dict, List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

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
        ],
        "noisy_queries": [
            {
                "text": "can you calclate standard kelly postion size fraction and amnt if win_rate is 0.55, profit_loss ratio is 1.8, and my balance is 10000?",
                "ground_truth": {"fraction": 0.30, "amount": 3000.0}
            },
            {
                "text": "Calculate my Kelly leverage bet size... win probability=0.6, payoff ratio=1.5, total cash=50000. I need to make sure I don't overleverage. typos: kely fraction.",
                "ground_truth": {"fraction": 0.333333, "amount": 16666.67}
            },
            {
                "text": "kely_position_size sizing request: bankroll is 100000, edge is 5%, win rate 0.52, payout is 2.0. what is the kelly amount?",
                "ground_truth": {"fraction": 0.28, "amount": 28000.0}
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
        ],
        "noisy_queries": [
            {
                "text": "Hey! My trader friend suggested using ATR for my trailing stop. My current ATR is 15.2 and I want to set a multiplier of 2.0. Can you compute the stop distance? Pls ignore the noise.",
                "ground_truth": {"stop_distance": 30.4}
            },
            {
                "text": "compute atr stop distance: multiplier is 3.0, atr_value=2.5. i need this for risk management.",
                "ground_truth": {"stop_distance": 7.5}
            },
            {
                "text": "Calculate Stop Distance: atr is 5.0, mult is 1.5. typos: atr_dynamic_stop distance.",
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
        ],
        "noisy_queries": [
            {
                "text": "Calculate expected value (EV) for a trade with win probability 0.6, winning return of 200.0, losing return of -100.0, and commission fee of 5.0. Make sure to deduct commission.",
                "ground_truth": {"ev": 75.0}
            },
            {
                "text": "trade ev calculation: win rate 0.4, target reward 300, risk loss is -50, fee is 10. compute the net ev.",
                "ground_truth": {"ev": 80.0}
            },
            {
                "text": "Calculate expected value of a trade with 0.5 win probability, $100 win return, -$50 loss return, fee is $5. what is the trade_ev_calculator net ev?",
                "ground_truth": {"ev": 20.0}
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
        ],
        "noisy_queries": [
            {
                "text": "I was reviewing my feature engineering pipeline and found a column total_revenue_offset defined as target * 5 + 10. Could you check if there is any lookhaed leakage bias in this?",
                "ground_truth": {"leakage_detected": True, "contains_reason": "target"}
            },
            {
                "text": "check for data leakage in column 'lagged_price_2' defined as 'price_{t-2}'. is there any lookahead bias?",
                "ground_truth": {"leakage_detected": False}
            },
            {
                "text": "lookahead check on column 'future_return' calculated as 'price_{t+1} / price_t - 1'. leakage?",
                "ground_truth": {"leakage_detected": True}
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
        ],
        "noisy_queries": [
            {
                "text": "Apply standard scale transformation like sklearn on this list of numbers: [12, 17, 33, 44, 51]. Give me the pop mean and std dev.",
                "ground_truth": {"mean": 31.4, "std": 15.027973915}
            },
            {
                "text": "apply standard scaler on [10, 20, 30, 40, 50]. calculate mean and population std.",
                "ground_truth": {"mean": 30.0, "std": 14.1421356}
            },
            {
                "text": "StandardScaler apply for [5, 10, 15, 20, 25]. What is the mean and population standard deviation?",
                "ground_truth": {"mean": 15.0, "std": 7.0710678}
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
        ],
        "noisy_queries": [
            {
                "text": "Evaluate data drft warning. KS score is 0.35 and the warning threshld is 0.25. Has it drifted?",
                "ground_truth": {"data_drift_detected": True}
            },
            {
                "text": "data drift check: ks score is 0.12, threshold is 0.20. did it drift?",
                "ground_truth": {"data_drift_detected": False}
            },
            {
                "text": "check for data drift ks_score=0.40, threshold=0.30. drift status?",
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

def call_llm(system: str, user: str, retries: int = 4) -> str:
    req_data = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ],
        "temperature": 0,
        "max_tokens": 300  # Larger max tokens to accommodate python code block
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
            timeout=2.0
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

def parse_model_response(raw: str, library: List[Dict[str, Any]]) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
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

    chosen_tool = None
    normalized_raw = raw.lower()
    for s in library:
        tid = s["id"]
        if tid in normalized_raw:
            chosen_tool = tid
            break
            
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
    sel_ok = (chosen_tool == target_id)
    
    schema_keys = [k for k in ground_truth.keys() if k != "contains_reason"]
    schema_ok = False
    if parsed is not None:
        schema_ok = all(k in parsed for k in schema_keys)
        
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
                    if abs(p_v - gt_v) > 0.05:
                        exec_ok = False
                elif p_v != gt_v:
                    exec_ok = False
                    
    e2e_ok = sel_ok and exec_ok and schema_ok
    return sel_ok, exec_ok, schema_ok, e2e_ok

def run_experiment(dry_run: bool = False, noise: bool = False):
    print("=" * 75)
    print(f"SANDBOX DELEGATION SWEEP (Model: {MODEL}, Noise: {noise})")
    print("=" * 75)
    
    # Load fillers from skill_experiment
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
        q_list = target["noisy_queries"] if noise else target["queries"]
        for q in q_list:
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
                    "Second, write a self-contained Python script to compute the outputs. "
                    "The response MUST be a JSON object containing the chosen skill ID under 'skill_id', and a key 'python_code' containing the Python script.\n"
                    "The Python script must print the final calculated keys as a JSON dictionary matching the output schema.\n"
                    "Respond ONLY with valid JSON.\n\n"
                    f"Available skills:\n{listing}"
                )
            else:
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
                future = executor.submit(call_llm, system_prompt, user_prompt)
                futures.append((task, future))
                
            sel_correct = 0
            exec_correct = 0
            schema_correct = 0
            e2e_correct = 0
            
            for task, future in futures:
                raw_response = future.result()
                chosen_tool, parsed_json = parse_model_response(raw_response, library)
                
                # Execute Python sandbox if applicable
                sandbox_run_info = None
                if condition == "expert_sandbox" and parsed_json is not None and "python_code" in parsed_json:
                    code_to_run = parsed_json["python_code"]
                    sandbox_result = execute_sandbox_code(code_to_run)
                    if sandbox_result is not None:
                        sandbox_run_info = {
                            "code": code_to_run,
                            "captured_output": sandbox_result
                        }
                        # Merge sandbox computed values back into parsed_json
                        for k, v in sandbox_result.items():
                            parsed_json[k] = v
                
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
                    "sandbox_run_info": sandbox_run_info,
                    "selection_ok": sel_ok,
                    "execution_ok": exec_ok,
                    "schema_ok": schema_ok,
                    "e2e_ok": e2e_ok
                })
                
                pct = (done / total_calls) * 100
                icon = "✓" if e2e_ok else "✗"
                if done % 5 == 0 or done == total_calls:
                    print(f"  [{done}/{total_calls} {pct:.1f}%] {condition} {icon} | Query: \"{task['query'][:35]}...\" -> Selected: {chosen_tool}")
                    
            results[size][condition] = {
                "total": len(eval_tasks),
                "selection_acc": (sel_correct / len(eval_tasks)) * 100,
                "execution_acc": (exec_correct / len(eval_tasks)) * 100,
                "schema_acc": (schema_correct / len(eval_tasks)) * 100,
                "e2e_acc": (e2e_correct / len(eval_tasks)) * 100
            }
            print(f"  → Result [{condition}]: Selection Acc: {results[size][condition]['selection_acc']:.1f}% | Execution Acc: {results[size][condition]['execution_acc']:.1f}% | E2E Success: {results[size][condition]['e2e_acc']:.1f}%")

    model_clean = MODEL.replace("/", "_").replace("-", "_")
    prefix = "results_sandbox_expanded_noise_sweep" if noise else "results_sandbox_sweep"
    out_file = f"blackbox_os/roles/data_scientist/workflows/{prefix}_{model_clean}.json"
    with open(out_file, "w") as f:
        json.dump({"results": results, "logs": logs}, f, indent=2)
        
    print("\n" + "=" * 75)
    print(f"Experiment completed. Detailed logs written to: {out_file}")
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
    parser.add_argument("--noise", action="store_true")
    args = parser.parse_args()
    
    MODEL = args.model
    PROVIDER = args.provider
    
    run_experiment(dry_run=args.dry_run, noise=args.noise)
