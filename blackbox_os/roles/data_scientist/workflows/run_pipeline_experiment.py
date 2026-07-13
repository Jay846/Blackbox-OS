import os
import sys
import json
import random
import csv
import subprocess
import tempfile
import time
from typing import Dict, Any, List, Optional, Tuple, TypedDict
from concurrent.futures import ThreadPoolExecutor
from langgraph.graph import StateGraph, END

# Add workspace to path
sys.path.append(os.getcwd())

MODEL = "openai/gpt-4o-mini"
PROVIDER = "openrouter"

TARGET_SKILLS = [
    {
        "id": "kelly_position_size",
        "bare_desc": "Compute Kelly fraction and dollar amount based on execution CSV files.",
        "expert_desc": (
            "Computes the optimal leverage and bet size to maximize logarithmic growth rate from execution history. "
            "Input: a CSV file containing columns 'symbol', 'side', 'price', 'amount', 'realized_pnl'. "
            "Formula: f = (win_rate * payoff - (1 - win_rate)) / payoff, where win_rate is count of positive pnl / total trades, "
            "and payoff is average positive pnl / average absolute negative pnl. Dollar amount = f * bankroll. "
            "Note: asset names in fills.csv have trading pairs format (e.g. 'BTC/USDT' or 'ETH/USDT'), so when filtering for 'BTC trades', check if 'symbol' starts with or contains 'BTC' (or equals 'BTC/USDT'). "
            "Output schema: {\"fraction\": float, \"amount\": float}. Do not confuse with kelly_fractional_sizing."
        )
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
        )
    },
    {
        "id": "standard_scaler_apply",
        "bare_desc": "Compute StandardScaler parameters on CSV values column.",
        "expert_desc": (
            "Computes population StandardScaler parameters (mean, std) on returns in a CSV file. "
            "Input: a CSV file containing column names clean_return and leaked_return. "
            "Formula: mean = sum(x)/N, std = sqrt(sum((x - mean)^2)/N). "
            "Output schema: {\"mean\": float, \"std\": float}. Do not confuse with sample standard deviation (ddof=1)."
        )
    }
]

# State schema for LangGraph
class PipelineState(TypedDict):
    run_index: int
    features_path: str
    returns_path: str
    fills_path: str
    condition: str
    library: List[Dict[str, Any]]
    
    # State values
    leakage_detected: Optional[bool]
    target_column: Optional[str]
    scaled_mean: Optional[float]
    scaled_std: Optional[float]
    kelly_fraction: Optional[float]
    kelly_amount: Optional[float]
    
    # Tracking
    step_history: List[str]
    error_logs: List[str]
    success: bool

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
            "Authorization": f"Bearer {api_key}",
            "HTTP-Referer": "https://github.com/google/antigravity"
        }
    else:
        return '{"error": "Unknown provider"}'
        
    import urllib.request
    
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
        "max_tokens": 1024
    }
    
    try:
        with urllib.request.urlopen(req, data=json.dumps(data).encode("utf-8"), timeout=15) as response:
            res_body = json.loads(response.read().decode("utf-8"))
            return res_body["choices"][0]["message"]["content"]
    except Exception as e:
        return f'{{"error": "{str(e)}"}}'

def execute_sandbox_code(code: str) -> Optional[Dict[str, Any]]:
    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
        f.write(code)
        temp_name = f.name
        
    try:
        res = subprocess.run(
            [sys.executable, temp_name],
            capture_output=True,
            text=True,
            timeout=5
        )
        stdout = res.stdout.strip()
        if os.path.exists(temp_name):
            os.remove(temp_name)
            
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
        if os.path.exists(temp_name):
            os.remove(temp_name)
        return None

def query_node_with_retry(system_prompt: str, user_prompt: str, condition: str, target_tool: str) -> Optional[Dict[str, Any]]:
    for attempt in range(3):
        raw_response = query_llm(system_prompt, user_prompt)
        parsed = None
        try:
            start = raw_response.find("{")
            end = raw_response.rfind("}")
            if start != -1 and end != -1:
                parsed = json.loads(raw_response[start:end+1])
        except Exception:
            try:
                import ast
                start = raw_response.find("{")
                end = raw_response.rfind("}")
                if start != -1 and end != -1:
                    parsed = ast.literal_eval(raw_response[start:end+1])
            except Exception:
                pass
                
        if parsed is not None:
            chosen = parsed.get("chosen_skill_id") or parsed.get("skill_id")
            if chosen == target_tool:
                if condition == "expert_sandbox":
                    code = parsed.get("python_code")
                    if code:
                        sandbox_res = execute_sandbox_code(code)
                        if sandbox_res is not None:
                            return sandbox_res
                else:
                    return parsed
        time.sleep(1)
    return None

# Node functions
def audit_node(state: PipelineState) -> PipelineState:
    listing = format_skill_list(state["library"], state["condition"])
    
    if state["condition"] == "expert_sandbox":
        system_prompt = (
            "You are an intelligent multi-agent routing and execution component.\n"
            "Review the request. First, select lookahead_bias_audit from the list.\n"
            "Next, write a complete self-contained python script to execute it.\n"
            "The python script must print its final result to stdout as a single JSON dictionary matching the required schema.\n"
            "Your script MUST print a JSON dictionary to stdout containing the exact key 'leakage_detected': boolean.\n"
            "CRITICAL: Evaluate ONLY the mathematical formula string logic for future information reference (e.g. check for 'target' or index offsets like 't+1'). Do NOT flag a column as leaking lookahead bias based on its name or label alone (e.g. even if a column name contains the word 'leak').\n\n"
            "Output Format:\n"
            "Return ONLY a JSON dictionary of the form:\n"
            "{\n"
            "  \"chosen_skill_id\": \"lookahead_bias_audit\",\n"
            "  \"python_code\": \"...\"\n"
            "}\n\n"
            "Available Skills:\n" + listing
        )
    else:
        system_prompt = (
            "You are an intelligent multi-agent routing component.\n"
            "Select lookahead_bias_audit, compute the results, and return ONLY a JSON response:\n"
            "{\n"
            "  \"chosen_skill_id\": \"lookahead_bias_audit\",\n"
            "  \"leakage_detected\": bool\n"
            "}\n\n"
            "Available Skills:\n" + listing
        )
        
    user_prompt = f"Target Query: Audit features file '{state['features_path']}' for lookahead bias. You must output the result with key 'leakage_detected'."
    res = query_node_with_retry(system_prompt, user_prompt, state["condition"], "lookahead_bias_audit")
    
    if res is not None:
        val = None
        for k in ["leakage_detected", "leakage", "has_leakage", "detected", "result"]:
            if k in res:
                val = res[k]
                break
        if val is None:
            for v in res.values():
                if isinstance(v, bool):
                    val = v
                    break
        state["leakage_detected"] = val
    else:
        state["leakage_detected"] = None
        state["error_logs"].append("Audit node failed to parse or run correctly.")
        
    state["step_history"].append("audit_node")
    return state

def remediation_node(state: PipelineState) -> PipelineState:
    if state["leakage_detected"] is True:
        state["target_column"] = "clean_return"
        state["step_history"].append("remediation_node_leakage_dropped")
    else:
        state["target_column"] = "leaked_return"
        state["step_history"].append("remediation_node_no_leakage")
    return state

def scale_node(state: PipelineState) -> PipelineState:
    target_column = state.get("target_column") or "leaked_return"
    listing = format_skill_list(state["library"], state["condition"])
    
    if state["condition"] == "expert_sandbox":
        system_prompt = (
            "You are an intelligent multi-agent routing and execution component.\n"
            "Review the request. First, select standard_scaler_apply from the list.\n"
            "Next, write a complete self-contained python script to execute it.\n"
            "The python script must print its final result to stdout as a single JSON dictionary matching the required schema.\n"
            "Your script MUST print a JSON dictionary to stdout containing the keys 'mean': float and 'std': float.\n"
            "CRITICAL: When computing standard deviation, you MUST compute population standard deviation (ddof=0 in pandas or numpy.std) rather than sample standard deviation.\n\n"
            "Output Format:\n"
            "Return ONLY a JSON dictionary of the form:\n"
            "{\n"
            "  \"chosen_skill_id\": \"standard_scaler_apply\",\n"
            "  \"python_code\": \"...\"\n"
            "}\n\n"
            "Available Skills:\n" + listing
        )
    else:
        system_prompt = (
            "You are an intelligent multi-agent routing component.\n"
            "Select standard_scaler_apply, compute the results, and return ONLY a JSON response:\n"
            "{\n"
            "  \"chosen_skill_id\": \"standard_scaler_apply\",\n"
            "  \"mean\": float,\n"
            "  \"std\": float\n"
            "}\n\n"
            "Available Skills:\n" + listing
        )
        
    user_prompt = f"Target Query: Calculate StandardScaler parameters (mean, std) for column '{target_column}' in return file '{state['returns_path']}'."
    res = query_node_with_retry(system_prompt, user_prompt, state["condition"], "standard_scaler_apply")
    
    if res is not None:
        mean_val = None
        for k in ["mean", "scaled_mean", "average", "avg"]:
            if k in res:
                mean_val = res[k]
                break
        std_val = None
        for k in ["std", "scaled_std", "std_dev", "standard_deviation"]:
            if k in res:
                std_val = res[k]
                break
        state["scaled_mean"] = mean_val
        state["scaled_std"] = std_val
    else:
        state["scaled_mean"] = None
        state["scaled_std"] = None
        state["error_logs"].append("Scale node failed to parse or run correctly.")
        
    state["step_history"].append("scale_node")
    return state

def kelly_node(state: PipelineState) -> PipelineState:
    listing = format_skill_list(state["library"], state["condition"])
    
    if state["condition"] == "expert_sandbox":
        system_prompt = (
            "You are an intelligent multi-agent routing and execution component.\n"
            "Review the request. First, select kelly_position_size from the list.\n"
            "Next, write a complete self-contained python script to execute it.\n"
            "The python script must print its final result to stdout as a single JSON dictionary matching the required schema.\n"
            "Your script MUST print a JSON dictionary to stdout containing the keys 'fraction': float and 'amount': float.\n\n"
            "Output Format:\n"
            "Return ONLY a JSON dictionary of the form:\n"
            "{\n"
            "  \"chosen_skill_id\": \"kelly_position_size\",\n"
            "  \"python_code\": \"...\"\n"
            "}\n\n"
            "Available Skills:\n" + listing
        )
    else:
        system_prompt = (
            "You are an intelligent multi-agent routing component.\n"
            "Select kelly_position_size, compute the results, and return ONLY a JSON response:\n"
            "{\n"
            "  \"chosen_skill_id\": \"kelly_position_size\",\n"
            "  \"fraction\": float,\n"
            "  \"amount\": float\n"
            "}\n\n"
            "Available Skills:\n" + listing
        )
        
    user_prompt = f"Target Query: Calculate win-rate and payoff for BTC trades in '{state['fills_path']}' and return Kelly fraction/amount for 50000 bankroll."
    res = query_node_with_retry(system_prompt, user_prompt, state["condition"], "kelly_position_size")
    
    if res is not None:
        frac_val = None
        for k in ["fraction", "kelly_fraction", "size", "leverage", "f"]:
            if k in res:
                frac_val = res[k]
                break
        amt_val = None
        for k in ["amount", "dollar_amount", "bet_size"]:
            if k in res:
                amt_val = res[k]
                break
        state["kelly_fraction"] = frac_val
        state["kelly_amount"] = amt_val
        if state["kelly_fraction"] is not None:
            state["success"] = True
    else:
        state["kelly_fraction"] = None
        state["kelly_amount"] = None
        state["error_logs"].append("Kelly node failed to parse or run correctly.")
        
    state["step_history"].append("kelly_node")
    return state


# Setup LangGraph WorkFlow
def build_pipeline_graph() -> StateGraph:
    workflow = StateGraph(PipelineState)
    workflow.add_node("audit", audit_node)
    workflow.add_node("remediation", remediation_node)
    workflow.add_node("scale", scale_node)
    workflow.add_node("kelly", kelly_node)
    
    workflow.set_entry_point("audit")
    workflow.add_edge("audit", "remediation")
    workflow.add_edge("remediation", "scale")
    workflow.add_edge("scale", "kelly")
    workflow.add_edge("kelly", END)
    
    return workflow.compile()

# Reference Math Engine
def calculate_ground_truth(run_idx: int) -> Dict[str, Any]:
    # 1. Audit
    leakage = (run_idx % 2 == 0)
    
    # 2. Scale (returns calculation)
    # Recreate deterministic returns series
    random.seed(run_idx)
    clean_rets = [random.normalvariate(30.0, 10.0) for _ in range(15)]
    leaked_rets = [random.normalvariate(40.0, 12.0) for _ in range(15)]
    
    target_col = "clean_return" if leakage else "leaked_return"
    selected_rets = clean_rets if leakage else leaked_rets
    
    mean = sum(selected_rets) / len(selected_rets)
    variance = sum((x - mean) ** 2 for x in selected_rets) / len(selected_rets)
    std = variance ** 0.5
    
    # 3. Kelly calculations
    pnls = [150.0 + run_idx*10, 180.0 - run_idx*5, -100.0 + run_idx*2, 120.0 + run_idx*4, -80.0 - run_idx*3, 
            210.0 + run_idx*8, -120.0 - run_idx*4, 160.0 + run_idx*3, -60.0 + run_idx*2, 140.0 - run_idx*6]
    
    wins = [p for p in pnls if p > 0]
    losses = [abs(p) for p in pnls if p < 0]
    
    win_rate = len(wins) / len(pnls)
    avg_win = sum(wins) / len(wins) if wins else 0
    avg_loss = sum(losses) / len(losses) if losses else 0
    
    payoff = avg_win / avg_loss if avg_loss else 0
    fraction = (win_rate * payoff - (1 - win_rate)) / payoff if payoff else 0
    if fraction <= 0:
        fraction = 0.0
        amount = 0.0
    else:
        amount = fraction * 50000
        
    return {
        "leakage_detected": leakage,
        "target_column": target_col,
        "mean": mean,
        "std": std,
        "fraction": fraction,
        "amount": amount
    }

def generate_datasets(base_dir="blackbox_os/roles/data_scientist/workflows/mock_data/pipeline_runs"):
    os.makedirs(base_dir, exist_ok=True)
    
    for i in range(10):
        run_dir = os.path.join(base_dir, f"run_{i}")
        os.makedirs(run_dir, exist_ok=True)
        
        # 1. Features
        has_leakage = (i % 2 == 0)
        features = [
            {"column_name": "clean_return", "formula": "price_t / price_t-1 - 1"},
            {"column_name": "leaked_return", "formula": "target * 2.0 - 0.5" if has_leakage else "lagged_price_1 - lagged_price_2"}
        ]
        with open(os.path.join(run_dir, "features.csv"), "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=features[0].keys())
            writer.writeheader()
            writer.writerows(features)
            
        # 2. Returns
        random.seed(i)
        clean_rets = [random.normalvariate(30.0, 10.0) for _ in range(15)]
        leaked_rets = [random.normalvariate(40.0, 12.0) for _ in range(15)]
        
        with open(os.path.join(run_dir, "returns.csv"), "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["clean_return", "leaked_return"])
            for c, l in zip(clean_rets, leaked_rets):
                writer.writerow([c, l])
                
        # 3. Fills
        pnls = [150.0 + i*10, 180.0 - i*5, -100.0 + i*2, 120.0 + i*4, -80.0 - i*3, 
                210.0 + i*8, -120.0 - i*4, 160.0 + i*3, -60.0 + i*2, 140.0 - i*6]
        
        fills = []
        for idx, pnl in enumerate(pnls):
            fills.append({
                "timestamp": f"2026-07-10T12:{idx:02d}:00Z",
                "symbol": "BTC/USDT",
                "side": "buy" if pnl > 0 else "sell",
                "price": 95000.0,
                "amount": 0.5,
                "realized_pnl": pnl
            })
        with open(os.path.join(run_dir, "fills.csv"), "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fills[0].keys())
            writer.writeheader()
            writer.writerows(fills)

def run_experiment(dry_run: bool = False):
    print("=" * 80)
    print(f"LANGGRAPH MULTI-NODE PIPELINE SWEEP (Model: {MODEL}, Provider: {PROVIDER})")
    print("=" * 80)
    
    # 1. Load Fillers
    try:
        fillers = json.load(open("skill_experiment/fillers_v4.json"))
        try:
            fillers += json.load(open("skill_experiment/hq_fillers.json"))
            fillers += json.load(open("skill_experiment/additional_fillers.json"))
        except Exception:
            pass
    except Exception as e:
        print("Failed to load filler skills:", e)
        return
        
    seen_ids = set()
    unique_fillers = []
    for f in fillers:
        if f.get("id") and f["id"] not in seen_ids:
            seen_ids.add(f["id"])
            unique_fillers.append(f)
            
    print(f"Loaded {len(unique_fillers)} background fillers.")
    
    # Generate datasets
    generate_datasets()
    print("Successfully generated 10 pipeline dataset runs under mock_data/pipeline_runs/")
    
    # 2. Build Graph app
    app = build_pipeline_graph()
    
    SIZES = [60] if dry_run else [60, 500]
    CONDITIONS = ["expert", "expert_sandbox"]
    
    results = {}
    logs = []
    
    for size in SIZES:
        results[size] = {}
        library = build_library(size, unique_fillers, size * 29 + 13)
        print(f"\n── Catalog Size={size} (Total tools: {len(library)}) ──")
        
        for condition in CONDITIONS:
            print(f"Running Condition: {condition}...")
            
            run_successes = 0
            start_time = time.time()
            
            for i in range(10):
                # Calculate reference values
                ref = calculate_ground_truth(i)
                
                # Initialize state
                state: PipelineState = {
                    "run_index": i,
                    "features_path": f"blackbox_os/roles/data_scientist/workflows/mock_data/pipeline_runs/run_{i}/features.csv",
                    "returns_path": f"blackbox_os/roles/data_scientist/workflows/mock_data/pipeline_runs/run_{i}/returns.csv",
                    "fills_path": f"blackbox_os/roles/data_scientist/workflows/mock_data/pipeline_runs/run_{i}/fills.csv",
                    "condition": condition,
                    "library": library,
                    "leakage_detected": None,
                    "target_column": None,
                    "scaled_mean": None,
                    "scaled_std": None,
                    "kelly_fraction": None,
                    "kelly_amount": None,
                    "step_history": [],
                    "error_logs": [],
                    "success": False
                }
                
                # Execute graph
                try:
                    final_state = app.invoke(state)
                except Exception as e:
                    final_state = state
                    final_state["error_logs"].append(f"Graph execution failed: {str(e)}")
                
                # Verify outputs mathematically
                e2e_ok = True
                
                # 1. Audit check
                if final_state.get("leakage_detected") != ref["leakage_detected"]:
                    e2e_ok = False
                # 2. Target column check
                if final_state.get("target_column") != ref["target_column"]:
                    e2e_ok = False
                # 3. Scale mean/std check
                m_v = final_state.get("scaled_mean")
                s_v = final_state.get("scaled_std")
                if m_v is None or s_v is None:
                    e2e_ok = False
                else:
                    if abs(m_v - ref["mean"]) > 0.05 or abs(s_v - ref["std"]) > 0.05:
                        e2e_ok = False
                # 4. Kelly positioning check
                k_f = final_state.get("kelly_fraction")
                k_a = final_state.get("kelly_amount")
                if k_f is None or k_a is None:
                    e2e_ok = False
                else:
                    if abs(k_f - ref["fraction"]) > 0.05 or abs(k_a - ref["amount"]) > 1.0:
                        e2e_ok = False
                        
                if e2e_ok:
                    run_successes += 1
                    
                status_icon = "✓" if e2e_ok else "✗"
                print(f"  [Run {i}/10] {status_icon} | History: {final_state['step_history']} | Target Col: {final_state.get('target_column')} | Kelly Fraction: {final_state.get('kelly_fraction')}")
                
                logs.append({
                    "size": size,
                    "condition": condition,
                    "run_index": i,
                    "state": {
                        "leakage_detected": final_state.get("leakage_detected"),
                        "target_column": final_state.get("target_column"),
                        "scaled_mean": final_state.get("scaled_mean"),
                        "scaled_std": final_state.get("scaled_std"),
                        "kelly_fraction": final_state.get("kelly_fraction"),
                        "kelly_amount": final_state.get("kelly_amount")
                    },
                    "ground_truth": ref,
                    "history": final_state["step_history"],
                    "errors": final_state["error_logs"],
                    "e2e_ok": e2e_ok
                })
                
            elapsed = time.time() - start_time
            results[size][condition] = {
                "success_count": run_successes,
                "success_rate": (run_successes / 10.0) * 100,
                "avg_time": elapsed / 10.0
            }
            print(f"  → Result [{condition}]: Success Rate: {results[size][condition]['success_rate']:.1f}% | Avg Node Execution Time: {results[size][condition]['avg_time']:.2f}s")
            
    # Save outcomes
    model_clean = MODEL.replace("/", "_").replace("-", "_")
    out_file = f"blackbox_os/roles/data_scientist/workflows/results_pipeline_sweep_{model_clean}.json"
    with open(out_file, "w") as f:
        json.dump({"results": results, "logs": logs}, f, indent=2)
        
    print("\n" + "=" * 80)
    print(f"Pipeline Sweep Completed. Logs written to: {out_file}")
    print("=" * 80)
    for size in SIZES:
        for cond in CONDITIONS:
            res = results[size][cond]
            print(f"{size:<6} | {cond:<15} | Success Rate: {res['success_rate']:.1f}% | Time/Run: {res['avg_time']:.2f}s")
    print("=" * 80)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--model", type=str, default="openai/gpt-4o-mini")
    parser.add_argument("--provider", type=str, default="openrouter")
    args = parser.parse_args()
    
    MODEL = args.model
    PROVIDER = args.provider
    
    run_experiment(dry_run=args.dry_run)
