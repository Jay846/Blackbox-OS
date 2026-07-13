import os
import sys
import json
import random
import csv
import subprocess
import tempfile
import time
import urllib.request
from typing import Dict, Any, List, Optional, Tuple, TypedDict
from langgraph.graph import StateGraph, END

# Add workspace to path
sys.path.append(os.getcwd())

MODEL = "deepseek-chat"
PROVIDER = "deepseek"

# Target Skills Definition
TARGET_SKILLS = [
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
    },
    {
        "id": "pca_apply",
        "bare_desc": "Apply PCA and calculate explained variance ratio of top 2 components.",
        "expert_desc": (
            "Applies Principal Component Analysis (PCA) on features to reduce dimensionality. "
            "Input: CSV file containing multiple features (feat_1 to feat_15). "
            "Process: Fit PCA from sklearn.decomposition on the specified feature columns with n_components=2. "
            "Output the explained variance ratio of the first two principal components. "
            "Output schema: {\"explained_variance_ratio\": [float, float]}."
        )
    },
    {
        "id": "linear_regression_fit",
        "bare_desc": "Fit linear regression and return coefficients, intercept, and R-squared.",
        "expert_desc": (
            "Fits a linear regression model to predict target returns from features. "
            "Input: CSV file containing target column (e.g. clean_return) and feature columns (feat_1, feat_2). "
            "Process: Fit LinearRegression from sklearn.linear_model with X as feat_1 and feat_2, and y as target. "
            "Output coefficients, intercept, and R-squared score. "
            "Output schema: {\"coefficients\": [float, float], \"intercept\": float, \"r_squared\": float}."
        )
    },
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
    }
]

# State schema for LangGraph
class PipelineState(TypedDict):
    run_index: int
    features_path: str
    returns_path: str
    fills_path: str
    user_goal: str
    library: List[Dict[str, Any]]
    
    # State values
    leakage_detected: Optional[bool]
    target_column: Optional[str]
    chosen_branch: Optional[str]  # 'scale', 'pca', 'regression', 'kelly'
    
    # Results
    final_output: Optional[Dict[str, Any]]
    
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

def format_skill_list(library: List[Dict[str, Any]], is_routing: bool = False) -> str:
    lines = []
    for s in library:
        if s.get("id") in {t["id"] for t in TARGET_SKILLS}:
            desc = s["expert_desc"]
            if not is_routing:
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
    url = "https://api.deepseek.com/chat/completions"
    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        return '{"error": "DEEPSEEK_API_KEY not found"}'
        
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
        "max_tokens": 1024
    }
    
    try:
        req = urllib.request.Request(url, method="POST")
        for k, v in headers.items():
            req.add_header(k, v)
        req_data = json.dumps(data).encode("utf-8")
        with urllib.request.urlopen(req, data=req_data, timeout=60) as response:
            res = json.loads(response.read().decode("utf-8"))
            return res["choices"][0]["message"]["content"]
    except Exception as e:
        return json.dumps({"error": str(e)})

def execute_sandbox_code(python_code: str) -> Optional[Dict[str, Any]]:
    with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w") as f:
        f.write(python_code)
        temp_name = f.name
        
    try:
        res = subprocess.run([sys.executable, temp_name], capture_output=True, text=True, timeout=15)
        stdout = res.stdout.strip()
        
        # Try finding JSON block in stdout
        start = stdout.find("{")
        end = stdout.rfind("}")
        if start != -1 and end != -1:
            json_str = stdout[start:end+1]
            try:
                return json.loads(json_str)
            except Exception:
                import ast
                try:
                    return ast.literal_eval(json_str)
                except Exception:
                    pass
        return None
    except Exception:
        return None
    finally:
        if os.path.exists(temp_name):
            os.remove(temp_name)

def clean_json_response(raw: str) -> Optional[Dict[str, Any]]:
    raw = raw.strip()
    if raw.startswith("```"):
        first_newline = raw.find("\n")
        last_fence = raw.rfind("```")
        if first_newline != -1 and last_fence != -1:
            raw = raw[first_newline:last_fence].strip()
    try:
        return json.loads(raw)
    except Exception:
        import ast
        try:
            return ast.literal_eval(raw)
        except Exception:
            return None

def query_node_with_retry(system_prompt: str, user_prompt: str, target_tool: str) -> Optional[Dict[str, Any]]:
    for attempt in range(3):
        raw_response = query_llm(system_prompt, user_prompt)
        parsed = clean_json_response(raw_response)
        
        if parsed is not None:
            chosen = parsed.get("chosen_skill_id") or parsed.get("skill_id")
            if chosen == target_tool:
                code = parsed.get("python_code")
                if code:
                    sandbox_res = execute_sandbox_code(code)
                    if sandbox_res is not None:
                        return sandbox_res
        time.sleep(1.0)
    return None

# LangGraph Node Implementations
def audit_node(state: PipelineState) -> PipelineState:
    listing = format_skill_list(state["library"])
    
    system_prompt = (
        "You are an intelligent multi-agent routing and execution component.\n"
        "Review the request. First, select lookahead_bias_audit from the list.\n"
        "Next, write a complete self-contained python script to execute it.\n"
        "The python script must print its final result to stdout as a single JSON dictionary matching the required schema.\n"
        "Your script MUST print a JSON dictionary to stdout containing the exact key 'leakage_detected': boolean.\n"
        "CRITICAL: Evaluate ONLY the mathematical formula string logic for future information reference (e.g. check for 'target' or index offsets like 't+1'). Do NOT flag a column as leaking lookahead bias based on its name or label alone.\n\n"
        "Output Format:\n"
        "Return ONLY a JSON dictionary of the form:\n"
        "{\n"
        "  \"chosen_skill_id\": \"lookahead_bias_audit\",\n"
        "  \"python_code\": \"...\"\n"
        "}\n\n"
        "Available Skills:\n" + listing
    )
        
    user_prompt = f"Target Query: Audit features file '{state['features_path']}' for lookahead bias. You must output the result with key 'leakage_detected'."
    res = query_node_with_retry(system_prompt, user_prompt, "lookahead_bias_audit")
    
    if res is not None:
        val = res.get("leakage_detected")
        if val is None:
            for k in ["leakage", "has_leakage", "detected", "result"]:
                if k in res:
                    val = res[k]
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

def decision_node(state: PipelineState) -> PipelineState:
    # Filter out already executed lookahead_bias_audit tool to prevent LLM from selecting it again
    filtered_library = [s for s in state["library"] if s.get("id") != "lookahead_bias_audit"]
    listing = format_skill_list(filtered_library, is_routing=True)
    
    system_prompt = (
        "You are an intelligent multi-agent routing component.\n"
        "Analyze the user's high-level goal and determine which of the available skills is needed next.\n"
        "Choose exactly one of the following next actions: 'standard_scaler_apply', 'pca_apply', 'linear_regression_fit', or 'kelly_position_size'.\n"
        "Note: The leakage audit and data remediation steps have ALREADY been successfully completed. Select the next downstream analytical action to fulfill the remaining portion of the goal.\n\n"
        "Output Format:\n"
        "Return ONLY a JSON dictionary of the form:\n"
        "{\n"
        "  \"chosen_skill_id\": \"id_of_selected_skill\"\n"
        "}\n\n"
        "Available Skills:\n" + listing
    )
    
    user_prompt = (
        f"User High-level Goal: {state['user_goal']}\n"
        f"Completed Steps: [lookahead_bias_audit, remediation_node]\n"
        f"Cleaned column target: {state['target_column']}\n"
        f"Select the NEXT downstream skill required to satisfy the goal."
    )
    
    for attempt in range(3):
        raw = query_llm(system_prompt, user_prompt)
        parsed = clean_json_response(raw)
        if parsed:
            chosen = parsed.get("chosen_skill_id") or parsed.get("skill_id")
            if chosen in ["standard_scaler_apply", "pca_apply", "linear_regression_fit", "kelly_position_size"]:
                if chosen == "standard_scaler_apply":
                    state["chosen_branch"] = "scale"
                elif chosen == "pca_apply":
                    state["chosen_branch"] = "pca"
                elif chosen == "linear_regression_fit":
                    state["chosen_branch"] = "regression"
                elif chosen == "kelly_position_size":
                    state["chosen_branch"] = "kelly"
                break
            else:
                print(f"[Decision Node Warning] LLM returned non-target skill: {chosen}")
        else:
            print(f"[Decision Node Warning] Failed to parse JSON from: {raw}")
        time.sleep(1.0)
        
    if not state.get("chosen_branch"):
        state["chosen_branch"] = "scale"  # fallback
        state["error_logs"].append("Decision node failed to select a valid branch. Falling back to scale.")
        
    state["step_history"].append(f"decision_node_routed_to_{state['chosen_branch']}")
    return state

def scale_node(state: PipelineState) -> PipelineState:
    target_column = state.get("target_column") or "leaked_return"
    listing = format_skill_list(state["library"])
    
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
    
    user_prompt = f"Target Query: Calculate StandardScaler parameters (mean, std) for column '{target_column}' in return file '{state['returns_path']}'."
    res = query_node_with_retry(system_prompt, user_prompt, "standard_scaler_apply")
    if res:
        state["final_output"] = res
        state["success"] = True
    else:
        state["error_logs"].append("Scale node execution failed.")
    state["step_history"].append("scale_node")
    return state

def pca_node(state: PipelineState) -> PipelineState:
    listing = format_skill_list(state["library"])
    
    system_prompt = (
        "You are an intelligent multi-agent routing and execution component.\n"
        "Review the request. First, select pca_apply from the list.\n"
        "Next, write a complete self-contained python script to execute it.\n"
        "The python script must print its final result to stdout as a single JSON dictionary matching the required schema.\n"
        "Your script MUST print a JSON dictionary to stdout containing the key 'explained_variance_ratio': list of 2 floats.\n\n"
        "Output Format:\n"
        "Return ONLY a JSON dictionary of the form:\n"
        "{\n"
        "  \"chosen_skill_id\": \"pca_apply\",\n"
        "  \"python_code\": \"...\"\n"
        "}\n\n"
        "Available Skills:\n" + listing
    )
    
    user_prompt = f"Target Query: Apply PCA with n_components=2 to features (feat_1 to feat_15) in '{state['returns_path']}' and return the explained variance ratio."
    res = query_node_with_retry(system_prompt, user_prompt, "pca_apply")
    if res:
        state["final_output"] = res
        state["success"] = True
    else:
        state["error_logs"].append("PCA node execution failed.")
    state["step_history"].append("pca_node")
    return state

def regression_node(state: PipelineState) -> PipelineState:
    target_column = state.get("target_column") or "leaked_return"
    listing = format_skill_list(state["library"])
    
    system_prompt = (
        "You are an intelligent multi-agent routing and execution component.\n"
        "Review the request. First, select linear_regression_fit from the list.\n"
        "Next, write a complete self-contained python script to execute it.\n"
        "The python script must print its final result to stdout as a single JSON dictionary matching the required schema.\n"
        "Your script MUST print a JSON dictionary to stdout containing the keys 'coefficients': list of 2 floats, 'intercept': float, and 'r_squared': float.\n\n"
        "Output Format:\n"
        "Return ONLY a JSON dictionary of the form:\n"
        "{\n"
        "  \"chosen_skill_id\": \"linear_regression_fit\",\n"
        "  \"python_code\": \"...\"\n"
        "}\n\n"
        "Available Skills:\n" + listing
    )
    
    user_prompt = f"Target Query: Fit a linear regression using features 'feat_1' and 'feat_2' to predict target column '{target_column}' in file '{state['returns_path']}'. Return coefficients, intercept, and r_squared."
    res = query_node_with_retry(system_prompt, user_prompt, "linear_regression_fit")
    if res:
        state["final_output"] = res
        state["success"] = True
    else:
        state["error_logs"].append("Regression node execution failed.")
    state["step_history"].append("regression_node")
    return state

def kelly_node(state: PipelineState) -> PipelineState:
    listing = format_skill_list(state["library"])
    
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
    
    user_prompt = f"Target Query: Calculate win-rate and payoff for BTC trades in '{state['fills_path']}' and return Kelly fraction/amount for 50000 bankroll."
    res = query_node_with_retry(system_prompt, user_prompt, "kelly_position_size")
    if res:
        state["final_output"] = res
        state["success"] = True
    else:
        state["error_logs"].append("Kelly node execution failed.")
    state["step_history"].append("kelly_node")
    return state

def route_decision(state: PipelineState) -> str:
    branch = state.get("chosen_branch")
    if branch == "scale":
        return "scale"
    elif branch == "pca":
        return "pca"
    elif branch == "regression":
        return "regression"
    elif branch == "kelly":
        return "kelly"
    return "scale"

# Build LangGraph workflow graph
def build_graph() -> StateGraph:
    workflow = StateGraph(PipelineState)
    
    workflow.add_node("audit", audit_node)
    workflow.add_node("remediation", remediation_node)
    workflow.add_node("decision", decision_node)
    workflow.add_node("scale", scale_node)
    workflow.add_node("pca", pca_node)
    workflow.add_node("regression", regression_node)
    workflow.add_node("kelly", kelly_node)
    
    workflow.set_entry_point("audit")
    workflow.add_edge("audit", "remediation")
    workflow.add_edge("remediation", "decision")
    
    # Conditional edge routing from decision_node to target branch
    workflow.add_conditional_edges(
        "decision",
        route_decision,
        {
            "scale": "scale",
            "pca": "pca",
            "regression": "regression",
            "kelly": "kelly"
        }
    )
    
    workflow.add_edge("scale", END)
    workflow.add_edge("pca", END)
    workflow.add_edge("regression", END)
    workflow.add_edge("kelly", END)
    
    return workflow.compile()

# Deterministic Reference Math Engine
def calculate_ground_truth(run_idx: int) -> Dict[str, Any]:
    # 1. Audit
    leakage = (run_idx % 2 == 0)
    target_col = "clean_return" if leakage else "leaked_return"
    
    random.seed(run_idx)
    clean_rets = [random.normalvariate(30.0, 10.0) for _ in range(15)]
    leaked_rets = [random.normalvariate(40.0, 12.0) for _ in range(15)]
    
    # 15 features columns
    features_cols = {}
    for f_idx in range(1, 16):
        features_cols[f"feat_{f_idx}"] = [random.normalvariate(0.0, 1.0) for _ in range(15)]
        
    df_data = {"clean_return": clean_rets, "leaked_return": leaked_rets}
    df_data.update(features_cols)
    
    import pandas as pd
    import numpy as np
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LinearRegression
    
    df = pd.DataFrame(df_data)
    
    # Target values based on routing choice
    # Runs 0-2: Scale
    # Runs 3-4: PCA
    # Runs 5-7: Regression
    # Runs 8-9: Kelly
    
    if run_idx in [0, 1, 2]:
        chosen_branch = "scale"
        selected_rets = clean_rets if leakage else leaked_rets
        mean = sum(selected_rets) / len(selected_rets)
        std = (sum((x - mean) ** 2 for x in selected_rets) / len(selected_rets)) ** 0.5
        expected = {"mean": mean, "std": std}
    elif run_idx in [3, 4]:
        chosen_branch = "pca"
        cols = [f"feat_{idx}" for idx in range(1, 16)]
        pca = PCA(n_components=2)
        pca.fit(df[cols])
        expected = {"explained_variance_ratio": pca.explained_variance_ratio_.tolist()}
    elif run_idx in [5, 6, 7]:
        chosen_branch = "regression"
        X = df[["feat_1", "feat_2"]]
        y = df[target_col]
        reg = LinearRegression().fit(X, y)
        expected = {
            "coefficients": reg.coef_.tolist(),
            "intercept": float(reg.intercept_),
            "r_squared": float(reg.score(X, y))
        }
    else:
        chosen_branch = "kelly"
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
            fraction, amount = 0.0, 0.0
        else:
            amount = fraction * 50000
        expected = {"fraction": fraction, "amount": amount}
        
    return {
        "leakage_detected": leakage,
        "target_column": target_col,
        "chosen_branch": chosen_branch,
        "expected": expected
    }

def generate_datasets(base_dir="blackbox_os/roles/data_scientist/workflows/mock_data/branching_runs"):
    os.makedirs(base_dir, exist_ok=True)
    
    for i in range(10):
        run_dir = os.path.join(base_dir, f"run_{i}")
        os.makedirs(run_dir, exist_ok=True)
        
        # 1. Features Formula csv
        has_leakage = (i % 2 == 0)
        features = [
            {"column_name": "clean_return", "formula": "price_t / price_t-1 - 1"},
            {"column_name": "leaked_return", "formula": "target * 2.0 - 0.5" if has_leakage else "lagged_price_1 - lagged_price_2"}
        ]
        with open(os.path.join(run_dir, "features.csv"), "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=features[0].keys())
            writer.writeheader()
            writer.writerows(features)
            
        # 2. Returns csv (containing targets and 15 feature columns)
        random.seed(i)
        clean_rets = [random.normalvariate(30.0, 10.0) for _ in range(15)]
        leaked_rets = [random.normalvariate(40.0, 12.0) for _ in range(15)]
        
        feature_cols = {}
        for f_idx in range(1, 16):
            feature_cols[f"feat_{f_idx}"] = [random.normalvariate(0.0, 1.0) for _ in range(15)]
            
        header = ["clean_return", "leaked_return"] + [f"feat_{idx}" for idx in range(1, 16)]
        with open(os.path.join(run_dir, "returns.csv"), "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for idx in range(15):
                row = [clean_rets[idx], leaked_rets[idx]] + [feature_cols[f"feat_{f_idx}"][idx] for f_idx in range(1, 16)]
                writer.writerow(row)
                
        # 3. Fills csv
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

def run_experiment():
    print("=" * 80)
    print("MULTI-STEP BRANCHING AUTOMATION SWEEP (DeepSeek V4, N=500)")
    print("=" * 80)
    
    generate_datasets()
    print("Mock datasets successfully generated.")
    
    # Load background fillers
    try:
        fillers = json.load(open("skill_experiment/fillers_v4.json"))
        try:
            fillers += json.load(open("skill_experiment/hq_fillers.json"))
            fillers += json.load(open("skill_experiment/additional_fillers.json"))
        except Exception:
            pass
    except Exception as e:
        print("Failed to load background tools:", e)
        return
        
    seen_ids = set()
    unique_fillers = []
    for f in fillers:
        if f.get("id") and f["id"] not in seen_ids:
            seen_ids.add(f["id"])
            unique_fillers.append(f)
            
    print(f"Loaded {len(unique_fillers)} background fillers.")
    
    user_goals = {
        0: "First audit features.csv for formula leakage and remediate it. Once clean, normalize the return features to zero mean and unit variance using StandardScaler.",
        1: "First audit features.csv for formula leakage and remediate it. Once clean, normalize the return features to zero mean and unit variance using StandardScaler.",
        2: "First audit features.csv for formula leakage and remediate it. Once clean, normalize the return features to zero mean and unit variance using StandardScaler.",
        3: "Audit column formulas for lookahead bias. Then, perform a Principal Component Analysis (PCA) on features feat_1 to feat_15 of the clean dataset to compute the explained variance ratio of the first 2 principal components.",
        4: "Audit column formulas for lookahead bias. Then, perform a Principal Component Analysis (PCA) on features feat_1 to feat_15 of the clean dataset to compute the explained variance ratio of the first 2 principal components.",
        5: "Audit the formulas for lookahead leakage. Once remediated, fit a linear regression using features feat_1 and feat_2 to predict the clean target column return. Return coefficients, intercept, and R-squared.",
        6: "Audit the formulas for lookahead leakage. Once remediated, fit a linear regression using features feat_1 and feat_2 to predict the clean target column return. Return coefficients, intercept, and R-squared.",
        7: "Audit the formulas for lookahead leakage. Once remediated, fit a linear regression using features feat_1 and feat_2 to predict the clean target column return. Return coefficients, intercept, and R-squared.",
        8: "Perform leakage auditing and drop any contaminated feature columns. Then, use the fills execution log to calculate the optimal Kelly sizing fraction and amount for a $50k bankroll.",
        9: "Perform leakage auditing and drop any contaminated feature columns. Then, use the fills execution log to calculate the optimal Kelly sizing fraction and amount for a $50k bankroll."
    }
    
    app = build_graph()
    
    path_successes = 0
    execution_successes = 0
    e2e_successes = 0
    
    results = []
    
    for i in range(10):
        print(f"\n── Run {i}/10 ──", flush=True)
        ref = calculate_ground_truth(i)
        
        # Build library
        library = build_library(500, unique_fillers, seed=(i * 10))
        
        initial_state: PipelineState = {
            "run_index": i,
            "features_path": f"blackbox_os/roles/data_scientist/workflows/mock_data/branching_runs/run_{i}/features.csv",
            "returns_path": f"blackbox_os/roles/data_scientist/workflows/mock_data/branching_runs/run_{i}/returns.csv",
            "fills_path": f"blackbox_os/roles/data_scientist/workflows/mock_data/branching_runs/run_{i}/fills.csv",
            "user_goal": user_goals[i],
            "library": library,
            
            "leakage_detected": None,
            "target_column": None,
            "chosen_branch": None,
            "final_output": None,
            
            "step_history": [],
            "error_logs": [],
            "success": False
        }
        
        # Execute workflow
        final_state = app.invoke(initial_state)
        
        # 1. Path Selection check
        path_correct = (final_state.get("chosen_branch") == ref["chosen_branch"])
        if path_correct:
            path_successes += 1
            
        # 2. Execution check
        exec_correct = False
        output = final_state.get("final_output")
        expected = ref["expected"]
        
        if output:
            exec_correct = True
            for k, expected_val in expected.items():
                out_val = output.get(k)
                if out_val is None:
                    exec_correct = False
                    break
                    
                # Evaluate numerically (within tolerance)
                tol = 0.05 if k == "amount" else 1e-4
                if isinstance(expected_val, float):
                    if abs(out_val - expected_val) > tol:
                        exec_correct = False
                        break
                elif isinstance(expected_val, list):
                    if not isinstance(out_val, list) or len(out_val) != len(expected_val):
                        exec_correct = False
                        break
                    for ov, ev in zip(out_val, expected_val):
                        if abs(ov - ev) > tol:
                            exec_correct = False
                            break
                else:
                    if out_val != expected_val:
                        exec_correct = False
                        break
                        
        if exec_correct:
            execution_successes += 1
            
        # 3. E2E success check
        e2e_correct = path_correct and exec_correct
        if e2e_correct:
            e2e_successes += 1
            
        print(f"Path Selection: {'SUCCESS' if path_correct else 'FAILED'} (Got: {final_state.get('chosen_branch')}, Expected: {ref['chosen_branch']})")
        print(f"Execution: {'SUCCESS' if exec_correct else 'FAILED'} (Got: {output}, Expected: {expected})")
        print(f"E2E Success: {'SUCCESS' if e2e_correct else 'FAILED'}")
        
        results.append({
            "run_index": i,
            "user_goal": user_goals[i],
            "step_history": final_state["step_history"],
            "error_logs": final_state["error_logs"],
            "chosen_branch": final_state.get("chosen_branch"),
            "expected_branch": ref["chosen_branch"],
            "output": output,
            "expected": expected,
            "path_correct": path_correct,
            "exec_correct": exec_correct,
            "e2e_correct": e2e_correct
        })
        
    path_acc = (path_successes / 10.0) * 100.0
    exec_acc = (execution_successes / 10.0) * 100.0
    e2e_acc = (e2e_successes / 10.0) * 100.0
    
    print("\n" + "=" * 50)
    print("BRANCHING SWEEP SUMMARY")
    print("=" * 50)
    print(f"Path Selection Accuracy: {path_acc:.1f}% ({path_successes}/10)")
    print(f"Execution Accuracy:      {exec_acc:.1f}% ({execution_successes}/10)")
    print(f"End-to-End Success Rate: {e2e_acc:.1f}% ({e2e_successes}/10)")
    print("=" * 50)
    
    output_path = "blackbox_os/roles/data_scientist/workflows/results_branching_sweep_deepseek_chat.json"
    with open(output_path, "w") as f:
        json.dump({
            "path_accuracy": path_acc,
            "execution_accuracy": exec_acc,
            "e2e_success_rate": e2e_acc,
            "runs": results
        }, f, indent=2)
    print(f"Saved results to {output_path}")

if __name__ == "__main__":
    run_experiment()
