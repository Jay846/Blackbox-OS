import os
import sys
import json
import random
import subprocess
import tempfile
import time
import re
from typing import Dict, Any, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor

# ── Dynamic Workspace Root Resolution ─────────────────────────────────────
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

MODEL = "deepseek-v4-flash"
PROVIDER = "openrouter"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

OPENROUTER_MODEL_MAP = {
    "deepseek-v4-flash": "deepseek/deepseek-chat",
    "deepseek-v4-pro": "deepseek/deepseek-r1",
    "gpt-4o-mini": "openai/gpt-4o-mini",
    "nvidia/nemotron-3-ultra-550b-a55b:free": "nvidia/nemotron-3-ultra-550b-a55b:free",
    "google/gemma-4-26b-a4b-it:free": "google/gemma-4-26b-a4b-it:free",
    "openai/gpt-oss-20b:free": "openai/gpt-oss-20b:free",
}

# Target skills using exact mock_data paths matching disk
TARGET_SKILLS = [
    {
        "id": "kelly_position_size",
        "bare_desc": "Compute Kelly fraction and dollar amount based on execution CSV files.",
        "expert_desc": (
            "Computes optimal leverage and bet size. "
            "Input: a CSV file containing columns 'symbol', 'side', 'price', 'amount', 'realized_pnl'. "
            "Formula: f = (win_rate * payoff - (1 - win_rate)) / payoff. Dollar amount = f * bankroll. "
            "Output schema: {\"fraction\": float, \"amount\": float}."
        ),
        "queries": [
            {
                "text": "Analyze the trade history in blackbox_os/roles/data_scientist/workflows/mock_data/grid_bot_fills.csv. Calculate the win rate and payoff ratio for BTC trades, and determine the Kelly fraction for a total bankroll of 50000.",
                "ground_truth": {"fraction": 0.375, "amount": 18750.0}
            },
            {
                "text": "Using blackbox_os/roles/data_scientist/workflows/mock_data/grid_bot_fills.csv, find Kelly fraction for BTC trades with bankroll 100000.",
                "ground_truth": {"fraction": 0.375, "amount": 37500.0}
            },
            {
                "text": "Evaluate BTC trades in blackbox_os/roles/data_scientist/workflows/mock_data/grid_bot_fills.csv and calculate Kelly optimal allocation for bankroll 20000.",
                "ground_truth": {"fraction": 0.375, "amount": 7500.0}
            }
        ]
    },
    {
        "id": "atr_dynamic_stop",
        "bare_desc": "Compute stop distance from ATR calculated over historical price CSV.",
        "expert_desc": (
            "Calculates trailing stop distance based on ATR from a CSV. "
            "Input: CSV with columns 'timestamp', 'high', 'low', 'close'. "
            "TR = max(high - low, abs(high - close_prev), abs(low - close_prev)). ATR is 14-period average. "
            "Output schema: {\"stop_distance\": float}."
        ),
        "queries": [
            {
                "text": "Read blackbox_os/roles/data_scientist/workflows/mock_data/eth_usd_prices.csv, calculate the ATR over the last 14 periods, and determine the stop loss distance using a multiplier of 2.0.",
                "ground_truth": {"stop_distance": 20.0}
            },
            {
                "text": "Calculate 14-period ATR from blackbox_os/roles/data_scientist/workflows/mock_data/eth_usd_prices.csv with trailing stop multiplier 1.5.",
                "ground_truth": {"stop_distance": 15.0}
            },
            {
                "text": "Compute trailing stop distance for blackbox_os/roles/data_scientist/workflows/mock_data/eth_usd_prices.csv using 14-period ATR and 2.5 multiplier.",
                "ground_truth": {"stop_distance": 25.0}
            }
        ]
    },
    {
        "id": "trade_ev_calculator",
        "bare_desc": "Compute net expected value of trades from performance JSON file.",
        "expert_desc": (
            "Computes expected value from JSON performance data. "
            "Input: JSON file under 'trades' key with 'return' key. "
            "Formula: ev = (win_prob * (win_return - fee)) + ((1 - win_prob) * (loss_return - fee)). "
            "Output schema: {\"ev\": float}."
        ),
        "queries": [
            {
                "text": "Load blackbox_os/roles/data_scientist/workflows/mock_data/trade_performance.json, count successful vs failed trades to find probability, compute average win vs average loss, and calculate the net EV deducting a fee of 5.0 per trade.",
                "ground_truth": {"ev": 75.0}
            },
            {
                "text": "Calculate net expected trade value from blackbox_os/roles/data_scientist/workflows/mock_data/trade_performance.json with transaction fee 2.0.",
                "ground_truth": {"ev": 78.0}
            },
            {
                "text": "Determine net EV per trade in blackbox_os/roles/data_scientist/workflows/mock_data/trade_performance.json with zero fee.",
                "ground_truth": {"ev": 80.0}
            }
        ]
    },
    {
        "id": "lookahead_bias_audit",
        "bare_desc": "Detect lookahead bias in columns from feature CSV.",
        "expert_desc": (
            "Audits feature CSV formulas for lookahead bias. "
            "Input: CSV with 'column_name', 'formula'. "
            "If formula contains future timestamps (t+1, t+2) or 'target', leakage_detected is true. "
            "Output schema: {\"leakage_detected\": bool}."
        ),
        "queries": [
            {
                "text": "Audit blackbox_os/roles/data_scientist/workflows/mock_data/features.csv for lookahead bias. Check if any column formula references future timestamps or targets.",
                "ground_truth": {"leakage_detected": True}
            },
            {
                "text": "Check blackbox_os/roles/data_scientist/workflows/mock_data/features.csv for data leakage or future timestamp lookahead in feature definitions.",
                "ground_truth": {"leakage_detected": True}
            },
            {
                "text": "Verify if blackbox_os/roles/data_scientist/workflows/mock_data/features.csv contains lookahead bias references like t+1 or target.",
                "ground_truth": {"leakage_detected": True}
            }
        ]
    },
    {
        "id": "standard_scaler_apply",
        "bare_desc": "Compute StandardScaler parameters on CSV values column.",
        "expert_desc": (
            "Computes population StandardScaler parameters (mean, std) on returns in CSV. "
            "Formula: mean = sum(x)/N, std = sqrt(sum((x - mean)^2)/N). "
            "Output schema: {\"mean\": float, \"std\": float}."
        ),
        "queries": [
            {
                "text": "Read blackbox_os/roles/data_scientist/workflows/mock_data/raw_returns.csv and apply standard scale normalization on the 'return' column. Return the population mean and std.",
                "ground_truth": {"mean": 31.4, "std": 15.027973915}
            },
            {
                "text": "Normalize return column in blackbox_os/roles/data_scientist/workflows/mock_data/raw_returns.csv using standard scaler and output mean and std.",
                "ground_truth": {"mean": 31.4, "std": 15.027973915}
            },
            {
                "text": "Calculate population mean and standard deviation for 'return' in blackbox_os/roles/data_scientist/workflows/mock_data/raw_returns.csv.",
                "ground_truth": {"mean": 31.4, "std": 15.027973915}
            }
        ]
    },
    {
        "id": "data_drift_monitor",
        "bare_desc": "Given drift metrics JSON, check for drift.",
        "expert_desc": (
            "Evaluates KS drift score against warning threshold in drift metrics JSON. "
            "Input: JSON file metrics -> implied_vol -> ks_score and warning_threshold. "
            "If ks_score > warning_threshold, data_drift_detected is true. "
            "Output schema: {\"data_drift_detected\": bool}."
        ),
        "queries": [
            {
                "text": "Analyze blackbox_os/roles/data_scientist/workflows/mock_data/drift_metrics.json. Compare the daily distribution metrics of the feature 'implied_vol' against baseline distribution, and alert if drift exceeds threshold.",
                "ground_truth": {"data_drift_detected": True}
            },
            {
                "text": "Read blackbox_os/roles/data_scientist/workflows/mock_data/drift_metrics.json and determine if implied_vol drift exceeds warning threshold.",
                "ground_truth": {"data_drift_detected": True}
            },
            {
                "text": "Check if feature implied_vol in blackbox_os/roles/data_scientist/workflows/mock_data/drift_metrics.json exhibits significant data drift.",
                "ground_truth": {"data_drift_detected": True}
            }
        ]
    }
]

def locate_filler_file(filename: str) -> str:
    cwd = os.getcwd()
    candidate_paths = [
        os.path.join(SCRIPT_DIR, "skill_experiment", filename),
        os.path.join(SCRIPT_DIR, "..", "skill_experiment", filename),
        os.path.join(SCRIPT_DIR, "..", "..", "skill_experiment", filename),
        os.path.join(SCRIPT_DIR, "..", "..", "..", "skill_experiment", filename),
        os.path.join(WORKSPACE_ROOT, "skill_experiment", filename),
        os.path.join(WORKSPACE_ROOT, "blackbox_os", "skill_experiment", filename),
        os.path.join(cwd, "skill_experiment", filename),
        os.path.join(cwd, filename),
    ]
    for path in candidate_paths:
        normalized = os.path.abspath(path)
        if os.path.exists(normalized):
            return normalized
    raise FileNotFoundError(f"Could not locate '{filename}'.")

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
            lines.append(f"{s['id']}: {desc}")
        else:
            concept = s.get("concept", "Tool option.")
            desc = f"{concept} {s['disambiguator']}" if "disambiguator" in s else concept
            lines.append(f"{s['id']}: {desc}")
    return "\n".join(lines)

def clean_and_extract_json(raw_response: str) -> Optional[Dict[str, Any]]:
    if not raw_response:
        return None
    cleaned = re.sub(r'<think>[\s\S]*?</think>', '', raw_response).strip()

    fence_match = re.search(r'```(?:json)?\s*(\{[\s\S]*?\})\s*```', cleaned, re.IGNORECASE)
    if fence_match:
        try:
            return json.loads(fence_match.group(1))
        except Exception:
            pass

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = cleaned[start:end + 1]
        try:
            return json.loads(candidate)
        except Exception:
            try:
                import ast
                return ast.literal_eval(candidate)
            except Exception:
                pass
    return None

def query_llm(system_prompt: str, user_prompt: str) -> str:
    import urllib.request

    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        return '{"error": "OPENROUTER_API_KEY environment variable not set"}'
        
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": "https://github.com/google/antigravity"
    }
    target_model = OPENROUTER_MODEL_MAP.get(MODEL, MODEL)

    data = {
        "model": target_model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.0,
        "max_tokens": 2048
    }

    for attempt in range(5):
        try:
            req = urllib.request.Request(OPENROUTER_URL, method="POST")
            for k, v in headers.items():
                req.add_header(k, v)
            with urllib.request.urlopen(req, data=json.dumps(data).encode("utf-8"), timeout=30) as response:
                res_body = json.loads(response.read().decode("utf-8"))
                return res_body["choices"][0]["message"]["content"]
        except Exception as e:
            if "429" in str(e) and attempt < 4:
                time.sleep(2 * (attempt + 1))
                continue
            return f'{{"error": "{str(e)}"}}'
    return '{"error": "Max retries exceeded"}'

def execute_sandbox_code(code: str) -> Optional[Dict[str, Any]]:
    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
        f.write(code)
        temp_name = f.name
        
    try:
        res = subprocess.run(
            [sys.executable, temp_name],
            capture_output=True,
            text=True,
            cwd=WORKSPACE_ROOT,
            timeout=5.0
        )
        if res.returncode == 0:
            return clean_and_extract_json(res.stdout.strip())
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
    parsed = clean_and_extract_json(raw_response)

    sandbox_run_info = None
    chosen_tool = None

    if parsed is not None:
        chosen_tool = (
            parsed.get("chosen_skill_id")
            or parsed.get("skill_id")
            or parsed.get("chosen_tool")
        )

    sel_ok = (chosen_tool == task["target_id"])

    # Execute sandbox if selected
    if condition == "expert_sandbox" and sel_ok and parsed is not None:
        code = parsed.get("python_code")
        if code:
            sandbox_res = execute_sandbox_code(code)
            sandbox_run_info = {"code": code, "result": sandbox_res}
            if sandbox_res is not None:
                parsed = sandbox_res

    schema_ok = parsed is not None

    exec_ok = False
    if sel_ok and parsed is not None:
        exec_ok = True
        for k, gt_v in task["ground_truth"].items():
            p_v = parsed.get(k)
            if p_v is None or str(p_v).lower() == "nan":
                exec_ok = False
                break
            if isinstance(gt_v, float) and isinstance(p_v, (int, float)):
                if abs(float(p_v) - gt_v) > 0.05:
                    exec_ok = False
            elif p_v != gt_v:
                exec_ok = False

    e2e_ok = sel_ok and exec_ok and schema_ok
    return sel_ok, exec_ok, schema_ok, raw_response, parsed, sandbox_run_info

def run_experiment(dry_run: bool = False):
    print("=" * 75)
    print(f"PRODUCTION-GRADE EXECUTION SWEEP (Model: {MODEL})")
    print(f"WORKSPACE_ROOT: {WORKSPACE_ROOT}")
    print("=" * 75)

    try:
        fillers = json.load(open(locate_filler_file("fillers_v4.json")))
        for extra in ["hq_fillers.json", "additional_fillers.json"]:
            try:
                fillers += json.load(open(locate_filler_file(extra)))
            except Exception:
                pass
    except Exception as e:
        print("Failed to load fillers:", e)
        return

    seen_ids = set()
    unique_fillers = [f for f in fillers if f.get("id") and not (f["id"] in seen_ids or seen_ids.add(f["id"]))]
    print(f"Loaded {len(unique_fillers)} filler skills.")

    SIZES = [60, 200] if dry_run else [60, 200, 500, 1000]
    CONDITIONS = ["expert", "expert_sandbox"]

    eval_tasks = [{"target_id": t["id"], "query": q["text"], "ground_truth": q["ground_truth"]} for t in TARGET_SKILLS for q in t["queries"]]
    total_calls = len(SIZES) * len(CONDITIONS) * len(eval_tasks)

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
                    "1. Select the single most appropriate skill ID from the list below.\n"
                    "2. Write a complete, self-contained Python script to load the data files mentioned in the user query and compute the output.\n"
                    "3. The script MUST print a single JSON dictionary to stdout containing the schema results.\n\n"
                    "Return ONLY a JSON dictionary in this exact format:\n"
                    "{\n"
                    '  "chosen_skill_id": "the_skill_id",\n'
                    '  "python_code": "import pandas as pd\\nimport json\\n...\\nprint(json.dumps({...}))"\n'
                    "}\n\n"
                    "Available Skills:\n" + listing
                )
            else:
                system_prompt = (
                    "You are an intelligent multi-agent routing component in an enterprise OS.\n"
                    "Select the most appropriate skill ID from the list below and compute the required values yourself.\n\n"
                    "Return ONLY a JSON dictionary in this exact format:\n"
                    "{\n"
                    '  "chosen_skill_id": "the_skill_id",\n'
                    '  "metric_name": computed_value\n'
                    "}\n\n"
                    "Available Skills:\n" + listing
                )

            for task in eval_tasks:
                user_prompt = f"Target Query: \"{task['query']}\"\nCompute and return the values."
                futures.append(executor.submit(evaluate_task, task, condition, system_prompt, user_prompt))

            sel_correct = exec_correct = schema_correct = e2e_correct = 0

            for task, fut in zip(eval_tasks, futures):
                try:
                    sel_ok, exec_ok, schema_ok, raw_response, parsed_json, sandbox_run_info = fut.result()
                except Exception as e:
                    sel_ok = exec_ok = schema_ok = False
                    raw_response, parsed_json, sandbox_run_info = str(e), None, None

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
                    "selection_ok": sel_ok,
                    "execution_ok": exec_ok,
                    "schema_ok": schema_ok,
                    "e2e_ok": e2e_ok,
                    "parsed_json": parsed_json,
                    "sandbox_run_info": sandbox_run_info
                })

                if done % 6 == 0 or done == total_calls:
                    icon = "✓" if e2e_ok else "✗"
                    print(f"  [{done}/{total_calls}] {condition:15} {icon}  {task['target_id']}")

            results[size][condition] = {
                "selection_acc": (sel_correct / len(eval_tasks)) * 100,
                "execution_acc": (exec_correct / len(eval_tasks)) * 100,
                "schema_acc": (schema_correct / len(eval_tasks)) * 100,
                "e2e_acc": (e2e_correct / len(eval_tasks)) * 100
            }
            print(
                f"  → Result [{condition}]: "
                f"Selection {results[size][condition]['selection_acc']:.1f}% | "
                f"Execution {results[size][condition]['execution_acc']:.1f}% | "
                f"E2E {results[size][condition]['e2e_acc']:.1f}%"
            )

    model_clean = MODEL.replace("/", "_").replace("-", "_").replace(":", "_")
    out_dir = os.path.join(WORKSPACE_ROOT, "blackbox_os", "roles", "data_scientist", "workflows")
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, f"results_production_sweep_{model_clean}.json")

    with open(out_file, "w") as f:
        json.dump({"results": results, "logs": logs}, f, indent=2)

    print("\n" + "=" * 75)
    print(f"Completed. Results saved to:\n{out_file}")
    print("=" * 75)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--model", type=str, default="deepseek-v4-flash")
    parser.add_argument("--provider", type=str, default="openrouter")
    # Adding this line to accept --all-models:
    parser.add_argument("--all-models", action="store_true", help="Run experiment sequentially across all mapped models")
    args = parser.parse_args()

    MODEL = args.model
    PROVIDER = args.provider

    if args.all_models:
        for m in OPENROUTER_MODEL_MAP.keys():
            MODEL = m
            run_experiment(dry_run=args.dry_run)
    else:
        run_experiment(dry_run=args.dry_run) 
