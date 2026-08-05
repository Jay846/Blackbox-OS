import os
import sys
import json
import random
import time
import re
import ast
from typing import Dict, Any, List, Optional


# ── Dynamic Workspace Root ───────────────────────────────────────────────────
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

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODEL_MAP = {
    "deepseek-v4-flash": "deepseek/deepseek-chat",
    "gpt-4o-mini": "openai/gpt-4o-mini",
    "nvidia/nemotron-3-ultra-550b-a55b:free": "nvidia/nemotron-3-ultra-550b-a55b:free",
    "openai/gpt-oss-20b:free": "openai/gpt-oss-20b:free",
}

# ── Skills under test ────────────────────────────────────────────────────────
SKILLS_UNDER_TEST = {
    "lookahead_bias_audit": {
        "id": "lookahead_bias_audit",
        "expert_desc": (
            "Audits column formulas in features CSV for leakage/lookahead bias. "
            "Input: CSV with columns 'column_name', 'formula'. "
            "If any formula references future timestamps (t+1, t+2) or contains 'target', "
            "leakage_detected is true. Output schema: {\"leakage_detected\": bool}."
        )
    },
    "trade_ev_calculator": {
        "id": "trade_ev_calculator",
        "expert_desc": (
            "Computes expected value including transaction fee from JSON performance data. "
            "Input: JSON with list of dicts under 'trades' key with 'return'. "
            "Formula: ev = (win_prob * (win_return - fee)) + ((1 - win_prob) * (loss_return - fee)). "
            "Output schema: {\"ev\": float}."
        )
    }
}

# ── 2×2 Query Matrix (Cleaned & Calibrated) ──────────────────────────────────
MATRIX_CELLS = {
    "LN_LA": [  # Low Novelty, Low Ambiguity
        {"skill_id": "lookahead_bias_audit",
         "text": "Audit features.csv for lookahead bias. Check if any column formula references future timestamps."},
        {"skill_id": "trade_ev_calculator",
         "text": "Load trade_performance.json and calculate the net expected value (EV) deducting a transaction fee of 5.0."}
    ],
    "HN_LA": [  # High Novelty, Low Ambiguity
        {"skill_id": "lookahead_bias_audit",
         "text": "Scan our predictive features database schema to detect temporal contamination or future causality leaks in our mathematical formulas."},
        {"skill_id": "trade_ev_calculator",
         "text": "Analyze our trade outcomes database to calculate the mathematical expectation of return edge after adjusting for execution frictional cost."}
    ],
    "LN_HA": [  # Low Novelty, High Ambiguity
        {"skill_id": "lookahead_bias_audit",
         "text": "Check features.csv to audit if the columns are clean, valid, and free of bias or leakage."},
        {"skill_id": "trade_ev_calculator",
         "text": "Check trade_performance.json to compute the average return expectation and see if we have positive edge."}
    ],
    "HN_HA": [  # High Novelty, High Ambiguity (Fracture Point)
        {"skill_id": "lookahead_bias_audit",
         "text": "Assess if the feature definitions suffer from hidden temporal leakage or future-dependent scaling issues."},
        {"skill_id": "trade_ev_calculator",
         "text": "Evaluate historical execution logs to determine our net expected return payout per trade after costs."}
    ]
}

# ── Helpers ──────────────────────────────────────────────────────────────────
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

def seeded_shuffle(arr: list, seed: int) -> list:
    a = arr.copy()
    random.Random(seed).shuffle(a)
    return a

def build_library(size: int, fillers: List[Dict], seed: int) -> List[Dict]:
    target_ids = set(SKILLS_UNDER_TEST.keys())
    unique = [f for f in fillers if f.get("id") not in target_ids]
    n = max(0, size - len(target_ids))
    selected = seeded_shuffle(unique, seed)[:n]
    return seeded_shuffle(list(SKILLS_UNDER_TEST.values()) + selected, seed + 1)

def format_skill_list(library: List[Dict]) -> str:
    lines = []
    target_ids = set(SKILLS_UNDER_TEST.keys())
    for s in library:
        if s.get("id") in target_ids:
            lines.append(f"{s['id']}: {s['expert_desc']}")
        else:
            concept = s.get("concept", "Tool option.")
            extra = f" {s['disambiguator']}" if "disambiguator" in s else ""
            lines.append(f"{s['id']}: {concept}{extra}")
    return "\n".join(lines)

def clean_and_extract_json(raw: str) -> Optional[Dict]:
    if not raw:
        return None
    cleaned = re.sub(r'<think>[\s\S]*?</think>', '', raw).strip()
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start != -1 and end > start:
        candidate = cleaned[start:end+1]
        try:
            return json.loads(candidate)
        except Exception:
            try:
                return ast.literal_eval(candidate)
            except Exception:
                pass
    return None

def query_llm(system_prompt: str, user_prompt: str, model_name: str) -> str:
    import urllib.request
    api_key = os.environ.get("OPENROUTER_API_KEY", "Your_API_KEY")
    if not api_key:
        return '{"error": "OPENROUTER_API_KEY not set"}'
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": "https://github.com/google/antigravity"
    }
    target = OPENROUTER_MODEL_MAP.get(model_name, model_name)
    data = {
        "model": target,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.0,
        "max_tokens": 1024
    }
    for attempt in range(4):
        try:
            req = urllib.request.Request(OPENROUTER_URL, method="POST")
            for k, v in headers.items():
                req.add_header(k, v)
            with urllib.request.urlopen(req, data=json.dumps(data).encode(), timeout=45) as resp:
                return json.loads(resp.read().decode())["choices"][0]["message"]["content"]
        except Exception as e:
            if "429" in str(e) and attempt < 3:
                time.sleep(2 * (attempt + 1))
                continue
            return f'{{"error": "{str(e)}"}}'
    return '{"error": "Max retries exceeded"}'

# ── Main Experiment ──────────────────────────────────────────────────────────
def run_matrix_experiment(model_name: str, trials: int = 5, catalog_size: int = 500):
    print("=" * 70)
    print(f"QUERY VARIATION MATRIX (Model: {model_name}, N={catalog_size})")
    print("=" * 70)

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

    seen = set()
    unique_fillers = [f for f in fillers if f.get("id") and not (f["id"] in seen or seen.add(f["id"]))]
    print(f"Loaded {len(unique_fillers)} background fillers.")

    results = {cell: {"correct": 0, "total": 0, "runs": []} for cell in MATRIX_CELLS}

    for cell_name, queries in MATRIX_CELLS.items():
        print(f"\n── Cell: {cell_name} ──")
        for trial in range(trials):
            library = build_library(catalog_size, unique_fillers, seed=trial * 97 + 13)
            listing = format_skill_list(library)
            system_prompt = (
                "You are an intelligent multi-agent routing component in an enterprise OS.\n"
                "Select the single most appropriate skill_id from the list below.\n"
                "Return ONLY a JSON object in this exact format:\n"
                '{"chosen_skill_id": "the_skill_id"}\n\n'
                "Available Skills:\n" + listing
            )
            for q in queries:
                target = q["skill_id"]
                raw = query_llm(system_prompt, q["text"], model_name)
                parsed = clean_and_extract_json(raw)
                chosen = None
                if parsed:
                    chosen = parsed.get("chosen_skill_id") or parsed.get("skill_id")
                
                success = (chosen == target)
                if success:
                    results[cell_name]["correct"] += 1
                results[cell_name]["total"] += 1
                results[cell_name]["runs"].append({
                    "trial": trial,
                    "target": target,
                    "query": q["text"],
                    "chosen": chosen,
                    "success": success
                })
                icon = "✓" if success else "✗"
                print(f"  [{cell_name}] trial={trial+1}/{trials} {icon} | Target: {target} | Chosen: {chosen}")
                time.sleep(0.4)

        acc = (results[cell_name]["correct"] / results[cell_name]["total"]) * 100
        print(f"  → Cell {cell_name} Accuracy: {acc:.1f}% ({results[cell_name]['correct']}/{results[cell_name]['total']})")

    # Summary Grid Output
    print("\n" + "=" * 50)
    print("2×2 QUERY MATRIX SUMMARY (Selection Accuracy %)")
    print("=" * 50)
    print(f"               Low Ambiguity    High Ambiguity")
    ln_la_acc = (results['LN_LA']['correct'] / results['LN_LA']['total']) * 100
    ln_ha_acc = (results['LN_HA']['correct'] / results['LN_HA']['total']) * 100
    hn_la_acc = (results['HN_LA']['correct'] / results['HN_LA']['total']) * 100
    hn_ha_acc = (results['HN_HA']['correct'] / results['HN_HA']['total']) * 100
    print(f"Low Novelty   :    {ln_la_acc:5.1f}%            {ln_ha_acc:5.1f}%")
    print(f"High Novelty  :    {hn_la_acc:5.1f}%            {hn_ha_acc:5.1f}%")
    print("=" * 50)

    # Save Results
    model_clean = model_name.replace("/", "_").replace("-", "_").replace(":", "_")
    out_dir = os.path.join(WORKSPACE_ROOT, "blackbox_os", "roles", "data_scientist", "workflows")
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, f"results_query_matrix_{model_clean}.json")
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved → {out_file}\n")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument("--all-models", action="store_true")
    parser.add_argument("--trials", type=int, default=5)
    args = parser.parse_args()

    if args.all_models:
        for m in OPENROUTER_MODEL_MAP:
            run_matrix_experiment(m, trials=args.trials)
    else:
        run_matrix_experiment(args.model, trials=args.trials)
