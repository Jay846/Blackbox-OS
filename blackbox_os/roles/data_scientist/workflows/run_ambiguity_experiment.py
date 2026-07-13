import os
import sys
import json
import random
import urllib.request
import time
from typing import Dict, Any, List, Optional, Tuple

# Add workspace to path
sys.path.append(os.getcwd())

MODEL = "deepseek-chat"
PROVIDER = "deepseek"

# Target Skills definitions matching production
SKILLS_UNDER_TEST = {
    "lookahead_bias_audit": {
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
    "trade_ev_calculator": {
        "id": "trade_ev_calculator",
        "bare_desc": "Compute net expected value of trades from performance JSON file.",
        "expert_desc": (
            "Computes expected value including transaction fee from JSON performance data. "
            "Input: JSON file containing list of dicts under 'trades' key with 'return' key. "
            "Formula: ev = (win_prob * (win_return - fee)) + ((1 - win_prob) * (loss_return - fee)). "
            "Where win_prob is count of positive returns / total trades, win_return is average of positive returns, "
            "and loss_return is average of negative returns. "
            "The final JSON response dictionary from Python MUST contain exactly the key 'ev' for expected value. "
            "Output schema: {\"ev\": float}. Do not confuse with raw risk/reward ratio."
        )
    }
}

# The 2x2 Matrix of queries under test
# Systematically varying Novelty (Low vs High) and Ambiguity (Low vs High)
MATRIX_CELLS = {
    "LN_LA": [
        {"skill_id": "lookahead_bias_audit", "text": "Audit features.csv for lookahead bias. Check if any column formula references future timestamps."},
        {"skill_id": "trade_ev_calculator", "text": "Load trade_performance.json and calculate the net expected value (EV) deducting a transaction fee of 5.0."}
    ],
    "HN_LA": [
        {"skill_id": "lookahead_bias_audit", "text": "Scan our predictive features database schema to detect temporal contamination or future causality leaks in our mathematical formulas."},
        {"skill_id": "trade_ev_calculator", "text": "Analyze our trade outcomes database to calculate the mathematical expectation of return edge after adjusting for execution frictional cost."}
    ],
    "LN_HA": [
        {"skill_id": "lookahead_bias_audit", "text": "Check features.csv to audit if the columns are clean, valid, and free of bias or leakage."},
        {"skill_id": "trade_ev_calculator", "text": "Check trade_performance.json to compute the average return expectation and see if we have positive edge."}
    ],
    "HN_HA": [
        {"skill_id": "lookahead_bias_audit", "text": "Assess if the training columns suffer from leakage, distribution drift, or any future-dependent scaling issues."},
        {"skill_id": "trade_ev_calculator", "text": "Assess features.csv to analyze the mean return expectation and calculate our optimal allocation scale."}
    ]
}

# Helper to shuffle fillers deterministically
def seeded_shuffle(arr: list, seed: int) -> list:
    a = arr.copy()
    r = random.Random(seed)
    r.shuffle(a)
    return a

def build_library(size: int, fillers: List[Dict[str, Any]], seed: int) -> List[Dict[str, Any]]:
    # Remove test targets from fillers to avoid duplication
    target_ids = set(SKILLS_UNDER_TEST.keys())
    unique_fillers = [f for f in fillers if f.get("id") not in target_ids]
    
    n_fillers = max(0, size - len(target_ids))
    shuffled_fillers = seeded_shuffle(unique_fillers, seed)[:n_fillers]
    
    # Merge targets with fillers
    library = list(SKILLS_UNDER_TEST.values()) + shuffled_fillers
    return seeded_shuffle(library, seed + 1)

def format_skill_list(library: List[Dict[str, Any]]) -> str:
    lines = []
    for s in library:
        if s.get("id") in SKILLS_UNDER_TEST:
            desc = s["expert_desc"]
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
        "max_tokens": 512
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

def run_experiment():
    print("=" * 80)
    print(f"QUERY VARIATION MATRIX EXPERIMENT (Model: {MODEL}, N=500)")
    print("=" * 80)
    
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
    
    results = {
        "LN_LA": {"runs": [], "score": 0.0},
        "HN_LA": {"runs": [], "score": 0.0},
        "LN_HA": {"runs": [], "score": 0.0},
        "HN_HA": {"runs": [], "score": 0.0}
    }
    
    # Run 5 trials per cell
    trials = 5
    
    for cell_name, query_list in MATRIX_CELLS.items():
        print(f"\nEvaluating Cell: {cell_name}...", flush=True)
        correct_selections = 0
        total_selections = 0
        
        for run_idx in range(trials):
            # Seed-shuffled library per trial to ensure variety
            library = build_library(500, unique_fillers, seed=(run_idx * 100))
            listing = format_skill_list(library)
            
            system_prompt = (
                "You are an intelligent multi-agent routing component.\n"
                "Review the user query and select the single most appropriate skill_id from the list of available skills.\n"
                "Do NOT write any code. Just return a JSON containing the chosen skill.\n\n"
                "Output Format:\n"
                "Return ONLY a JSON dictionary of the form:\n"
                "{\n"
                "  \"chosen_skill_id\": \"selected_skill_id\"\n"
                "}\n\n"
                "Available Skills:\n" + listing
            )
            
            for query_info in query_list:
                target_skill = query_info["skill_id"]
                user_prompt = query_info["text"]
                
                raw_resp = query_llm(system_prompt, user_prompt)
                parsed = clean_json_response(raw_resp)
                
                chosen_skill = None
                if parsed:
                    chosen_skill = parsed.get("chosen_skill_id") or parsed.get("skill_id")
                
                success = (chosen_skill == target_skill)
                if success:
                    correct_selections += 1
                total_selections += 1
                
                results[cell_name]["runs"].append({
                    "trial": run_idx,
                    "target_skill": target_skill,
                    "user_prompt": user_prompt,
                    "response": raw_resp,
                    "chosen_skill": chosen_skill,
                    "success": success
                })
                
                # Sleep briefly to respect platform limits
                time.sleep(1.0)
                
        cell_acc = (correct_selections / total_selections) * 100.0
        results[cell_name]["score"] = cell_acc
        print(f"Cell {cell_name} Accuracy: {cell_acc:.1f}% ({correct_selections}/{total_selections})")
        
    # Print clean summary grid
    print("\n" + "=" * 50)
    print("FINAL 2x2 QUERY VARIATION MATRIX RESULTS")
    print("=" * 50)
    print(f"Low phrasing novelty | Low ambiguity:  {results['LN_LA']['score']:.1f}%")
    print(f"High phrasing novelty| Low ambiguity:  {results['HN_LA']['score']:.1f}%")
    print(f"Low phrasing novelty | High ambiguity: {results['LN_HA']['score']:.1f}%")
    print(f"High phrasing novelty| High ambiguity: {results['HN_HA']['score']:.1f}%")
    print("=" * 50)
    
    # Save output
    output_path = "blackbox_os/roles/data_scientist/workflows/results_ambiguity_sweep_deepseek_chat.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results successfully saved to {output_path}")

if __name__ == "__main__":
    run_experiment()
