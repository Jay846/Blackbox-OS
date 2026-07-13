import json
import urllib.request
import math
from typing import Any, Dict, Optional
from blackbox_os.state.shared_state import SharedState
from blackbox_os.state.validation_manager import ValidationManager

import os
DEEPSEEK_API_KEY = os.environ.get('DEEPSEEK_API_KEY', 'YOUR_API_KEY_HERE')
DEEPSEEK_URL = 'https://api.deepseek.com/chat/completions'
MODEL = 'deepseek-chat'

# Ground Truth for the task:
# salary values = [12, 17, 33, 44, 51]
# mean = 31.4
# std (ddof=0) = sqrt(225.84) = 15.0279739...
# Leakage column: total_revenue_offset (derived from target variable)
# Data drift is present (drift score = 0.35, which is > 0.25 threshold)
GROUND_TRUTH = {
    "mean": 31.4,
    "std": 15.027973915,
    "leakage_columns": ["total_revenue_offset"],
    "data_drift_detected": True
}


def call_deepseek(system_prompt: str, user_prompt: str) -> str:
    req_data = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0,
        "max_tokens": 300
    }
    
    req = urllib.request.Request(
        DEEPSEEK_URL,
        data=json.dumps(req_data).encode('utf-8'),
        headers={
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {DEEPSEEK_API_KEY}'
        },
        method='POST'
    )
    
    try:
        with urllib.request.urlopen(req) as response:
            res_body = response.read().decode('utf-8')
            res_json = json.loads(res_body)
            return res_json['choices'][0]['message']['content'].strip()
    except Exception as e:
        return f"Error: {e}"

def clean_json_string(raw: str) -> str:
    # Strip markdown block if present
    if "```json" in raw:
        raw = raw.split("```json", 1)[1]
    if "```" in raw:
        raw = raw.split("```", 1)[0]
    return raw.strip()

def parse_model_output(raw: str) -> Optional[Dict[str, Any]]:
    clean_raw = clean_json_string(raw)
    try:
        return json.loads(clean_raw)
    except Exception:
        return None

def score_output(parsed: Optional[Dict[str, Any]]) -> Dict[str, float]:
    """
    Grades the model's output against the ground truth.
    Returns:
      - f1_score (proxy for accuracy of calculations)
      - leakage_score (1.0 if incorrect, 0.0 if correct)
      - drift_score (1.0 if incorrect, 0.0 if correct)
      - latency_ms (simulated or parsed)
    """
    if parsed is None:
        return {"f1_score": 0.0, "leakage_score": 1.0, "drift_score": 1.0}
    
    # 1. Performance / Math Accuracy
    mean_val = parsed.get("normalized_salary_mean", 0.0)
    std_val = parsed.get("normalized_salary_std", 0.0)
    
    mean_err = abs(mean_val - GROUND_TRUTH["mean"]) / GROUND_TRUTH["mean"] if GROUND_TRUTH["mean"] != 0 else 0
    std_err = abs(std_val - GROUND_TRUTH["std"]) / GROUND_TRUTH["std"] if GROUND_TRUTH["std"] != 0 else 0
    
    # Mathematical correctness score between 0 and 1
    math_score = max(0.0, 1.0 - (mean_err + std_err))
    
    # 2. Leakage check
    leakage_cols = parsed.get("leakage_columns", [])
    # Normalize list
    leakage_cols = [c.lower().strip() for c in leakage_cols]
    correct_leakage = "total_revenue_offset" in leakage_cols and len(leakage_cols) == 1
    leakage_score = 0.0 if correct_leakage else 1.0
    
    # 3. Drift check
    drift_detected = parsed.get("data_drift_detected", None)
    drift_score = 0.0 if drift_detected == GROUND_TRUTH["data_drift_detected"] else 1.0
    
    # F1 score is a combination of math accuracy and correctly identifying drift
    f1_score = (math_score + (1.0 - drift_score)) / 2.0
    
    return {
        "f1_score": f1_score,
        "leakage_score": leakage_score,
        "drift_score": drift_score
    }

def run_benchmark():
    print("=" * 70)
    print("PHASE 3: DATA SCIENTIST EXECUTION BENCHMARK (DEEPSEEK V4 FLASH)")
    print("=" * 70)
    
    user_query = (
        "We have a dataset with columns: ['age', 'salary', 'customer_churned', 'total_revenue_offset', 'timestamp'].\n"
        "- The 'salary' values are: [12, 17, 33, 44, 51].\n"
        "- The 'customer_churned' is the target variable. The column 'total_revenue_offset' is computed as 'customer_churned * 5 + 10'.\n"
        "- There is significant data drift detected in the 'age' feature distribution, with a Kolmogorov-Smirnov drift score of 0.35.\n\n"
        "Task: Calculate StandardScaler parameters (population mean and population standard deviation ddof=0) "
        "for 'salary', identify leakage columns, and check if data drift was detected (drift threshold is 0.25).\n\n"
        "Output ONLY a valid JSON object with the keys:\n"
        "{\n"
        "  \"normalized_salary_mean\": float,\n"
        "  \"normalized_salary_std\": float,\n"
        "  \"leakage_columns\": [str],\n"
        "  \"data_drift_detected\": bool\n"
        "}"
    )
    
    # --- 1. BARE PROMPT CONDITION ---
    bare_system = "You are a data scientist. Solve the user's request. Return ONLY valid JSON."
    print("\n--- Running BARE Condition ---")
    bare_raw = call_deepseek(bare_system, user_query)
    print("Bare Raw Output:\n", bare_raw)
    
    bare_parsed = parse_model_output(bare_raw)
    bare_scores = score_output(bare_parsed)
    
    # Setup SharedState for Bare
    # In a real pipeline, if the agent detects leakage or drift, it flags it.
    # If the agent FAILS to detect leakage but it's present, the auditor (grading) flags it as leakage error.
    # Therefore, leakage is present if either the agent correctly found it OR the auditor found it.
    agent_found_leakage = (bare_parsed is not None and "total_revenue_offset" in [c.lower() for c in bare_parsed.get("leakage_columns", [])])
    agent_found_drift = (bare_parsed is not None and bare_parsed.get("data_drift_detected") == True)

    bare_state = SharedState(
        active_role="data_scientist",
        active_workflow="Bare Ingestion",
        metrics={
            "f1_score": bare_scores["f1_score"],
            "leakage_score": bare_scores["leakage_score"],
            "drift_score": bare_scores["drift_score"],
            "latency_ms": 120.0
        },
        target_leakage_detected=agent_found_leakage,
        data_drift_detected=agent_found_drift
    )
    
    bare_result = ValidationManager.evaluate(bare_state)
    
    # --- 2. EXPERT SOP PROMPT CONDITION ---
    expert_system = (
        "You are a Senior Data Scientist Agent operating under strict enterprise governance rules.\n"
        "You must execute the request by adhering to the following standard operating procedures (SOPs):\n\n"
        "Stage 2 (Data Ingestion & Integrity): Scan for target leakage. Identify columns derived directly from the target.\n"
        "Stage 4 (Preprocessing & Features): Apply StandardScaler. Calculate population mean and population standard deviation (ddof=0).\n"
        "Stage 8 (Compliance): Check if data drift was flagged.\n\n"
        "Output strictly as a valid JSON object. Do not include markdown blocks, explanation or preamble."
    )
    print("\n--- Running EXPERT SOP Condition ---")
    expert_raw = call_deepseek(expert_system, user_query)
    print("Expert Raw Output:\n", expert_raw)
    
    expert_parsed = parse_model_output(expert_raw)
    expert_scores = score_output(expert_parsed)
    
    # Setup SharedState for Expert
    agent_found_leakage_exp = (expert_parsed is not None and "total_revenue_offset" in [c.lower() for c in expert_parsed.get("leakage_columns", [])])
    agent_found_drift_exp = (expert_parsed is not None and expert_parsed.get("data_drift_detected") == True)

    expert_state = SharedState(
        active_role="data_scientist",
        active_workflow="SOP Ingestion",
        metrics={
            "f1_score": expert_scores["f1_score"],
            "leakage_score": expert_scores["leakage_score"],
            "drift_score": expert_scores["drift_score"],
            "latency_ms": 110.0
        },
        target_leakage_detected=agent_found_leakage_exp,
        data_drift_detected=agent_found_drift_exp
    )
    
    expert_result = ValidationManager.evaluate(expert_state)
    
    # --- 3. COMPARATIVE REPORT ---
    print("\n" + "=" * 70)
    print("COMPARATIVE EVALUATION REPORT")
    print("=" * 70)
    
    print(f"{'Condition':<15} | {'Execution Acc':<15} | {'Gatekeeper Appr':<15} | {'Loopback Target':<20}")
    print("-" * 70)
    print(f"{'BARE':<15} | {bare_scores['f1_score']:<15.4f} | {str(bare_result.approved):<15} | {str(bare_result.target_node):<20}")
    print(f"{'EXPERT SOP':<15} | {expert_scores['f1_score']:<15.4f} | {str(expert_result.approved):<15} | {str(expert_result.target_node):<20}")
    
    print("\n--- Diagnostic Gatekeeper Feedback ---")
    print("BARE Feedback:\n", bare_result.feedback)
    print("\nEXPERT SOP Feedback:\n", expert_result.feedback)
    print("=" * 70)

if __name__ == "__main__":
    run_benchmark()
