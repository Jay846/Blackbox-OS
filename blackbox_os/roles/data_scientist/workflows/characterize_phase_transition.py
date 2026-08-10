#!/usr/bin/env python3
"""
Empirical phase-transition characterization
1) Semantic density: mean max nearest-neighbor cosine vs catalog size N
2) Optional live LLM selection accuracy at the same N values

Usage:
  # embeddings only (free, real geometry)
  python characterize_phase_transition.py

  # embeddings + live selection accuracy on 2 models
  python characterize_phase_transition.py --live --models deepseek-v4-flash gpt-4o-mini

  # full 4-model live (more cost)
  python characterize_phase_transition.py --live --models deepseek-v4-flash gpt-4o-mini nvidia/nemotron-3-ultra-550b-a55b:free openai/gpt-oss-20b:free
"""

import os
import sys
import json
import random
import time
import re
import ast
import argparse
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
root = SCRIPT_DIR
while root != os.path.dirname(root):
    if os.path.exists(os.path.join(root, "blackbox_os")):
        WORKSPACE_ROOT = root
        break
    root = os.path.dirname(root)
else:
    WORKSPACE_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "..", ".."))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("ERROR: pip install sentence-transformers")
    sys.exit(1)

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODEL_MAP = {
    "deepseek-v4-flash": "deepseek/deepseek-chat",
    "gpt-4o-mini": "openai/gpt-4o-mini",
    "nvidia/nemotron-3-ultra-550b-a55b:free": "nvidia/nemotron-3-ultra-550b-a55b:free",
    "openai/gpt-oss-20b:free": "openai/gpt-oss-20b:free",
}

# ── Targets (same family as your other experiments) ──────────────────────────
SKILLS_UNDER_TEST = [
    {
        "id": "lookahead_bias_audit",
        "bare": "Audits CSV features for lookahead bias and future timestamp leakage.",
        "expert": (
            "Audits column formulas in features CSV for leakage/lookahead bias. "
            "Input: CSV with 'column_name', 'formula'. Future timestamps (t+1,t+2) or 'target' "
            "=> leakage_detected true. Output: {\"leakage_detected\": bool}."
        ),
        "query": "Audit features.csv for lookahead bias. Check if any column formula references future timestamps.",
    },
    {
        "id": "trade_ev_calculator",
        "bare": "Calculates expected value of trades after transaction fee.",
        "expert": (
            "Computes expected value with fee from JSON trades[].return. "
            "ev = win_prob*(win-fee)+(1-win_prob)*(loss-fee). Output: {\"ev\": float}."
        ),
        "query": "Load trade_performance.json and calculate net expected value deducting a fee of 5.0.",
    },
    {
        "id": "kelly_position_size",
        "bare": "Calculates Kelly fraction position size from trade history.",
        "expert": (
            "Computes Kelly fraction and dollar amount from CSV realized_pnl. "
            "Output: {\"fraction\": float, \"amount\": float}."
        ),
        "query": "Analyze BTC trades in fills.csv and calculate Kelly fraction for a $50,000 bankroll.",
    },
    {
        "id": "atr_dynamic_stop",
        "bare": "Calculates ATR-based trailing stop distance.",
        "expert": (
            "14-period ATR from OHLC; stop_distance = ATR * multiplier. "
            "Output: {\"stop_distance\": float}."
        ),
        "query": "Read prices.csv, compute 14-period ATR, and find stop distance with multiplier 2.0.",
    },
    {
        "id": "data_drift_monitor",
        "bare": "Monitors feature drift against a warning threshold.",
        "expert": (
            "KS score vs warning_threshold in drift JSON. "
            "Output: {\"data_drift_detected\": bool}."
        ),
        "query": "Analyze drift metrics in drift.json and check if implied_vol exceeds warning threshold.",
    },
]
TARGET_IDS = {t["id"] for t in SKILLS_UNDER_TEST}

# ── IO helpers ───────────────────────────────────────────────────────────────
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

def load_fillers() -> List[Dict[str, str]]:
    fillers, seen = [], set(TARGET_IDS)
    for fn in ["fillers_v4.json", "hq_fillers.json", "additional_fillers.json"]:
        try:
            data = json.load(open(locate_filler_file(fn)))
        except Exception:
            continue
        for item in data:
            fid = item.get("id")
            if not fid or fid in seen:
                continue
            desc = (
                item.get("expert_desc")
                or item.get("bare_desc")
                or item.get("concept")
                or item.get("desc")
                or ""
            )
            if item.get("disambiguator"):
                desc = f"{desc} {item['disambiguator']}".strip()
            if not desc:
                continue
            fillers.append({"id": fid, "desc": desc})
            seen.add(fid)
    print(f"Loaded {len(fillers)} fillers.")
    return fillers

def seeded_sample(items: list, k: int, seed: int) -> list:
    a = list(items)
    random.Random(seed).shuffle(a)
    return a[:k]

# ── Embeddings ───────────────────────────────────────────────────────────────
def mean_max_nn(target_e: np.ndarray, catalog_e: np.ndarray, target_idx: List[int]) -> float:
    sims = np.dot(target_e, catalog_e.T)
    vals = []
    for i, j in enumerate(target_idx):
        row = sims[i].copy()
        row[j] = -1.0
        vals.append(float(np.max(row)))
    return float(np.mean(vals))

def compute_semantic_curve(model, fillers, sizes: List[int]) -> Dict[str, Any]:
    bare_t = model.encode([t["bare"] for t in SKILLS_UNDER_TEST], normalize_embeddings=True)
    expert_t = model.encode([t["expert"] for t in SKILLS_UNDER_TEST], normalize_embeddings=True)
    filler_e = model.encode([f["desc"] for f in fillers], normalize_embeddings=True, show_progress_bar=True)

    n_targets = len(SKILLS_UNDER_TEST)
    out = {"N": [], "bare_mean_max_cos": [], "expert_mean_max_cos": []}

    print("\n{:<8} {:>14} {:>14}".format("N", "Bare NN", "Expert NN"))
    print("-" * 40)
    for N in sizes:
        if N < n_targets:
            continue
        chosen = seeded_sample(range(len(fillers)), N - n_targets, seed=1000 + N)
        bare_cat = np.vstack([bare_t, filler_e[chosen]])
        expert_cat = np.vstack([expert_t, filler_e[chosen]])
        idx = list(range(n_targets))
        b = mean_max_nn(bare_t, bare_cat, idx)
        e = mean_max_nn(expert_t, expert_cat, idx)
        out["N"].append(N)
        out["bare_mean_max_cos"].append(b)
        out["expert_mean_max_cos"].append(e)
        print(f"{N:<8} {b:>14.4f} {e:>14.4f}")
    return out

# ── Live LLM selection (empirical) ───────────────────────────────────────────
def clean_json(raw: str) -> Optional[Dict]:
    if not raw:
        return None
    cleaned = re.sub(r"<think>[\s\S]*?</think>", "", raw).strip()
    s, e = cleaned.find("{"), cleaned.rfind("}")
    if s != -1 and e > s:
        cand = cleaned[s : e + 1]
        try:
            return json.loads(cand)
        except Exception:
            try:
                return ast.literal_eval(cand)
            except Exception:
                return None
    return None

def query_llm(system: str, user: str, model_name: str) -> str:
    import urllib.request
    key = os.environ.get("OPENROUTER_API_KEY", "Your_API_Key")
    if not key:
        return '{"error":"OPENROUTER_API_KEY not set"}'
    target = OPENROUTER_MODEL_MAP.get(model_name, model_name)
    payload = {
        "model": target,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0.0,
        "max_tokens": 2048,
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {key}",
        "HTTP-Referer": "https://github.com/google/antigravity",
    }
    for attempt in range(4):
        try:
            req = urllib.request.Request(OPENROUTER_URL, method="POST")
            for k, v in headers.items():
                req.add_header(k, v)
            with urllib.request.urlopen(req, data=json.dumps(payload).encode(), timeout=45) as resp:
                return json.loads(resp.read().decode())["choices"][0]["message"]["content"]
        except Exception as ex:
            if "429" in str(ex) and attempt < 3:
                time.sleep(2 * (attempt + 1))
                continue
            return f'{{"error":"{ex}"}}'
    return '{"error":"retries"}'

def format_catalog(tools: List[Dict[str, str]]) -> str:
    return "\n".join(f"{t['id']}: {t['desc']}" for t in tools)

def build_catalog_for_N(fillers: List[Dict], N: int, seed: int, use_expert_text: bool) -> List[Dict[str, str]]:
    n_targets = len(SKILLS_UNDER_TEST)
    dist = seeded_sample(fillers, max(0, N - n_targets), seed=seed)
    tools = []
    for t in SKILLS_UNDER_TEST:
        tools.append({"id": t["id"], "desc": t["expert"] if use_expert_text else t["bare"]})
    for f in dist:
        tools.append({"id": f["id"], "desc": f["desc"]})
    random.Random(seed + 7).shuffle(tools)
    return tools

def live_selection_accuracy(
    model_name: str,
    fillers: List[Dict],
    sizes: List[int],
    trials_per_target: int = 3,
    use_expert_text: bool = True,
) -> Dict[str, Any]:
    """
    Real empirical selection: for each N, each target query, ask model to pick skill_id.
    """
    acc_by_n = []
    print(f"\nLIVE selection accuracy | model={model_name}")
    for N in sizes:
        if N < len(SKILLS_UNDER_TEST):
            continue
        correct = total = 0
        for trial in range(trials_per_target):
            catalog = build_catalog_for_N(fillers, N, seed=5000 + N * 17 + trial, use_expert_text=use_expert_text)
            listing = format_catalog(catalog)
            system = (
                "You are a tool-routing component.\n"
                "Select the single best skill_id for the user query.\n"
                "Return ONLY JSON: {\"chosen_skill_id\": \"id\"}\n\n"
                f"Available skills:\n{listing}"
            )
            for t in SKILLS_UNDER_TEST:
                raw = query_llm(system, t["query"], model_name)
                parsed = clean_json(raw)
                chosen = None
                if parsed:
                    chosen = parsed.get("chosen_skill_id") or parsed.get("skill_id")
                ok = chosen == t["id"]
                correct += int(ok)
                total += 1
                time.sleep(0.2)
        rate = 100.0 * correct / total if total else 0.0
        acc_by_n.append(rate)
        print(f"  N={N:<5} selection_acc={rate:5.1f}%  ({correct}/{total})")
    return {"N": [n for n in sizes if n >= len(SKILLS_UNDER_TEST)], "selection_acc": acc_by_n}

# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true", help="Run real LLM selection accuracy vs N")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["deepseek-v4-flash", "gpt-4o-mini"],
        help="Models for --live (default: 2 strongest; pass 4 if you want full set)",
    )
    parser.add_argument("--trials", type=int, default=3, help="Trials per target per N")
    args = parser.parse_args()

    sizes = [15, 30, 60, 100, 200, 500, 1000]

    print("=" * 78)
    print("PHASE TRANSITION CHARACTERIZATION")
    print("Semantic density = REAL embeddings | Accuracy = REAL LLM calls if --live")
    print("=" * 78)

    fillers = load_fillers()
    if len(fillers) < 200:
        print("WARNING: few fillers; large N will repeat less diversity")

    print("\n[1/2] Semantic density (local, free)...")
    st_model = SentenceTransformer("all-MiniLM-L6-v2")
    semantic = compute_semantic_curve(st_model, fillers, sizes)

    live_results = {}
    if args.live:
        if not os.environ.get("OPENROUTER_API_KEY"):
            print("ERROR: OPENROUTER_API_KEY required for --live")
            sys.exit(1)
        print("\n[2/2] Live LLM selection accuracy...")
        # Optimal default: 2 models. User may pass 4.
        for m in args.models:
            live_results[m] = live_selection_accuracy(
                m, fillers, sizes, trials_per_target=args.trials, use_expert_text=True
            )
    else:
        print("\n[2/2] Skipped live LLM (--live not set). Semantic-only run.")

    out_dir = os.path.join(WORKSPACE_ROOT, "blackbox_os", "roles", "data_scientist", "workflows")
    os.makedirs(out_dir, exist_ok=True)
    out = {"semantic": semantic, "live_selection": live_results, "sizes": sizes}
    json_path = os.path.join(out_dir, "phase_transition_results.json")
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved → {json_path}")

    # Plot
    fig, ax1 = plt.subplots(figsize=(10, 5.8))
    ax1.plot(semantic["N"], semantic["bare_mean_max_cos"], "s--", color="#d97706", lw=2, label="Bare NN cosine")
    ax1.plot(semantic["N"], semantic["expert_mean_max_cos"], "o-", color="#2563eb", lw=2.3, label="Expert NN cosine")
    ax1.axhspan(0.53, 0.56, color="#ef4444", alpha=0.12, label="Hypothesized high-confusability band")
    ax1.set_xscale("log")
    ax1.set_xlabel("Catalog size N", fontweight="bold")
    ax1.set_ylabel("Mean max nearest-neighbor cosine", fontweight="bold")
    ax1.set_ylim(0.25, 0.85)
    ax1.grid(True, which="both", ls="--", alpha=0.4)

    if live_results:
        ax2 = ax1.twinx()
        colors = ["#16a34a", "#dc2626", "#9333ea", "#0891b2"]
        for i, (m, res) in enumerate(live_results.items()):
            short = m.split("/")[-1].replace(":free", "")
            ax2.plot(res["N"], res["selection_acc"], "D-", color=colors[i % len(colors)],
                     lw=2, label=f"{short} selection %")
        ax2.set_ylabel("Live selection accuracy (%)", fontweight="bold")
        ax2.set_ylim(0, 105)
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="best", fontsize=8)
    else:
        ax1.legend(loc="best", fontsize=9)

    title = "Semantic density vs N"
    if live_results:
        title += " + live selection accuracy"
    ax1.set_title(title)
    fig.tight_layout()
    img = os.path.join(out_dir, "semantic_density_vs_N.png")
    plt.savefig(img, dpi=300)
    plt.close()
    print(f"Saved figure → {img}")

if __name__ == "__main__":
    main()
