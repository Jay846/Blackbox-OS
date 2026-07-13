import json
import numpy as np
import matplotlib.pyplot as plt
import os
from sentence_transformers import SentenceTransformer

# 1. Load Data
targets_path = "./skill_experiment/targets_v4.json"
fillers_path = "./skill_experiment/fillers_v4.json"

targets = json.load(open(targets_path))
fillers = json.load(open(fillers_path))

print(f"Loaded {len(targets)} targets and {len(fillers)} fillers.")

# 2. LCG Shuffler matching runner_v4_flash.js exactly
def seeded_shuffle(arr, seed):
    a = list(arr)
    s = seed
    for i in range(len(a) - 1, 0, -1):
        s = (s * 9301 + 49297) % 233280
        j = int((s / 233280) * (i + 1))
        a[i], a[j] = a[j], a[i]
    return a

def build_library(size, targets, fillers_pool):
    seed = size * 13 + 7
    n_fillers = max(0, size - len(targets))
    shuffled_fillers = seeded_shuffle(fillers_pool, seed)
    fillers_selected = shuffled_fillers[:n_fillers]
    return seeded_shuffle(targets + fillers_selected, seed + 1)

# 3. Formatting
def format_skill(skill, expert):
    if not expert:
        return f"{skill['id']}: {skill['concept']}"
    disamb = skill.get('disambiguator', '') or ''
    ex = skill.get('example', '') or ''
    return f"{skill['id']}: {skill['concept']} {disamb} Example: \"{ex}\""

# 4. Load model
print("Loading all-MiniLM-L6-v2 model...")
model = SentenceTransformer('all-MiniLM-L6-v2')

# Target library sizes
sizes = [5, 20, 50, 100, 200, 300, 500, 1000, 2000]

# Find all unique skills to embed (targets + relevant fillers across all sizes)
all_unique_skills = {}
for size in sizes:
    lib = build_library(size, targets, fillers)
    for skill in lib:
        all_unique_skills[skill['id']] = skill

unique_skill_list = list(all_unique_skills.values())
print(f"Total unique skills across all configurations: {len(unique_skill_list)}")

# Compute embeddings
print("Computing embeddings for Bare formatting...")
bare_texts = [format_skill(s, False) for s in unique_skill_list]
bare_embeddings = model.encode(bare_texts, show_progress_bar=True, convert_to_numpy=True)
bare_emb_dict = {s['id']: emb for s, emb in zip(unique_skill_list, bare_embeddings)}

print("Computing embeddings for Expert formatting...")
expert_texts = [format_skill(s, True) for s in unique_skill_list]
expert_embeddings = model.encode(expert_texts, show_progress_bar=True, convert_to_numpy=True)
expert_emb_dict = {s['id']: emb for s, emb in zip(unique_skill_list, expert_embeddings)}

def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

# 5. Analysis
results = {}
for size in sizes:
    lib = build_library(size, targets, fillers)
    lib_ids = [s['id'] for s in lib]
    target_ids_in_lib = [s['id'] for s in targets if s['id'] in lib_ids]
    
    print(f"\nAnalyzing size N={size} (Targets: {len(target_ids_in_lib)}, Library Size: {len(lib_ids)})")
    
    for mode, emb_dict in [("bare", bare_emb_dict), ("expert", expert_emb_dict)]:
        max_sims = []
        mean_sims = []
        density_70 = []
        density_75 = []
        
        # Get embeddings for this library
        lib_embs = np.array([emb_dict[id_] for id_ in lib_ids])
        
        for t_id in target_ids_in_lib:
            t_emb = emb_dict[t_id]
            # Compute similarity to all other tools in the library
            sims = []
            for other_id in lib_ids:
                if other_id == t_id:
                    continue
                sims.append(cosine_similarity(t_emb, emb_dict[other_id]))
            
            sims = np.array(sims)
            max_sims.append(np.max(sims))
            mean_sims.append(np.mean(sims))
            density_70.append(np.sum(sims >= 0.70))
            density_75.append(np.sum(sims >= 0.75))
            
        results.setdefault(size, {})[mode] = {
            "avg_max_similarity": float(np.mean(max_sims)),
            "std_max_similarity": float(np.std(max_sims)),
            "avg_mean_similarity": float(np.mean(mean_sims)),
            "avg_density_70": float(np.mean(density_70)),
            "avg_density_75": float(np.mean(density_75))
        }
        
        print(f"  [{mode.upper()}] Avg Max Sim: {results[size][mode]['avg_max_similarity']:.4f}, Avg Mean Sim: {results[size][mode]['avg_mean_similarity']:.4f}, Density > 0.70: {results[size][mode]['avg_density_70']:.2f}")

# 6. Save results
output_json = "./skill_experiment/characterization_results.json"
with open(output_json, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved results to {output_json}")

# 7. Generate Plot
# Hardcoded accuracy for DeepSeek V4 Flash from results_v4_flash.json
v4_acc = {
    5: {"bare": 90.8, "expert": 93.1},
    20: {"bare": 93.7, "expert": 92.5},
    50: {"bare": 91.4, "expert": 96.0},
    100: {"bare": 90.2, "expert": 90.8},
    200: {"bare": 87.9, "expert": 90.2},
    300: {"bare": 91.4, "expert": 90.2},
    500: {"bare": 85.1, "expert": 87.9},
    1000: {"bare": 67.8, "expert": 87.4},
    2000: {"bare": 83.9, "expert": 81.0}
}

fig, ax1 = plt.subplots(figsize=(10, 6))

color = 'tab:blue'
ax1.set_xlabel('Library Size (N)', fontsize=12)
ax1.set_ylabel('Avg Max Cosine Similarity', color=color, fontsize=12)

# Extract max similarities
bare_max_sims = [results[s]["bare"]["avg_max_similarity"] for s in sizes]
expert_max_sims = [results[s]["expert"]["avg_max_similarity"] for s in sizes]

ax1.plot(sizes, bare_max_sims, 'o--', color='lightblue', label='Max Similarity (Bare)')
ax1.plot(sizes, expert_max_sims, 'o-', color='blue', label='Max Similarity (Expert)')
ax1.tick_params(axis='y', labelcolor=color)
ax1.set_xscale('log')
ax1.set_xticks(sizes)
ax1.get_xaxis().set_major_formatter(plt.ScalarFormatter())
ax1.legend(loc='upper left')

ax2 = ax1.twinx()  
color = 'tab:red'
ax2.set_ylabel('DeepSeek V4 Flash Accuracy (%)', color=color, fontsize=12)

bare_accs = [v4_acc[s]["bare"] for s in sizes]
expert_accs = [v4_acc[s]["expert"] for s in sizes]

ax2.plot(sizes, bare_accs, 's--', color='salmon', label='Accuracy (Bare)')
ax2.plot(sizes, expert_accs, 's-', color='red', label='Accuracy (Expert)')
ax2.tick_params(axis='y', labelcolor=color)
ax2.legend(loc='lower left')

plt.title('Semantic Packing Density & Max Similarity vs Routing Accuracy', fontsize=14, pad=15)
fig.tight_layout()

plot_path = "/Users/jaysalvi11/.gemini/antigravity/brain/606d300f-175e-4ed5-bb6e-de1f70f3b028/semantic_density_vs_accuracy.png"
plt.savefig(plot_path, dpi=300)
print(f"Saved plot to {plot_path}")
