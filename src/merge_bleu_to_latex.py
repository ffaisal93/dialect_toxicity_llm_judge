import os
import json
import pandas as pd

# === File paths ===
nllb_path = "evaluation_scores/nllb_scores.json"
gpt_path = "evaluation_scores/nllb_gpt_assisted_scores.json"
lang_map_path = "metadata/lang_mapping.json"
data_path = "data/processed_data"

# === Load data ===
with open(nllb_path, "r", encoding="utf-8") as f:
    nllb_data = json.load(f)
with open(gpt_path, "r", encoding="utf-8") as f:
    gpt_data = json.load(f)
with open(lang_map_path, "r", encoding="utf-8") as f:
    lang_mapping = json.load(f)

# === Build cluster -> variety mapping ===
lang_clusters = {}
for file in os.listdir(data_path):
    if file.endswith(".json"):
        cluster = file.replace(".json", "")
        with open(os.path.join(data_path, file), "r", encoding="utf-8") as f:
            data = json.load(f)
        lang_clusters[cluster] = list(data.keys())

# === Build table rows ===
multi_rows = []

for cluster, varieties in lang_clusters.items():
    cluster_rows = []

    for variety in varieties:
        full_name = lang_mapping.get(variety, variety)
        nllb_score = round(nllb_data["dialect_scores"].get(variety, 0), 2)
        gpt_score = round(gpt_data["dialect_scores"].get(variety, 0), 2)

        # Discard row if both scores are zero
        if nllb_score == 0 and gpt_score == 0:
            continue

        # Compute delta
        if nllb_score == 0:
            delta_fmt = ""
        else:
            delta = ((gpt_score - nllb_score) / nllb_score) * 100
            if delta < 0:
                delta_fmt = ""
            # elif delta >= 10:
            #     delta_fmt = f"\\textbf{{{delta:.1f}}}"
            else:
                delta_fmt = f"{delta:.1f}"

        cluster_rows.append(((cluster.capitalize(), full_name), [nllb_score, gpt_score, delta_fmt]))

    # Only add non-empty clusters
    multi_rows.extend(cluster_rows)

# === Create DataFrame ===
index = pd.MultiIndex.from_tuples([row[0] for row in multi_rows], names=["Language Cluster", "Language Variety"])
df = pd.DataFrame([row[1] for row in multi_rows], index=index, columns=["NLLB", "NLLB+GPT", "$\Delta$ (\%)"])

# === Add Global row ===
nllb_global = round(nllb_data["global_bleu"], 2)
gpt_global = round(gpt_data["global_bleu"], 2)

if nllb_global == 0:
    global_delta_fmt = ""
else:
    global_delta = ((gpt_global - nllb_global) / nllb_global) * 100
    if global_delta < 0:
        global_delta_fmt = ""
    # elif global_delta >= 10:
    #     global_delta_fmt = f"\\textbf{{{global_delta:.1f}}}"
    else:
        global_delta_fmt = f"{global_delta:.1f}"

global_row = pd.DataFrame(
    [[nllb_global, gpt_global, global_delta_fmt]],
    index=pd.MultiIndex.from_tuples([("Global", "")], names=["Language Cluster", "Language Variety"]),
    columns=["NLLB", "NLLB+GPT", "$\Delta$ (\%)"]
)
df = pd.concat([df, global_row])

# === Convert to LaTeX ===
latex = df.to_latex(
    escape=False,
    multicolumn=True,
    multirow=True,
    caption=(
    "BLEU scores by language variety grouped by language clusters. "
    "Bolded $\\Delta$ (\\%) values indicate $\\geq$ 20\\% improvement. "
    "Negative changes are omitted."
),
    label="tab:bleu_scores_grouped",
    column_format="llccc"
)

print(latex)
