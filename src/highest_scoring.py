import json
import pandas as pd
import os

# Load the highest scoring dialects dictionary
highest_scoring_dialects_path = 'latex_tables/highest_scoring_dialects.json'
with open(highest_scoring_dialects_path, 'r') as json_file:
    highest_scoring_dialects_per_model = json.load(json_file)

# Define standard dialect mapping based on underlined varieties in LaTeX table
standard_dialects = {
    'arabic': 'Standard Arabic',
    'bengali': 'Standard',
    'chinese': 'Cantonese',
    'finnish': 'Finnish',
    'kurdish': 'Central Kurdish',
    'norwegian': 'Norwegian Bokmal',
    'latvian': 'Latvian',
    'english': 'Standard',
    'sotho-tswana': 'Northern Sotho',
    'common\_turkic': 'Central Oghuz'
}

# Prepare records for LaTeX table
records = []
clusters = set()

# Extract all language clusters
for model, clusters_data in highest_scoring_dialects_per_model.items():
    clusters.update(clusters_data.keys())

# Sort clusters alphabetically
clusters = sorted(clusters)

# Create records for each language cluster
standard_counts = {'NeMo': 0, 'Aya': 0, 'Phi': 0}
for cluster in clusters:
    cluster_data = [cluster]
    for model, model_display_name in zip(['Mistral-NeMo-Minitron-8B-Instruct', 'aya-23-8B', 'Phi-3-mini-4k-instruct'], ['NeMo', 'Aya', 'Phi']):
        if cluster in highest_scoring_dialects_per_model[model]:
            dialect = highest_scoring_dialects_per_model[model][cluster]['Dialect']
            if dialect.lower() == standard_dialects.get(cluster.lower(), '').lower():
                cluster_data.append(f'\\underline{{{dialect}}}')
                standard_counts[model_display_name] += 1
            else:
                cluster_data.append(dialect)
        else:
            cluster_data.append('-')
    records.append(cluster_data)

# Create DataFrame
columns = ['Language Cluster', 'NeMo', 'Aya', 'Phi']
df = pd.DataFrame(records, columns=columns)

# Capitalize Language Cluster names
df['Language Cluster'] = df['Language Cluster'].str.capitalize()

# Add percentage of standard dialect for each model as the last row
total_clusters = len(clusters)
percentages = [
    'Percentage of Cluster Representative',
    f'{(standard_counts["NeMo"] / total_clusters) * 100:.2f}\%',
    f'{(standard_counts["Aya"] / total_clusters) * 100:.2f}\%',
    f'{(standard_counts["Phi"] / total_clusters) * 100:.2f}\%'
]
df.loc[len(df)] = percentages

# Save LaTeX table to file
latex_tables_dir = 'latex_tables'
os.makedirs(latex_tables_dir, exist_ok=True)
latex_table_path = os.path.join(latex_tables_dir, 'highest_scoring_dialects_table.tex')
with open(latex_table_path, 'w') as f:
    f.write(df.to_latex(index=False, escape=False, caption="Comparison of Highest Scoring Dialects and Cluster Representative Dialects for Each Language Cluster. The highest performing variety for each cluster is not always the cluster representative, which generally represents the high-resource or standard variety. Underlined dialects indicate the cluster representative.", label="tab:highest_scoring_dialects", float_format="{:.2f}".format))
print(df)
