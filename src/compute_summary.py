import os
import json
import pandas as pd
import numpy as np

# Load language mapping from metadata/lang_mapping.json
with open('metadata/lang_mapping.json', 'r') as f:
    lang_mapping = json.load(f)

# Directory containing the evaluation scores
evaluation_scores_dir = 'evaluation_scores'
latex_tables_dir = 'latex_tables'

# Create latex_tables directory if it doesn't exist
os.makedirs(latex_tables_dir, exist_ok=True)

# Iterate over each model evaluation file in the evaluation_scores directory
for filename in os.listdir(evaluation_scores_dir):
    if filename.endswith('.json'):
        model_name = filename.replace('.json', '')
        file_path = os.path.join(evaluation_scores_dir, filename)

        # Load the JSON data
        with open(file_path, 'r') as json_file:
            result_data = json.load(json_file)

        # Prepare data for LaTeX table
        records = []
        for language_cluster, dialects in result_data.items():
            language_cluster = language_cluster.replace('_', '\_')  # Replace underscores for LaTeX compatibility
            for dialect, metrics in dialects.items():
                dialect_name = lang_mapping.get(dialect, dialect).replace('_', '\_')
                records.append([
                    language_cluster.capitalize(),
                    dialect_name.capitalize(),
                    metrics['accuracy_3_bins'] * 100,
                    metrics['accuracy_5_bins'] * 100,
                    metrics['f1_3_bins'] * 100,
                    metrics['f1_5_bins'] * 100
                ])

        df = pd.DataFrame(records, columns=[
            'Language Cluster', 'Dialect', 'ACC(bin=3)', 'ACC(bin=5)', 'F1(bin=3)', 'F1(bin=5)'
        ])

        # Normalize and round values
        df[['ACC(bin=3)', 'ACC(bin=5)', 'F1(bin=3)', 'F1(bin=5)']] = df[
            ['ACC(bin=3)', 'ACC(bin=5)', 'F1(bin=3)', 'F1(bin=5)']].round(1)

        # Set multi-level index
        df.set_index(['Language Cluster', 'Dialect'], inplace=True)

        # Calculate average row
        avg_values = df.mean().round(1)
        avg_row = pd.DataFrame([['Average', '-', avg_values['ACC(bin=3)'], avg_values['ACC(bin=5)'], avg_values['F1(bin=3)'], avg_values['F1(bin=5)']]],
                               columns=df.reset_index().columns).set_index(['Language Cluster', 'Dialect'])
        df = pd.concat([df, avg_row])

        # Save LaTeX table to file
        latex_table_path = os.path.join(latex_tables_dir, f'{model_name}_table.tex')
        with open(latex_table_path, 'w') as f:
            f.write(df.to_latex(multirow=True, caption=f"Evaluation Results for {model_name}", label=f"tab:{model_name}", float_format="{:.1f}".format))

# Create summary table with average F1 scores for each language cluster and model
summary_records = []
for filename in os.listdir(evaluation_scores_dir):
    if filename.endswith('.json'):
        model_name = filename.replace('.json', '')
        file_path = os.path.join(evaluation_scores_dir, filename)

        # Load the JSON data
        with open(file_path, 'r') as json_file:
            result_data = json.load(json_file)

        # Prepare data for summary
        for language_cluster, dialects in result_data.items():
            language_cluster = language_cluster.replace('_', '\_')  # Replace underscores for LaTeX compatibility
            f1_3_list = []
            f1_5_list = []
            for metrics in dialects.values():
                f1_3_list.append(metrics['f1_3_bins'] * 100)
                f1_5_list.append(metrics['f1_5_bins'] * 100)

            avg_f1_3 = np.mean(f1_3_list)
            avg_f1_5 = np.mean(f1_5_list)

            summary_records.append([
                language_cluster.capitalize(),
                model_name,
                round(avg_f1_3, 1),
                round(avg_f1_5, 1)
            ])

summary_df = pd.DataFrame(summary_records, columns=[
    'Language Cluster', 'Model', 'F1(bin=3)', 'F1(bin=5)'
])

# Pivot the summary table to have models as columns with multi-level columns
summary_pivot = summary_df.pivot(index='Language Cluster', columns='Model', values=['F1(bin=3)', 'F1(bin=5)'])
# Set the multi-level column index and reverse the pairs
# summary_pivot.columns = pd.MultiIndex.from_tuples([(col[1], col[0]) for col in summary_pivot.columns], names=['Model', 'Metric'])

# Save summary LaTeX table to file
summary_latex_table_path = os.path.join(latex_tables_dir, 'summary_models_table.tex')
with open(summary_latex_table_path, 'w') as f:
    f.write(summary_pivot.to_latex(
        multicolumn=True, multirow=True, caption="Summary of Evaluation Results for All Models (F1 Scores Only)",
        label="tab:summary_models", float_format="{:.1f}".format
    ))