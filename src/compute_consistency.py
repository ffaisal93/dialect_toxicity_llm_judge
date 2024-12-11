import os
import json
import pandas as pd
import numpy as np

# Load language mapping from metadata/lang_mapping.json
with open('metadata/lang_mapping.json', 'r') as f:
    lang_mapping = json.load(f)

# Directory containing the evaluation scores
evaluation_scores_dir = 'evaluation_scores'

# Define standard dialect mapping based on underlined varieties in LaTeX table
standard_dialects = {
    'arabic': 'Standard Arabic',
    'bengali': 'Standard',
    'chinese': 'Cantonese',
    'finnish': 'Finnish',
    'kurdish': 'Central Kurdish',
    'norwegian': 'Norwegian Bokmal',
    'high_german': 'Latvian',
    'english': 'Standard',
    'sotho-tswana': 'Northern Sotho',
    'common_turkic': 'Central Oghuz'
}

# Reverse lang_mapping to get dialect keys from values
reverse_lang_mapping = {v.lower(): k for k, v in lang_mapping.items()}

# Compute consistency scores
consistency_records = []
all_dialectal_consistency_records = []
for filename in os.listdir(evaluation_scores_dir):
    if filename.endswith('.json'):
        model_name = filename.replace('.json', '')
        file_path = os.path.join(evaluation_scores_dir, filename)

        # Load the JSON data
        with open(file_path, 'r') as json_file:
            result_data = json.load(json_file)

        # Prepare data for consistency scores
        llm_human_consistency = []
        multilingual_variance = []
        dialectal_consistency = []
        dialectal_consistency_records = []

        for language_cluster, dialects in result_data.items():
            print(language_cluster, dialects.keys())
            # Get standard dialect key from lang_mapping or use as is if already in key format
            standard_dialect_value = None
            for key in dialects.keys():
                if 'standard' == key.lower():
                    standard_dialect_value = 'standard'
                    standard_dialect = standard_dialect_value
            if standard_dialect_value is None:
                standard_dialect = reverse_lang_mapping.get(standard_dialects[language_cluster].lower(), standard_dialect_value)
            standard_accuracy = None
            dialect_accuracies = []

            for dialect, metrics in dialects.items():
                print(dialect,standard_dialect)
                accuracy_3_bin = metrics['accuracy_3_bins']
                llm_human_consistency.append(accuracy_3_bin)
                if dialect == standard_dialect:
                    standard_accuracy = accuracy_3_bin
                else:
                    dialect_accuracies.append(accuracy_3_bin)

            if standard_accuracy is not None:
                multilingual_variance.append(standard_accuracy)
                for dialect_accuracy in dialect_accuracies:
                    dialectal_consistency.append(abs(standard_accuracy - dialect_accuracy))

            # Calculate Dialectal Consistency for each language group
            if standard_accuracy is not None and len(dialect_accuracies) > 0:
                avg_dialectal_diff = np.mean([abs(standard_accuracy - acc) for acc in dialect_accuracies])
                C_dl = 1 - avg_dialectal_diff
                dialectal_consistency_records.append([language_cluster.capitalize(), model_name, round(C_dl * 100, 2)])

        all_dialectal_consistency_records.extend(dialectal_consistency_records)
        # print(all_dialectal_consistency_records)

        # Calculate LLM-Human Consistency
        C_lh = np.mean(llm_human_consistency) * 100
        # print(multilingual_variance)

        # Calculate Multilingual Consistency
        if len(multilingual_variance) > 1:
            variance = np.var(multilingual_variance)
            C_ml = (1 - variance) * 100
        else:
            C_ml = 0

        # Calculate Dialectal Consistency
        if len(dialectal_consistency) > 0:
            avg_dialectal_diff = np.mean(dialectal_consistency)
            C_dl = (1 - avg_dialectal_diff) * 100
        else:
            C_dl = 0

        # Append consistency scores
        consistency_records.append([
            model_name, round(C_lh, 2), round(C_ml, 2), round(C_dl, 2)
        ])

# # Create DataFrame for consistency scores
# consistency_df = pd.DataFrame(consistency_records, columns=['Model', 'LLM-Human Consistency', 'Multilingual Consistency', 'Dialectal Consistency'])

# # Add average row to consistency_df
# average_row = ['Average'] + [round(consistency_df[col].mean(), 2) for col in consistency_df.columns if col != 'Model']
# consistency_df.loc['Average'] = average_row

# # Save consistency scores to LaTeX table
# latex_table_path = os.path.join('latex_tables', 'consistency_scores.tex')
# with open(latex_table_path, 'w') as f:
#     f.write(consistency_df.to_latex(index=False, escape=True, caption='Consistency Scores for Each Model', label='tab:consistency_scores', float_format="{:.2f}".format))

# # Create DataFrame for dialectal consistency scores for each model
# dialectal_consistency_df = pd.DataFrame(all_dialectal_consistency_records, columns=['Language Cluster', 'Model', 'Dialectal Consistency'])

# # Pivot the dialectal consistency DataFrame to have models as columns
# dialectal_consistency_df = dialectal_consistency_df.pivot(index='Language Cluster', columns='Model', values='Dialectal Consistency')

# # Add average row to dialectal_consistency_df
# average_row = [round(dialectal_consistency_df[col].mean(), 2) for col in dialectal_consistency_df.columns]
# dialectal_consistency_df.loc['Average'] = average_row
# # Save dialectal consistency scores to LaTeX table
# dialectal_latex_table_path = os.path.join('latex_tables', 'dialectal_consistency_scores.tex')
# with open(dialectal_latex_table_path, 'w') as f:
#     f.write(dialectal_consistency_df.to_latex(caption='Dialectal Consistency Scores for Each Language Group and Model', label='tab:dialectal_consistency_scores', escape=True, float_format="{:.1f}".format))