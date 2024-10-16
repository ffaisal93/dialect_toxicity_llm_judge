import os
import json
from collections import Counter
import pandas as pd

# Directory containing the processed data
processed_data_dir = 'data/processed_data'

# Directory containing the results data
results_final_dir = 'results_final'

# Load language mapping from metadata
with open('metadata/lang_mapping.json', 'r') as mapping_file:
    lang_mapping = json.load(mapping_file)

# Iterate over each model in the results_final directory
sensitivity_stats_all_models = {}
for model_dir in os.listdir(results_final_dir):
    model_path = os.path.join(results_final_dir, model_dir)
    if os.path.isdir(model_path):
        sensitivity_stats = {}
        for filename in os.listdir(model_path):
            if filename.endswith('.json'):
                language_cluster = filename.replace('.json', '')
                file_path = os.path.join(model_path, filename)
                # Load the JSON data
                with open(file_path, 'r') as json_file:
                    data = json.load(json_file)
                    
                    # Iterate through each dialect
                    for dialect, entries in data.items():
                        # Map dialect name if it exists in lang_mapping
                        mapped_dialect = lang_mapping.get(dialect, dialect)

                        d0_count = 0
                        other_count = 0

                        # Count the occurrences of D0 and others
                        for entry in entries:
                            if isinstance(entry, dict) and 'Dialectal Sensitivity Impact' in entry:
                                impact = entry['Dialectal Sensitivity Impact']
                                if impact == 'D0':
                                    d0_count += 1
                                else:
                                    other_count += 1
                        
                        # Store the counts in sensitivity_stats
                        if language_cluster not in sensitivity_stats:
                            sensitivity_stats[language_cluster] = {}
                        sensitivity_stats[language_cluster][mapped_dialect] = {
                            'Other_percentage': (other_count * 100 / (d0_count + other_count)) if (d0_count + other_count) > 0 else 0
                        }

        # Store sensitivity stats for the model
        sensitivity_stats_all_models[model_dir] = sensitivity_stats

# Create latex_tables directory if not exists
latex_dir = 'latex_tables'
if not os.path.exists(latex_dir):
    os.makedirs(latex_dir)

# Combine sensitivity stats for all models into a DataFrame and save to LaTeX
combined_sensitivity_stats = {}
for model_name, stats in sensitivity_stats_all_models.items():
    for lang, dialect_data in stats.items():
        for dialect, values in dialect_data.items():
            if (lang, dialect) not in combined_sensitivity_stats:
                combined_sensitivity_stats[(lang, dialect)] = {'Language': lang, 'Dialect': dialect}
            combined_sensitivity_stats[(lang, dialect)][f'Other_percentage ({model_name})'] = values['Other_percentage']

combined_sensitivity_stats_df = pd.DataFrame.from_dict(combined_sensitivity_stats, orient='index')
combined_sensitivity_stats_df.reset_index(drop=True, inplace=True)
combined_sensitivity_stats_df.set_index(['Language', 'Dialect'], inplace=True)

# Add average column
combined_sensitivity_stats_df['Average'] = combined_sensitivity_stats_df.mean(axis=1)

# Sort each language group by ascending order based on the average value
combined_sensitivity_stats_df = combined_sensitivity_stats_df.sort_values(by=['Language', 'Average'], ascending=[True, True])

# Add average row
average_row = combined_sensitivity_stats_df.mean(axis=0).to_dict()
average_row_df = pd.DataFrame([average_row], index=pd.MultiIndex.from_tuples([('Average', '')], names=['Language', 'Dialect']))
combined_sensitivity_stats_df = pd.concat([combined_sensitivity_stats_df, average_row_df])

latex_file_path = os.path.join(latex_dir, "sensitivity_stats_all_models.tex")
with open(latex_file_path, 'w') as latex_file:
    latex_file.write(combined_sensitivity_stats_df.to_latex(multicolumn=True, multirow=True, escape=True, float_format="%.2f"))