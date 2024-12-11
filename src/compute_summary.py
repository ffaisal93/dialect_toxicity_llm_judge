import os
import json
import pandas as pd
import numpy as np
import pickle

# Load language mapping from metadata/lang_mapping.json
with open('metadata/lang_mapping.json', 'r') as f:
    lang_mapping = json.load(f)

# Directory containing the evaluation scores
evaluation_scores_dir = 'evaluation_scores'
latex_tables_dir = 'latex_tables'

# Create latex_tables directory if it doesn't exist
os.makedirs(latex_tables_dir, exist_ok=True)


def model_table_generation():
    # Dictionary to capture the highest scoring dialect for each model
    highest_scoring_dialects_per_model = {}

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
            highest_scoring_dialects = {}
            for language_cluster, dialects in result_data.items():
                language_cluster = language_cluster.replace('_', '\_')  # Replace underscores for LaTeX compatibility
                language_cluster = language_cluster.replace('high\_german', 'latvian')  # Replace underscores for LaTeX compatibility
                print(language_cluster)
                highest_score = -1
                highest_scoring_dialect = None
                for dialect, metrics in dialects.items():
                    dialect_name = lang_mapping.get(dialect, dialect).replace('_', '\_')
                    records.append([
                        language_cluster.capitalize(),
                        dialect_name.capitalize(),
                        metrics['f1'] * 100,
                        metrics['rmse_similarity'] * 100,
                        metrics['spearman_corr']
                    ])

                    # Update highest scoring dialect for the current language cluster
                    if metrics['rmse_similarity'] > highest_score:
                        highest_scoring_dialect = dialect_name
                        highest_score = metrics['rmse_similarity']

                # Store the highest scoring dialect for the current language cluster
                if highest_scoring_dialect:
                    highest_scoring_dialects[language_cluster] = {
                        'Dialect': highest_scoring_dialect,
                        'Highest RMSE_SIM Score': highest_score * 100
                    }

            # Store the highest scoring dialects for the model
            highest_scoring_dialects_per_model[model_name] = highest_scoring_dialects

            df = pd.DataFrame(records, columns=[
                'Language Cluster', 'Dialect', 'F1', 'RMSE-SIM', 'RANK-CORR'])

            # Normalize and round values
            df[['F1', 'RMSE-SIM', 'RANK-CORR']] = df[
                ['F1', 'RMSE-SIM', 'RANK-CORR']].round(1)

            df=df.replace('High\_german','Latvian')
            # Set multi-level index
            df.set_index(['Language Cluster', 'Dialect'], inplace=True)

            # Calculate average row
            avg_values = df.mean().round(1)
            avg_row = pd.DataFrame(
                [[avg_values['F1'], avg_values['RMSE-SIM'], avg_values['RANK-CORR']]],
                columns=['F1', 'RMSE-SIM', 'RANK-CORR'],
                index=pd.MultiIndex.from_tuples([('Average', 'Average')], names=['Language Cluster', 'Dialect'])
            )


            # Sort clusters based on max ACC(bin=3) value for each cluster
            cluster_max_acc = df.groupby(level='Language Cluster')['RMSE-SIM'].max()
            df = df.reindex(cluster_max_acc.sort_values(ascending=False).index, level='Language Cluster')
            # Concatenate the average row with the existing DataFrame
            df = pd.concat([df, avg_row])

            # Save LaTeX table to file
            latex_table_path = os.path.join(latex_tables_dir, f'{model_name}_table.tex')
            with open(latex_table_path, 'w') as f:
                f.write(df.to_latex(multirow=True, caption=f"Evaluation Results for {model_name}", label=f"tab:{model_name}", float_format="{:.2f}".format))

    # Save the highest scoring dialects dictionary as a JSON file
    highest_scoring_dialects_path = os.path.join(latex_tables_dir, 'highest_scoring_dialects.json')
    with open(highest_scoring_dialects_path, 'w') as json_file:
        json.dump(highest_scoring_dialects_per_model, json_file, indent=4)


def summary_table_generation():
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
                f1 = []
                rmse_sim = []
                for metrics in dialects.values():
                    f1.append(metrics['f1'] * 100)
                    rmse_sim.append(metrics['rmse_similarity'] * 100)

                avg_f1 = np.mean(f1)
                avg_rmse_sim = np.mean(rmse_sim)

                summary_records.append([
                    language_cluster.capitalize(),
                    model_name,
                    round(avg_f1, 1),
                    round(avg_rmse_sim, 1)
                ])

    summary_df = pd.DataFrame(summary_records, columns=[
        'Language Cluster', 'Model', 'F1', 'RMSE-SIM'
    ])

    summary_df=summary_df.replace('High\_german','Latvian')
    # Pivot the summary table to have models as columns with multi-level columns
    summary_pivot = summary_df.pivot(index='Language Cluster', columns='Model', values=['F1', 'RMSE-SIM'])

    # Calculate average row for summary pivot
    summary_pivot.loc['Average'] = summary_pivot.mean()

    # Sort values by average row values in descending order (excluding the average row itself)
    sorted_summary_pivot = summary_pivot.drop(index='Average').sort_values(by=('RMSE-SIM', summary_pivot.columns.levels[1][0]), ascending=False)
    summary_pivot = pd.concat([sorted_summary_pivot, summary_pivot.loc[['Average']]])

    # Save summary dataframe table to file
    summary_csv_table_path = os.path.join(latex_tables_dir, 'summary_models_table.pkl')
    summary_pivot.to_pickle(summary_csv_table_path)
    # Save summary LaTeX table to file
    summary_latex_table_path = os.path.join(latex_tables_dir, 'summary_models_table.tex')
    with open(summary_latex_table_path, 'w') as f:
        f.write(summary_pivot.to_latex(
            multicolumn=True, multirow=True, caption="Summary of Evaluation Results for All Models (F1 Scores Only)",
            label="tab:summary_models", float_format="{:.1f}".format
        ))



# Example usage
if __name__ == "__main__":
    model_table_generation()
    summary_table_generation()