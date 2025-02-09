import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, f1_score
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error
from sklearn.metrics import mean_absolute_error, cohen_kappa_score
import numpy as np

def assign_bins(values, num_bins):
    """
    Dynamically assign values to bins with fixed integer edges.

    Args:
    - values (list of float): Values to be binned.
    - num_bins (int): Number of bins.

    Returns:
    - list of int: The bin indices for each value.
    """
    # Use integer bin edges
    bin_edges = np.arange(1, num_bins + 1)  # [1, 2, 3, 4, ..., num_bins + 1]

    bins = []
    for val in values:
        for i in range(len(bin_edges)):
            # First bin includes only val == 1.0
            if i == 0 and val == bin_edges[i]:
                bins.append(1)
                break
            # Other bins handle > lower bound and <= upper bound
            elif bin_edges[i] < val <= bin_edges[i + 1]:
                bins.append(i+2)
                break
    return bins



def compute_metrics(filtered_ground_truth, filtered_predictions):
    """
    Compute accuracy, F1-score, RMSE-based similarity, MAE-based similarity, Spearman's Rank Correlation, 
    and Quadratic Weighted Kappa given filtered ground truth and predictions.

    Args:
    - filtered_ground_truth (list of int): Binned ground truth values.
    - filtered_predictions (list of int): Binned prediction values.

    Returns:
    - metrics (dict): A dictionary containing accuracy, F1-score, RMSE-based similarity, MAE-based similarity,
                      Spearman's Rank Correlation, and Quadratic Weighted Kappa.
    """
    # Calculate accuracy and F1-score for classification performance
    accuracy = accuracy_score(filtered_ground_truth, filtered_predictions)
    f1 = f1_score(filtered_ground_truth, filtered_predictions, average='weighted')
    
    # Calculate RMSE and convert to similarity measure
    rmse = mean_squared_error(filtered_ground_truth, filtered_predictions, squared=False)
    rmse_sim = 1 - (rmse / 4)  # Normalizing RMSE by max possible error (4) and converting to similarity
    
    # Calculate MAE and convert to similarity measure
    mae = mean_absolute_error(filtered_ground_truth, filtered_predictions)
    mae_sim = 1 - (mae / 4)  # Normalizing MAE by max possible error (4) and converting to similarity
    
    # Calculate Spearman's Rank Correlation for monotonic relationships
    spearman_corr, _ = spearmanr(filtered_ground_truth, filtered_predictions)
    
    # Calculate Quadratic Weighted Kappa (Cohen's Kappa with quadratic weights)
    kappa = cohen_kappa_score(filtered_ground_truth, filtered_predictions, weights='quadratic')
    
    # Return all metrics in a dictionary
    return {
        'accuracy': accuracy,
        'f1': f1,
        'rmse_similarity': rmse_sim,
        'mae_similarity': mae_sim,
        'spearman_corr': spearman_corr,
        'quadratic_weighted_kappa': kappa
    }




def compute_bins_and_metrics(ground_truth, predictions, num_bins):
    """
    Compute bins for ground truth and predictions, then calculate accuracy and F1-score.
    
    Args:
    - ground_truth (list of float): Ground truth values.
    - predictions (list of dict): Model predictions with 'Toxicity' key.
    - num_bins (int): The number of bins to divide the data into.
    - result_dict (dict): Dictionary to store the results.
    - lang (str): Language key for result_dict.
    - dialect (str): Dialect key for result_dict.
    """
    # Bin ground truth based on the number of bins
    
    # ground_truth_bins = [int(round((val - 1) * (num_bins - 1) / (5 - 1))) + 1 for val in ground_truth]
    ground_truth_bins = assign_bins(ground_truth, num_bins)
    predictions_bins = []

    # Map predictions to bins
    for pred in predictions:
        if isinstance(pred, dict) and 'Toxicity' in pred:
            toxicity = pred['Toxicity']
            if toxicity in ['S1']:
                predictions_bins.append(1)
            elif toxicity in ['S2']:
                predictions_bins.append(2 if num_bins > 2 else 1)
            elif toxicity == 'S3':
                predictions_bins.append(3 if num_bins > 3 else (2 if num_bins > 2 else 1))
            elif toxicity in ['S4', 'S5']:
                predictions_bins.append(num_bins)
        else:
            predictions_bins.append(None)

    # Filter out None values for valid comparisons
    filtered_data = [(g, p) for g, p in zip(ground_truth_bins, predictions_bins) if p is not None]
    if len(filtered_data) > 0:
        filtered_ground_truth, filtered_predictions = zip(*filtered_data)
        scores = compute_metrics(filtered_ground_truth, filtered_predictions)
        accuracy,f1=scores['accuracy'],scores['f1']
    else:
        scores = {
        'accuracy': 0,
        'f1': 0,
        'rmse_similarity': 0,
        'mae_similarity': 0,
        'spearman_corr': 0,
        'quadratic_weighted_kappa': 0
    }

    return scores,filtered_ground_truth, filtered_predictions



def evaluate_predictions(ground_truth, predictions, data_values, lang, dialect, result_dict, evaluation_stats):
    # Initial checks
    if not ground_truth or not predictions or not data_values:
        print(f"Skipping evaluation for {lang} - {dialect} due to missing data.")
        return
    if len(predictions) < len(data_values):
        data_values = data_values[:len(predictions)]
        ground_truth = ground_truth[:len(predictions)]
    
    # Discard indexes where data value is ''
    valid_indexes = [i for i, val in enumerate(data_values) if val != '']
    ground_truth = [ground_truth[i] for i in valid_indexes]
    predictions = [predictions[i] for i in valid_indexes]


    num_bins=5
    scores,filtered_ground_truth, filtered_predictions_5= compute_bins_and_metrics(ground_truth, predictions, num_bins)

    result_dict[lang][dialect] = scores




    # Store evaluation statistics
    evaluation_stats[(lang, dialect)] = {
        'original_count': len(predictions),
        'valid_count': len(filtered_predictions_5),
        'valid_percentage': len(filtered_predictions_5) * 100 / len(predictions) if len(predictions) > 0 else 0
    }
    print(lang, dialect,result_dict[lang][dialect])


def make_figure(language_avg_stats,latex_dir):
    # Create a bar plot for valid percentage output
    # Data preparation
    languages = language_avg_stats.index.str.replace('High_german', 'Latvian').str.capitalize()
    languages = [x.replace('High_german', 'Latvian').replace('_',' ').replace('Average',r'$\mathbf{Average}$') for x in list(languages)]

    print(list(languages))
    phi3_valid = language_avg_stats['Valid Percentage (Phi-3-mini-4k-instruct)']
    aya_valid = language_avg_stats['Valid Percentage (aya-23-8B)']
    nemo_valid = language_avg_stats['Valid Percentage (Mistral-NeMo-Minitron-8B-Instruct)']

    # Plotting
    fig, ax = plt.subplots(figsize=(15, 8))
    x = range(len(languages))
    width = 0.25

    # Plot bars for each model
    
    phi3_bars = ax.bar([pos - width for pos in x], phi3_valid, width, label='Phi-3', color='#d62728', alpha=0.6, edgecolor='black')
    aya_bars = ax.bar(x, aya_valid, width, label='Aya-23', color='#9467bd', alpha=0.6, edgecolor='black')
    nemo_bars = ax.bar([pos + width for pos in x], nemo_valid, width, label='NeMo', color='#8c564b', alpha=0.6, edgecolor='black')


    # Add value text on top of each bar
    for bars in [phi3_bars, aya_bars, nemo_bars]:
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, yval + 1, f'{yval:.1f}', ha='center', va='bottom', fontsize=9)

    # Labels and title
    ax.set_xlabel('Language Cluster', fontsize=16)
    ax.set_ylabel('Valid Formatted Output (%)', fontsize=12)
    ax.set_title('Percentage of Valid Formatted Output by Language Cluster and Model', fontsize=18)
    ax.set_xticks(x)
    ax.set_xticklabels(languages, rotation=45, ha='right', fontsize=14)

    # Add legend
    # Place legend in the empty space
    ax.legend(title='Model',loc='upper center', bbox_to_anchor=(0.5, -0.6), ncol=3, fontsize=12)

    # Adjust layout and save plot
    plt.tight_layout()
    plt.savefig(os.path.join(latex_dir, 'valid_percentage_bar_plot.pdf'))
    plt.show()

def create_latex_tables(evaluation_stats_all_models,latex_dir):
    # Combine evaluation stats for all models into a DataFrame and save to LaTeX
    combined_evaluation_stats = {}
    for model_name, stats in evaluation_stats_all_models.items():
        for (lang, dialect), values in stats.items():
            if (lang, dialect) not in combined_evaluation_stats:
                combined_evaluation_stats[(lang, dialect)] = {'Language': lang, 'Dialect': dialect}
            combined_evaluation_stats[(lang, dialect)][f'Valid Percentage ({model_name})'] = values['valid_percentage']

    combined_evaluation_stats_df = pd.DataFrame.from_dict(combined_evaluation_stats, orient='index')
    combined_evaluation_stats_df.reset_index(drop=True, inplace=True)
    latex_file_path = os.path.join(latex_dir, "evaluation_stats_all_models.tex")
    with open(latex_file_path, 'w') as latex_file:
        latex_file.write(combined_evaluation_stats_df.to_latex(index=False,escape=True,float_format="%.1f"))

    # Group by language and calculate the average percentage for each language
    language_avg_stats = combined_evaluation_stats_df.groupby('Language').mean(numeric_only=True)
    language_avg_stats.loc['Average'] = language_avg_stats.mean()

    # Sort by NeMo values
    language_avg_stats_no_avg = language_avg_stats.drop(index='Average')
    language_avg_stats_no_avg = language_avg_stats_no_avg.sort_values(by='Valid Percentage (aya-23-8B)', ascending=False)
    language_avg_stats = pd.concat([language_avg_stats_no_avg, language_avg_stats.loc[['Average']]])

    avg_latex_file_path = os.path.join(latex_dir, "evaluation_avg_stats_all_models.tex")
    with open(avg_latex_file_path, 'w') as avg_latex_file:
        avg_latex_file.write(language_avg_stats.to_latex(escape=True, float_format="%.1f"))

# Example usage
if __name__ == "__main__":
    df = pd.read_parquet("data/toxigen/test-00000-of-00001.parquet")
    ground_truth = list(df['intent'])

    # Directory containing the processed data
    processed_data_dir = 'data/processed_data'

    # Iterate over each file in the processed data directory
    result_data = {}
    data_values = {}
    for filename in os.listdir(processed_data_dir):
        if filename.endswith('.json'):
            language_cluster = filename.replace('.json', '')
            file_path = os.path.join(processed_data_dir, filename)
            # Load the JSON data
            with open(file_path, 'r') as json_file:
                data = json.load(json_file)
                data_values[language_cluster] = data
                print(f"{filename}: {len(data)} fields")

    # Directory containing the results data
    results_final_dir = 'results_final'
    
    # Iterate over each model in the results_final directory
    evaluation_stats_all_models = {}
    for model_dir in os.listdir(results_final_dir):
        model_path = os.path.join(results_final_dir, model_dir)
        if os.path.isdir(model_path):
            result_predictions = {}
            for filename in os.listdir(model_path):
                if filename.endswith('.json'):
                    language_cluster = filename.replace('.json', '')
                    file_path = os.path.join(model_path, filename)
                    # Load the JSON data
                    with open(file_path, 'r') as json_file:
                        data = json.load(json_file)
                        result_predictions[language_cluster] = data
                        print(f"{model_dir} {filename}: {len(data)} fields")

            # Initialize result dictionary
            result_dict = {lang: {dialect: {} for dialect in dialect_data} for lang, dialect_data in data_values.items()}
            evaluation_stats = {}

            # Compute accuracy and F1 for all languages and dialects
            for lang, dialect_data in data_values.items():
                for dialect, data_list in dialect_data.items():
                    if lang in result_predictions and dialect in result_predictions[lang]:
                        predictions = result_predictions[lang][dialect]
                    else:
                        predictions = []
                    if predictions:
                        evaluate_predictions(ground_truth, predictions, data_list, lang, dialect, result_dict, evaluation_stats)
                    else:
                        result_dict[lang][dialect] = {
                            'accuracy_5_bins': 0.0,
                            'f1_5_bins': 0.0,
                            'accuracy_4_bins': 0.0,
                            'f1_4_bins': 0.0,
                            'accuracy_3_bins': 0.0,
                            'f1_3_bins': 0.0,
                        }

            # Create evaluation_scores directory if not exists
            evaluation_scores_dir = 'evaluation_scores'
            if not os.path.exists(evaluation_scores_dir):
                os.makedirs(evaluation_scores_dir)

            # Save result dictionary to a JSON file
            result_file_path = os.path.join(evaluation_scores_dir, f"{model_dir}.json")
            with open(result_file_path, 'w') as result_file:
                json.dump(result_dict, result_file, indent=4)

            # Store evaluation stats for the model
            evaluation_stats_all_models[model_dir] = evaluation_stats

    # Create latex_tables directory if not exists
    latex_dir = 'latex_tables'
    if not os.path.exists(latex_dir):
        os.makedirs(latex_dir)

    



    # create_latex_tables(evaluation_stats_all_models,latex_dir)
    # make_figure(language_avg_stats,latex_dir)



