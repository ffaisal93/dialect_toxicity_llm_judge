# import os
# import json
# import pandas as pd
# import numpy as np

# # Load language mapping from metadata/lang_mapping.json
# with open('metadata/lang_mapping.json', 'r') as f:
#     lang_mapping = json.load(f)

# # Directory containing the evaluation scores
# evaluation_scores_dir = 'evaluation_scores'

# # Define standard dialect mapping based on underlined varieties in LaTeX table
# standard_dialects = {
#     'arabic': 'Standard Arabic',
#     'bengali': 'Standard',
#     'chinese': 'Cantonese',
#     'finnish': 'Finnish',
#     'kurdish': 'Central Kurdish',
#     'norwegian': 'Norwegian Bokmal',
#     'high_german': 'Latvian',
#     'english': 'Standard',
#     'sotho-tswana': 'Northern Sotho',
#     'common_turkic': 'Central Oghuz'
# }

# # Reverse lang_mapping to get dialect keys from values
# reverse_lang_mapping = {v.lower(): k for k, v in lang_mapping.items()}

# # Compute consistency scores
# consistency_records = []
# all_dialectal_consistency_records = []
# for filename in os.listdir(evaluation_scores_dir):
#     if filename.endswith('.json'):
#         model_name = filename.replace('.json', '')
#         file_path = os.path.join(evaluation_scores_dir, filename)

#         # Load the JSON data
#         with open(file_path, 'r') as json_file:
#             result_data = json.load(json_file)

#         # Prepare data for consistency scores
#         llm_human_consistency = []
#         multilingual_variance = []
#         dialectal_consistency = []
#         dialectal_consistency_records = []

#         for language_cluster, dialects in result_data.items():
#             print(language_cluster, dialects.keys())
#             # Get standard dialect key from lang_mapping or use as is if already in key format
#             standard_dialect_value = None
#             for key in dialects.keys():
#                 if 'standard' == key.lower():
#                     standard_dialect_value = 'standard'
#                     standard_dialect = standard_dialect_value
#             if standard_dialect_value is None:
#                 standard_dialect = reverse_lang_mapping.get(standard_dialects[language_cluster].lower(), standard_dialect_value)
#             standard_accuracy = None
#             dialect_accuracies = []

#             for dialect, metrics in dialects.items():
#                 print(dialect,standard_dialect)
#                 accuracy_3_bin = metrics['accuracy_3_bins']
#                 llm_human_consistency.append(accuracy_3_bin)
#                 if dialect == standard_dialect:
#                     standard_accuracy = accuracy_3_bin
#                 else:
#                     dialect_accuracies.append(accuracy_3_bin)

#             if standard_accuracy is not None:
#                 multilingual_variance.append(standard_accuracy)
#                 for dialect_accuracy in dialect_accuracies:
#                     dialectal_consistency.append(abs(standard_accuracy - dialect_accuracy))

#             # Calculate Dialectal Consistency for each language group
#             if standard_accuracy is not None and len(dialect_accuracies) > 0:
#                 avg_dialectal_diff = np.mean([abs(standard_accuracy - acc) for acc in dialect_accuracies])
#                 C_dl = 1 - avg_dialectal_diff
#                 dialectal_consistency_records.append([language_cluster.capitalize(), model_name, round(C_dl * 100, 2)])

#         all_dialectal_consistency_records.extend(dialectal_consistency_records)
#         # print(all_dialectal_consistency_records)

#         # Calculate LLM-Human Consistency
#         C_lh = np.mean(llm_human_consistency) * 100
#         # print(multilingual_variance)

#         # Calculate Multilingual Consistency
#         if len(multilingual_variance) > 1:
#             variance = np.var(multilingual_variance)
#             C_ml = (1 - variance) * 100
#         else:
#             C_ml = 0

#         # Calculate Dialectal Consistency
#         if len(dialectal_consistency) > 0:
#             avg_dialectal_diff = np.mean(dialectal_consistency)
#             C_dl = (1 - avg_dialectal_diff) * 100
#         else:
#             C_dl = 0

#         # Append consistency scores
#         consistency_records.append([
#             model_name, round(C_lh, 2), round(C_ml, 2), round(C_dl, 2)
#         ])

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



import numpy as np
from typing import List, Dict
import pandas as pd
import re
import os
import json

import re
from sklearn.metrics import accuracy_score, f1_score
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error
from sklearn.metrics import mean_absolute_error, cohen_kappa_score

def clean_discrepancies(results):
    """
    Cleans discrepancies in the result data, ensuring all labels are in the correct format,
    converts them to integers (e.g., "S4" -> 4), and handles unexpected labels by assigning -1.

    Args:
    - results (dict): The dictionary containing the results with potential discrepancies.

    Returns:
    - dict: A cleaned dictionary with labels standardized and converted to integers.
    """
    def clean_label(label):
        nonlocal unexpected_count
        # Remove any square brackets or whitespace around the label
        label = re.sub(r"[\[\]\s]", "", label)
        # Ensure it starts with "S" if it's numeric (e.g., "1" -> "S1")
        if label.isdigit():
            label = f"S{label}"
        # Extract the numeric part from labels like "S4"
        if label.startswith("S") and label[1:].isdigit():
            return int(label[1:])
        # Handle unexpected labels
        unexpected_count += 1
        return -1

    # Iterate through the dictionary and clean each label
    cleaned_results = {}
    for key, values in results.items():
        unexpected_count = 0  # Counter for unexpected labels
        cleaned_values = [clean_label(value) for value in values]
        cleaned_results[key] = cleaned_values

        # Print the total count of unexpected labels
        if unexpected_count > 0:
            print(f"Total unexpected labels encountered: {unexpected_count},{key}")

    return cleaned_results




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


def compute_llm_human_consistency(human_labels: List[float], 
                                  predictions: Dict[str, Dict[str, List[float]]],
                                  max_error: float = 4.0) -> float:
    """
    Compute LLM-Human Consistency (Clh).
    Considers all varieties (standard + non-standard).
    
    Args:
        human_labels (List[float]): The human-provided ground truth labels for each example.
        predictions (Dict[str, Dict[str, List[float]]]): Dictionary of predictions for each language and variety.
        max_error (float): Maximum possible deviation based on the label range.
    
    Returns:
        float: LLM-Human Consistency score (Clh).
    """
    # Collect all predictions across all varieties
    all_deviations = []
    for lang, varieties in predictions.items():
        for variety, scores in varieties.items():
            # Compute deviations between predictions and human labels
            deviations = [(pred - gt) for pred, gt in zip(scores, human_labels)]
            all_deviations.extend(deviations)
    
    # Calculate RMSE over all deviations
    rmse = np.sqrt(np.mean([dev ** 2 for dev in all_deviations]))
    
    # Normalize RMSE to similarity
    normalized_rmse_similarity = 1 - (rmse / max_error)

    return normalized_rmse_similarity





def compute_multilingual_consistency(predictions: Dict[str, Dict[str, List[float]]],
                                     standard_mapping: Dict[str, str],
                                     max_error: float = 4.0) -> float:
    """
    Compute Multilingual Consistency (Cml).
    Considers only cluster-representative languages using standard_mapping.
    
    Args:
        predictions (Dict[str, Dict[str, List[float]]]): Predictions for each language and variety.
        standard_mapping (Dict[str, str]): Mapping of languages to their cluster-representative varieties.
        max_error (float): Maximum possible deviation based on the label range.
    
    Returns:
        float: Multilingual Consistency score (Cml).
    """
    # Collect predictions for cluster-representative varieties
    cluster_predictions = []
    for lang, varieties in predictions.items():
        cluster_rep = standard_mapping.get(lang)
        if not cluster_rep or cluster_rep not in varieties:
            raise KeyError(f"Cluster representative '{cluster_rep}' not found for language '{lang}'.")
        cluster_predictions.append(varieties[cluster_rep])

    # Transpose predictions to compute per-example variability across languages
    cluster_predictions_transposed = list(zip(*cluster_predictions))
    
    # Compute deviations for each example
    deviations = []
    for example_scores in cluster_predictions_transposed:
        mean_score = sum(example_scores) / len(example_scores)  # Compute mean for this example
        deviation = np.sqrt(np.mean([(score - mean_score) ** 2 for score in example_scores]))
        deviations.append(deviation)
    # Aggregate deviations and compute consistency
    aggregate_dev = np.mean(deviations)
    multilingual_consistency = 1 - (aggregate_dev / max_error)

    return multilingual_consistency





def compute_dialectal_consistency(predictions: Dict[str, Dict[str, List[float]]],
                                  cluster_representatives: Dict[str, str],
                                  max_error: float = 4.0) -> float:
    """
    Compute Dialectal Consistency (Cdl).
    Considers deviations of dialectal varieties from their cluster-representative variety.
    
    Args:
        predictions (Dict[str, Dict[str, List[float]]]): Dictionary of predictions for each language and variety.
        cluster_representatives (Dict[str, str]): Mapping of languages to their cluster-representative varieties.
        max_error (float): Maximum possible deviation based on the label range.
    
    Returns:
        float: Dialectal Consistency score (Cdl).
    """
    cluster_level_consistencies = []
    langs=[]
    for lang, varieties in predictions.items():
        # Get the cluster representative for this language
        cluster_rep = cluster_representatives.get(lang)
        if not cluster_rep or cluster_rep not in varieties:
            raise KeyError(f"Cluster representative '{cluster_rep}' not found for language '{lang}'.")

        standard_predictions = varieties[cluster_rep]
        dialect_predictions = [scores for variety, scores in varieties.items() if variety != cluster_rep]

        if not dialect_predictions:
            continue  # Skip if no dialects are present

        # Transpose to compute per-example variability across dialects
        dialect_predictions_transposed = list(zip(*dialect_predictions))

        # Compute deviations for each example
        deviations = []
        for example_idx, (standard_score, dialect_scores) in enumerate(zip(standard_predictions, dialect_predictions_transposed)):
            deviation = np.sqrt(np.mean([(dialect_score - standard_score) ** 2 for dialect_score in dialect_scores]))
            deviations.append(deviation)

        # Aggregate deviations and compute cluster-level consistency
        cluster_aggregate_dev = np.mean(deviations)
        cluster_consistency = 1 - (cluster_aggregate_dev / max_error)
        cluster_level_consistencies.append(cluster_consistency)
        langs.append(lang)

    cluster_level_consistencies_dict=dict(zip(langs,cluster_level_consistencies))
    # Aggregate consistency across all clusters
    dialectal_consistency = np.mean(cluster_level_consistencies)

    return dialectal_consistency,cluster_level_consistencies_dict




def preprocess_predictions_and_ground_truth(ground_truth, predictions, indices_to_remove):
    """
    Preprocess ground truth and predictions by:
    1. Removing entries in predictions where the value is -1.
    2. Removing the corresponding ground truth entries.
    3. Removing entries at specific indices provided in indices_to_remove.

    Args:
    - ground_truth (list): List of ground truth labels.
    - predictions (dict): Dictionary of predictions for different languages/varieties.
    - indices_to_remove (list): List of indices to remove.

    Returns:
    - tuple: Filtered ground truth and predictions.
    """
    # Start with all valid indices
    valid_indices = set(range(len(ground_truth)))

    # Remove indices where predictions have -1
    for lang, varieties in predictions.items():
        for variety, scores in varieties.items():
            for idx, score in enumerate(scores):
                if score == -1:
                    valid_indices.discard(idx)

    # Remove additional indices specified in indices_to_remove
    valid_indices -= set(indices_to_remove)

    # Sort valid indices for consistent filtering
    valid_indices = sorted(valid_indices)

    # Filter ground truth based on valid indices
    filtered_ground_truth = [ground_truth[idx] for idx in valid_indices]

    # Filter predictions for each variety based on valid indices
    filtered_predictions = {}
    for lang, varieties in predictions.items():
        filtered_predictions[lang] = {
            variety: [scores[idx] for idx in valid_indices]
            for variety, scores in varieties.items()
        }

    return filtered_ground_truth, filtered_predictions


# Example Input
ground_truth = [1, 2, 3, 4, 5]
predictions = {
    'arabic': {
        'acq_arab': [3, 2, 3, 4, 5],
        'dialect1': [4, 3, 3, 3, 4],
        'dialect2': [2, 1, 3, 4, 5]
    },
    'english': {
        'standard': [5, 2, 3, 5, 5],
        'american': [5,2,2,5,4]
    }
}


# Define standard dialect mapping based on underlined varieties in LaTeX table
standard_mapping = {
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

# Load language mapping from metadata/lang_mapping.json
with open('metadata/lang_mapping.json', 'r') as f:
    lang_mapping = json.load(f)

def data_cutoff(data,cut_off):
    cut_off_data={}
    for k,v in data.items():
        cut_off_data[k]=v[:cut_off]
    return cut_off_data

def variety_mapping(data):
    mapped_data={}
    for k,v in data.items():
        mapped_data[lang_mapping[k]]=v
    return mapped_data


# Example usage
if __name__ == "__main__":
    num_bins=5
    cut_off=380
    indices_to_remove = [6,8,13,17,22]
    df = pd.read_parquet("data/toxigen/test-00000-of-00001.parquet")
    ground_truth_series = list(df['intent'])
    ground_truth = assign_bins(ground_truth_series, num_bins)
    ground_truth=ground_truth[:cut_off]

    # Directory containing the results data
    results_final_dir = 'results_final'    
    model_lists=['gpt-4o-2024-08-06']
    settings=['oneshot_eng']
    # Iterate over each model in the results_final directory
    predictions = {}
    for model_dir in model_lists:
        for setting in settings:
            model_path = os.path.join(results_final_dir, model_dir, setting)
            print(model_path)
            if os.path.isdir(model_path):
                result_predictions = {}
                for filename in os.listdir(model_path):
                    if filename.endswith('.json'):
                        language_cluster = filename.replace('.json', '')
                        file_path = os.path.join(model_path, filename)
                        # Load the JSON data
                        with open(file_path, 'r') as json_file:
                            data = json.load(json_file)
                            print(language_cluster)
                            data_processed = clean_discrepancies(data)
                            data_after_cut = data_cutoff(data_processed, cut_off)
                            mapped_data = variety_mapping(data_after_cut)
                            predictions[language_cluster]=mapped_data
                            print(language_cluster, mapped_data.keys())

    
            
            
            # Preprocess data
            filtered_ground_truth, filtered_predictions = preprocess_predictions_and_ground_truth(
                ground_truth, predictions, indices_to_remove
            )


            result_dict={}
            for lang,varieties in filtered_predictions.items():
                result_dict[lang]={}
                for variety,pred in varieties.items():
                    scores = compute_metrics(filtered_ground_truth, pred)
                    result_dict[lang][variety]=scores
            
            # Create evaluation_scores directory if not exists
            evaluation_scores_dir = 'evaluation_scores'
            if not os.path.exists(evaluation_scores_dir):
                os.makedirs(evaluation_scores_dir)

            # Save result dictionary to a JSON file
            result_file_path = os.path.join(evaluation_scores_dir, f"{model_dir}#{setting}.json")
            print(result_file_path)
            with open(result_file_path, 'w') as result_file:
                json.dump(result_dict, result_file, indent=4)


            
            # Compute Metrics
            llm_human_consistency = compute_llm_human_consistency(filtered_ground_truth, filtered_predictions)
            print(f"LLM-Human Consistency (Clh): {llm_human_consistency:.4f}")

            multilingual_consistency = compute_multilingual_consistency(filtered_predictions, standard_mapping)
            print(f"Multilingual Consistency (Cml): {multilingual_consistency:.4f}")

            dialectal_consistency,cluster_level_consistencies = compute_dialectal_consistency(filtered_predictions, standard_mapping)
            for lang, cdl in cluster_level_consistencies.items():
                print(f"Dialectal Consistency for {lang} (Cdl): {cdl:.4f}")
            print(f"Dialectal Consistency (Cdl): {dialectal_consistency:.4f}")

