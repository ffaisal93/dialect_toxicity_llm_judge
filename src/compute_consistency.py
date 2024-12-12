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

def compute_llm_human_consistency(ground_truth: List[float], 
                                  predictions: Dict[str, Dict[str, List[float]]],
                                  max_error: float = 4.0) -> float:
    """
    Compute LLM-Human Consistency (Clh).
    Considers all varieties (standard + non-standard).
    """
    # Collect predictions for all varieties
    all_preds = []
    for lang, varieties in predictions.items():
        for variety, scores in varieties.items():
            all_preds.extend(scores)

    # Calculate the repetition factor for ground_truth
    num_preds = len(all_preds)
    num_ground_truth = len(ground_truth)
    repetition_factor = num_preds // num_ground_truth

    # Repeat ground_truth for each variety
    repeated_ground_truth = ground_truth * repetition_factor
    

    # Ensure the repeated ground_truth matches the aggregated predictions length
    if len(repeated_ground_truth) != len(all_preds):
        raise ValueError("Mismatch: ground_truth length does not match the aggregated predictions length.")

    # Compute RMSE across all examples and varieties
    rmse = np.sqrt(np.mean([(gt - pred) ** 2 for gt, pred in zip(repeated_ground_truth, all_preds)]))

    # Normalize RMSE to similarity
    normalized_rmse_similarity = 1 - (rmse / max_error)

    return normalized_rmse_similarity





def compute_multilingual_consistency(predictions: Dict[str, Dict[str, Dict[str, List[float]]]], 
                                     standard_mapping: Dict[str, str], 
                                     max_possible_std: float = 1.0) -> float:
    """
    Compute Multilingual Consistency (Cml).
    Considers only standard varieties, fetched using standard_mapping.
    """
    # Check that all languages in standard_mapping have a standard variety in predictions
    for lang, standard_variety in standard_mapping.items():
        if lang not in predictions or standard_variety not in predictions[lang]:
            raise KeyError(f"Standard variety '{standard_variety}' not found for language '{lang}'.")

    # Determine the number of examples from the first standard variety
    first_language = next(iter(standard_mapping.keys()))
    num_examples = len(predictions[first_language][standard_mapping[first_language]])

    # Compute example-level standard deviation across all standard varieties
    std_list = []
    for i in range(num_examples):
        scores = [
            predictions[lang][standard_mapping[lang]][i]
            for lang in standard_mapping.keys()  # Use only languages from standard_mapping
        ]
        mean_score = np.mean(scores)
        std_list.append(np.sqrt(np.mean([(score - mean_score) ** 2 for score in scores])))

    # Aggregate STD
    aggregate_std = np.mean(std_list)

    # Normalize
    multilingual_consistency = 1 - (aggregate_std / max_possible_std)

    return multilingual_consistency




def compute_dialectal_consistency(predictions: Dict[str, Dict[str, List[float]]], 
                                  standard_mapping: Dict[str, str],
                                  max_error: float = 4.0) -> Dict[str, float]:
    """
    Compute Dialectal Consistency (Cdl) for each language using normalized RMSE similarity.
    Measures deviation from the standard dialect within the cluster.
    """
    results = {}

    for lang, dialects in predictions.items():
        # Ensure the standard dialect exists in the mapping and predictions
        standard_dialect = standard_mapping.get(lang)
        if not standard_dialect or standard_dialect not in dialects:
            raise KeyError(f"Standard dialect '{standard_dialect}' not found for language '{lang}'.")

        # Fetch scores for the standard dialect
        standard_scores = dialects[standard_dialect]
        num_examples = len(standard_scores)

        # Compute normalized RMSE similarity deviations for each example
        deviations = []
        for i in range(num_examples):
            dialect_scores = [scores[i] for name, scores in dialects.items() if name != standard_dialect]
            # Compute RMSE between standard and dialect scores
            rmse = np.sqrt(np.mean([(dialect - standard_scores[i]) ** 2 for dialect in dialect_scores]))
            # Convert RMSE to normalized similarity
            normalized_similarity = 1 - (rmse / max_error)
            deviations.append(normalized_similarity)

        # Aggregate normalized RMSE similarities
        aggregate_similarity = np.mean(deviations)

        # Store result for the language
        results[lang] = aggregate_similarity

    return results




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
standard_mapping = {
    'arabic': 'acq_arab',
    'english': 'standard'
}

# Compute Metrics
llm_human_consistency = compute_llm_human_consistency(ground_truth, predictions)
print(f"LLM-Human Consistency (Clh): {llm_human_consistency:.4f}")

multilingual_consistency = compute_multilingual_consistency(predictions, standard_mapping)
print(f"Multilingual Consistency (Cml): {multilingual_consistency:.4f}")

dialectal_consistency = compute_dialectal_consistency(predictions, standard_mapping)
for lang, cdl in dialectal_consistency.items():
    print(f"Dialectal Consistency for {lang} (Cdl): {cdl:.4f}")

