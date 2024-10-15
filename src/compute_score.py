import os
import json
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
import pandas as pd

def evaluate_predictions(ground_truth, predictions, data_values, lang, dialect, result_dict):
    # Discard indexes where data value is ''
    valid_indexes = [i for i, val in enumerate(data_values) if val != '']
    ground_truth = [ground_truth[i] for i in valid_indexes]
    predictions = [predictions[i] for i in valid_indexes]

    # Convert ground truth to bins of 5 and compute accuracy and F1
    ground_truth_bins_5 = [int(round((val - 1) * 4 / (5 - 1))) + 1 for val in ground_truth]
    predictions_bins_5 = []

    for pred in predictions:
        if isinstance(pred, dict) and 'Toxicity' in pred:
            toxicity = pred['Toxicity']
            if toxicity == 'S1':
                predictions_bins_5.append(1)
            elif toxicity == 'S2':
                predictions_bins_5.append(2)
            elif toxicity == 'S3':
                predictions_bins_5.append(3)
            elif toxicity == 'S4':
                predictions_bins_5.append(4)
            elif toxicity == 'S5':
                predictions_bins_5.append(5)
        else:
            predictions_bins_5.append(None)

    # Filter out None values for valid comparisons
    if len(predictions_bins_5) > 0:
        filtered_ground_truth_5, filtered_predictions_5 = zip(*[(g, p) for g, p in zip(ground_truth_bins_5, predictions_bins_5) if p is not None])
        accuracy_5 = accuracy_score(filtered_ground_truth_5, filtered_predictions_5)
        f1_5 = f1_score(filtered_ground_truth_5, filtered_predictions_5, average='weighted')
    else:
        accuracy_5 = 0.0
        f1_5 = 0.0
    result_dict[lang][dialect]['accuracy_5_bins'] = accuracy_5
    result_dict[lang][dialect]['f1_5_bins'] = f1_5

    # Convert ground truth and predictions to bins of 4 and compute accuracy and F1
    ground_truth_bins_4 = [int(round((val - 1) * 3 / (5 - 1))) + 1 for val in ground_truth]
    predictions_bins_4 = []

    for pred in predictions:
        if isinstance(pred, dict) and 'Toxicity' in pred:
            toxicity = pred['Toxicity']
            if toxicity in ['S1']:
                predictions_bins_4.append(1)
            elif toxicity in ['S2']:
                predictions_bins_4.append(2)
            elif toxicity in ['S3']:
                predictions_bins_4.append(3)
            elif toxicity in ['S4', 'S5']:
                predictions_bins_4.append(4)
        else:
            predictions_bins_4.append(None)

    # Filter out None values for valid comparisons
    if len(predictions_bins_4) > 0:
        filtered_ground_truth_4, filtered_predictions_4 = zip(*[(g, p) for g, p in zip(ground_truth_bins_4, predictions_bins_4) if p is not None])
        accuracy_4 = accuracy_score(filtered_ground_truth_4, filtered_predictions_4)
        f1_4 = f1_score(filtered_ground_truth_4, filtered_predictions_4, average='weighted')
    else:
        accuracy_4 = 0.0
        f1_4 = 0.0
    result_dict[lang][dialect]['accuracy_4_bins'] = accuracy_4
    result_dict[lang][dialect]['f1_4_bins'] = f1_4

    # Convert ground truth and predictions to bins of 3 and compute accuracy and F1
    ground_truth_bins_3 = [int(round((val - 1) * 2 / (5 - 1))) + 1 for val in ground_truth]
    predictions_bins_3 = []

    for pred in predictions:
        if isinstance(pred, dict) and 'Toxicity' in pred:
            toxicity = pred['Toxicity']
            if toxicity in ['S1', 'S2']:
                predictions_bins_3.append(1)
            elif toxicity == 'S3':
                predictions_bins_3.append(2)
            elif toxicity in ['S4', 'S5']:
                predictions_bins_3.append(3)
        else:
            predictions_bins_3.append(None)

    # Filter out None values for valid comparisons
    if len(predictions_bins_3) > 0:
        filtered_ground_truth_3, filtered_predictions_3 = zip(*[(g, p) for g, p in zip(ground_truth_bins_3, predictions_bins_3) if p is not None])
        accuracy_3 = accuracy_score(filtered_ground_truth_3, filtered_predictions_3)
        f1_3 = f1_score(filtered_ground_truth_3, filtered_predictions_3, average='weighted')
    else:
        accuracy_3 = 0.0
        f1_3 = 0.0
    result_dict[lang][dialect]['accuracy_3_bins'] = accuracy_3
    result_dict[lang][dialect]['f1_3_bins'] = f1_3

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
                        print(f"{filename}: {len(data)} fields")

            # Initialize result dictionary
            result_dict = {lang: {dialect: {} for dialect in dialect_data} for lang, dialect_data in data_values.items()}

            # Compute accuracy and F1 for all languages and dialects
            for lang, dialect_data in data_values.items():
                for dialect, data_list in dialect_data.items():
                    if lang in result_predictions and dialect in result_predictions[lang]:
                        predictions = result_predictions[lang][dialect]
                    else:
                        predictions = []
                    if predictions:
                        evaluate_predictions(ground_truth, predictions, data_list, lang, dialect, result_dict)
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