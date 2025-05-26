import os
import json

def analyze_toxicity_shift_cluster(ground_truth, predictions, cluster_representatives):
    """
    Analyze toxicity shift from English predictions to other language clusters and their dialectal varieties.
    Computes:
    - For toxic predictions (4,5): % of cases where toxicity reduces
    - For non-toxic predictions (1,2): % of cases where toxicity increases

    Args:
    - ground_truth (List[int]): English prediction labels.
    - predictions (Dict[str, Dict[str, List[int]]]): Predictions for each language and variety.
    - cluster_representatives (Dict[str, str]): Mapping of cluster representatives for each language.

    Returns:
    - Dict: Cluster-specific summary of toxicity shifts.
    """
    toxicity_results = {}

    for lang, varieties in predictions.items():
        cluster_rep = cluster_representatives.get(lang)
        if not cluster_rep or cluster_rep not in varieties:
            raise KeyError(f"Cluster representative '{cluster_rep}' not found for language '{lang}'.")

        toxicity_results[lang] = {"toxic": {}, "non_toxic": {}}
        cluster_rep_predictions = varieties[cluster_rep]

        # Filter indices based on English (ground truth) categories
        toxic_indices = [i for i, label in enumerate(ground_truth) if label in [4, 5]]
        non_toxic_indices = [i for i, label in enumerate(ground_truth) if label in [1, 2]]

        # Analyze Toxicity Reductions (for toxic inputs)
        toxic_reduced_cluster = [1 for i in toxic_indices if cluster_rep_predictions[i] < 4]
        toxic_reduced_dialects = [
            1
            for variety, preds in varieties.items()
            if variety != cluster_rep
            for i in toxic_indices
            if preds[i] < 4
        ]

        # Analyze Toxicity Increases (for non-toxic inputs)
        non_toxic_increased_cluster = [1 for i in non_toxic_indices if cluster_rep_predictions[i] > 2]
        non_toxic_increased_dialects = [
            1
            for variety, preds in varieties.items()
            if variety != cluster_rep
            for i in non_toxic_indices
            if preds[i] > 2
        ]

        toxicity_results[lang]["toxic"]["cluster_representative"] = {
            "percentage_reduced": 100 * len(toxic_reduced_cluster) / len(toxic_indices) if toxic_indices else 0.0,
            "count":f"{len(toxic_reduced_cluster)} / {len(toxic_indices)}"
        }
        toxicity_results[lang]["toxic"]["dialects"] = {
            "percentage_reduced": 100 * len(toxic_reduced_dialects) / (len(toxic_indices) * (len(varieties) - 1)) if toxic_indices and len(varieties) > 1 else 0.0,
            "count":f"{len(toxic_reduced_dialects)} / {(len(toxic_indices) * (len(varieties) - 1))}"
        }

        toxicity_results[lang]["non_toxic"]["cluster_representative"] = {
            "percentage_increased": 100 * len(non_toxic_increased_cluster) / len(non_toxic_indices) if non_toxic_indices else 0.0,
            "count":f"{len(non_toxic_increased_cluster)} / {len(non_toxic_indices)}"
        }
        toxicity_results[lang]["non_toxic"]["dialects"] = {
            "percentage_increased": 100 * len(non_toxic_increased_dialects) / (len(non_toxic_indices) * (len(varieties) - 1)) if non_toxic_indices and len(varieties) > 1 else 0.0,
            "count":f"{len(non_toxic_increased_dialects)} / {(len(non_toxic_indices) * (len(varieties) - 1))}"
        }

    return toxicity_results

import pandas as pd

def build_combined_toxicity_table(all_toxicity_shifts, category, metric_key):
    """
    Build a single table combining cluster and dialect values as (cluster, dialect).

    Args:
        all_toxicity_shifts (dict): The model-lang-toxicity shift data.
        category (str): "toxic" or "non_toxic"
        metric_key (str): "percentage_reduced" or "percentage_increased"

    Returns:
        pd.DataFrame
    """
    models = list(all_toxicity_shifts.keys())
    languages = sorted({lang for model in models for lang in all_toxicity_shifts[model]})

    combined_df = pd.DataFrame(index=languages, columns=models)

    for model in models:
        for lang in languages:
            info = all_toxicity_shifts[model].get(lang, {}).get(category, {})
            cluster_val = info.get("cluster_representative", {}).get(metric_key, 0.0)
            dialect_val = info.get("dialects", {}).get(metric_key, 0.0)
            combined_df.loc[lang, model] = f"({cluster_val:.1f}, {dialect_val:.1f})"

    return combined_df

def add_avg_row_and_column(df):
    # Parse cluster and dialect values
    cluster_vals = df.applymap(lambda x: float(x.strip('()').split(',')[0]) if isinstance(x, str) else 0.0)
    dialect_vals = df.applymap(lambda x: float(x.strip('()').split(',')[1]) if isinstance(x, str) else 0.0)

    # Average row (model-wise)
    avg_row = {
        model: f"({cluster_vals[model].mean():.1f}, {dialect_vals[model].mean():.1f})"
        for model in df.columns
    }
    df.loc["Avg"] = avg_row

    # Recalculate cluster/dialect values excluding 'Avg' row for column-wise avg
    cluster_vals = cluster_vals.drop(index="Avg", errors="ignore")
    dialect_vals = dialect_vals.drop(index="Avg", errors="ignore")

    # Average column (language-wise)
    df["Avg"] = [
        f"({cluster_vals.loc[lang].mean():.1f}, {dialect_vals.loc[lang].mean():.1f})"
        for lang in cluster_vals.index
    ] + [f"({cluster_vals.mean().mean():.1f}, {dialect_vals.mean().mean():.1f})"]  # grand average

    return df





# Example usage
if __name__ == "__main__":
    # Directory containing the processed data
    processed_data_dir = 'data/processed_data'

    # Directory containing the results data
    results_final_dir = 'vllm_results_gpt_assisted'
    results_eng_fin = 'vllm_results'
    
    # Iterate over each model in the results_final directory
    evaluation_stats_all_models = {}
    result_predictions = {}
    for model_dir in os.listdir(results_final_dir):
        model_path = os.path.join(results_final_dir, model_dir)
        if os.path.isdir(model_path):
            result_predictions[model_dir]={}
            for filename in os.listdir(model_path):
                if filename.endswith('.json'):
                    language_cluster = filename.replace('.json', '')
                    file_path = os.path.join(model_path, filename)
                    # Load the JSON data
                    with open(file_path, 'r') as json_file:
                        data = json.load(json_file)
                        result_predictions[model_dir][language_cluster] = data
                        print(f"{model_dir} {filename}: {len(data)} fields")
                    

    cluster_representatives = {
        "arabic": "arb_Arab",
        "english": "standard",
        "common_turkic":"tur_Latn",
        "bengali":"standard_Bn",
        "sotho-tswana":"nso_Latn",
        "english":"standard",
        "high_german":"lvs_Latn",
        "norwegian":"nob_Latn",
        "kurdish":"ckb_Arab",
        "chinese":"yue_Hant",
        "finnish":"fin_Latn"

    }
    
    num_bins = 5
    cut_off=380
    all_toxicity_shifts={}
    for model in result_predictions.keys():
        print(model)
        for lang in result_predictions[model].keys():
            for dialect in result_predictions[model][lang].keys():
                predictions = result_predictions[model][lang][dialect]
                predictions = predictions[:cut_off]
                predictions_bins=[]
                # Map predictions to bins
                for pred in predictions:
                    # if isinstance(pred, dict) and 'Toxicity' in pred:
                    toxicity = pred
                    if toxicity in ['S1']:
                        predictions_bins.append(1)
                    elif toxicity in ['S2']:
                        predictions_bins.append(2 if num_bins > 2 else 1)
                    elif toxicity == 'S3':
                        predictions_bins.append(3 if num_bins > 3 else (2 if num_bins > 2 else 1))
                    elif toxicity in ['S4']:
                        predictions_bins.append(4)
                    elif toxicity in ['S5']:
                        predictions_bins.append(5)
                    else:
                        predictions_bins.append(-1)
                result_predictions[model][lang][dialect]=predictions_bins

        ground_truth=result_predictions[model]['english']['standard'][:cut_off]
        
        all_toxicity_shifts[model] = analyze_toxicity_shift_cluster(ground_truth, 
                                                                    result_predictions[model], 
                                                                    cluster_representatives)
        print(all_toxicity_shifts[model])
    
    # For toxic sentences → % reduction
    toxic_df = build_combined_toxicity_table(all_toxicity_shifts, "toxic", "percentage_reduced")
    toxic_df = add_avg_row_and_column(toxic_df)

    # For non-toxic sentences → % increase
    non_toxic_df = build_combined_toxicity_table(all_toxicity_shifts, "non_toxic", "percentage_increased")
    non_toxic_df = add_avg_row_and_column(non_toxic_df)

    # Mapping of long model names to shorter ones
    short_names = {
        "gpt-4.1-2025-04-14": "GPT",
        "Mistral-Nemo-Instruct-2407": "Nemo",
        "Llama-3.1-8B": "LLaMA",
        "Qwen2.5-7B-Instruct": "Qwen",
        "gemma-3-12b-it": "Gemma",
    }

    # Rename columns in both toxic and non-toxic DataFrames
    toxic_df.rename(columns=short_names, inplace=True)
    non_toxic_df.rename(columns=short_names, inplace=True)

    toxic_df = toxic_df[[  'GPT', 'Nemo', 'LLaMA', 'Qwen','Gemma', 'Avg']]
    non_toxic_df = non_toxic_df[[  'GPT', 'Nemo', 'LLaMA', 'Qwen','Gemma', 'Avg']]


    # Convert to LaTeX
    print(toxic_df.to_latex(escape=False, caption="Toxic sentences: (Cluster Rep., Dialect) % reduced", label="tab:toxicity_combined_toxic"))
    print(non_toxic_df.to_latex(escape=False, caption="Non-toxic sentences: (Cluster Rep., Dialect) % increased", label="tab:toxicity_combined_nontoxic"))

        


    # # Example Usage
    # ground_truth = [1, 2, 3, 4, 5]
    # predictions = {
    #     "arabic": {
    #         "standard": [1, 5, 1, 1, 5],
    #         "dialect1": [2, 4, 4, 3, 4],
    #         "dialect2": [1, 2, 3, 2, 1],
    #     },
    #     "english": {
    #         "standard": [5, 1, 5, 5, 1],
    #         "dialect1": [2, 3, 4, 2, 1],
    #     }
    # }



    # toxicity_shift_results = analyze_toxicity_shift_cluster(ground_truth, predictions, cluster_representatives)
    # print(toxicity_shift_results)
