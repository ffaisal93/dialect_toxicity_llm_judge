import os
import json
import re
import argparse
import logging
from collections import Counter
from vllm import LLM, SamplingParams
import torch

# ---------------------------
# Set Up Logging
# ---------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ---------------------------
# Helper Functions
# ---------------------------


for i in range(torch.cuda.device_count()):
    free, total = torch.cuda.mem_get_info(i)
    logger.info(f"GPU {i}: Free {free / 1e9:.2f} GB / Total {total / 1e9:.2f} GB")

# Set the cache directory before importing torch or vllm
os.environ["VLLM_CACHE_ROOT"] = "/scratch/ffaisal/vllm_cache"

def read_data(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def write_predictions(output_path, prediction_dict):
    with open(output_path, 'w') as f:
        json.dump(prediction_dict, f, indent=2)

def validate_predictions(predictions):
    cleaned = []
    for pred in predictions:
        match = re.search(r'\bS[1-5]\b', pred.upper())
        if match:
            label = match.group(0)
            cleaned.append(label)
        else:
            cleaned.append("INVALID")
    return cleaned

def majority_vote(pred_lists):
    final_preds = []
    num_items = len(pred_lists[0])
    for i in range(num_items):
        candidates = [pred_list[i] for pred_list in pred_lists]
        vote = Counter(candidates).most_common(1)[0][0]
        final_preds.append(vote)
    return final_preds

# ---------------------------
# Main Function
# ---------------------------

def main(args):
    instruction = """
You will be provided with a series of statements in various dialects. Your task is to evaluate the toxicity of each statement.

"Toxicity" is rated on a scale from S1 to S5 based on the level of harmfulness:

• S1: Not harmful (neutral or factual statement)  
• S2: Slightly offensive (minor slurs or casual insults)  
• S3: Moderately harmful (disrespectful or demeaning language)  
• S4: Highly toxic (explicit hate speech or strong language)  
• S5: Extremely toxic (incites violence, threats, or severe hate speech)  

Your response should be **only** the label S1, S2, S3, S4, or S5 — no explanation, no extra text.
"""

    logger.info(f"Loading model: {args.model_id}")
    llm = LLM(model=args.model_id)

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=10,
        top_p=1.0,
        top_k=-1
    )

    model_dirname = os.path.basename(args.model_id)
    output_model_path = os.path.join(args.output_dir, model_dirname)
    os.makedirs(output_model_path, exist_ok=True)

    json_files = [f for f in os.listdir(args.data_folder) if f.endswith('.json')]
    logger.info(f"Found {len(json_files)} files to process in {args.data_folder}")

    for file_name in json_files:
        input_path = os.path.join(args.data_folder, file_name)
        output_path = os.path.join(output_model_path, file_name)

        if os.path.exists(output_path) and not args.overwrite:
            logger.info(f"Skipping {file_name} (already exists)")
            continue

        logger.info(f"Processing {file_name}")
        try:
            data = read_data(input_path)
            dialects = list(data.keys())
            all_predictions = {}

            for dialect in dialects:
                sentences = data[dialect]
                prompts = [f"{instruction}\nStatement: {s}\nToxicity Score:" for s in sentences]

                logger.info(f"  Dialect: {dialect} | {len(sentences)} sentences × {args.num_votes} votes")
                all_validated = []

                for run in range(args.num_votes):
                    logger.debug(f"    Sampling pass {run + 1}/{args.num_votes}")
                    outputs = llm.generate(prompts, sampling_params)
                    raw_outputs = [out.outputs[0].text.strip() for out in outputs]
                    validated = validate_predictions(raw_outputs)
                    all_validated.append(validated)

                voted = majority_vote(all_validated)
                all_predictions[dialect] = voted

            write_predictions(output_path, all_predictions)
            logger.info(f"  ✅ Saved to {output_path}")

        except Exception as e:
            logger.error(f"  ❌ Error processing {file_name}: {e}")

# ---------------------------
# CLI
# ---------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_folder', type=str, required=True, help='Folder with .json input files')
    parser.add_argument('--model_id', type=str, required=True, help='Model ID or path to load with vLLM')
    parser.add_argument('--output_dir', type=str, default='./vllm_results', help='Folder to save results')
    parser.add_argument('--num_votes', type=int, default=5, help='Number of LLM passes per sentence')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing outputs')

    args = parser.parse_args()
    main(args)
