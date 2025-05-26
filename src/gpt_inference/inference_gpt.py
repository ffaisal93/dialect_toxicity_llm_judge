import os
import json
import argparse
import logging
from typing import Dict, List
import openai
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# Function to load prompts from a file
def load_prompt(file_path: str) -> List[Dict[str, str]]:
    logging.info(f"Loading prompts from {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)  # The file should be a JSON file with the prompt structure

# Function to load data from a specified JSON file
def load_sentences(file_path: str) -> Dict[str, List[str]]:
    logging.info(f"Loading sentences from {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)  # Load the file as a dictionary

# Function to invoke GPT model for generating responses
def generate_toxicity_scores(api_key: str, prompt_template: List[Dict[str, str]], prompt_type: str, model_name: str, sentences: List[str]) -> List[str]:
    client = openai.OpenAI(api_key=api_key)
    responses = []

    for sentence in tqdm(sentences, desc="Processing Sentences"):
        try:
            # Replace {{input_statement}} placeholder in the prompt template
            conversation = json.loads(json.dumps(prompt_template))[prompt_type]
            conversation[-2]['content'] = conversation[-2]['content'].replace("{{input_statement}}", sentence)

            # Invoke GPT model
            response = client.chat.completions.create(
                model=model_name,
                messages=conversation,
                temperature=0.7
            )
            # Extract the toxicity level
            responses.append(response.choices[0].message.content.strip())
        except Exception as e:
            logging.error(f"Error processing sentence '{sentence}': {e}")
            responses.append("")
    return responses

# Function to save results in the specified structure
def save_results(results: Dict[str, List[str]], output_dir: str, model_name: str, prompt_type: str, input_file_name: str):
    result_path = os.path.join(output_dir, model_name, prompt_type)
    os.makedirs(result_path, exist_ok=True)

    output_file_path = os.path.join(result_path, input_file_name)
    logging.info(f"Saving results to {output_file_path}")
    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4)

# Main function to execute the pipeline
def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Toxicity score pipeline")
    parser.add_argument('--api_key_file', required=True, help="Path to the file containing the OpenAI API key")
    parser.add_argument('--prompt_file', required=True, help="Path to the prompt file")
    parser.add_argument('--input_file', required=True, help="Path to the input JSON file containing sentences")
    parser.add_argument('--output_dir', required=True, help="Directory to save results")
    parser.add_argument('--prompt_type', required=True, help="Type of prompt (e.g., oneshot_eng, 5shot_eng)")
    parser.add_argument('--model_name', required=True, help="Model name for organizing results")
    parser.add_argument('--cut_off', type=str, default=None, help="Maximum number of sentences to process per dialect")
    args = parser.parse_args()
    print(args)

    # Load the API key from the file
    logging.info(f"Loading API key from {args.api_key_file}")
    with open(args.api_key_file, 'r', encoding='utf-8') as f:
        api_key = f.read().strip()

    # Load prompts and data
    prompt_template = load_prompt(args.prompt_file)
    sentences_by_dialect = load_sentences(args.input_file)

    logging.info(sentences_by_dialect.keys())

    # Process each dialect
    results = {}
    for dialect, sentences in sentences_by_dialect.items():
        logging.info(f"Processing dialect: {dialect}")
        if args.cut_off:
            args.cut_off=int(args.cut_off)
            sentences = sentences[:args.cut_off]  # Apply cut-off limit
        results[dialect] = generate_toxicity_scores(api_key, prompt_template, args.prompt_type, args.model_name, sentences)

    # Save results
    input_file_name = os.path.basename(args.input_file)  # Get the input file name
    save_results(results, args.output_dir, args.model_name, args.prompt_type, input_file_name)

if __name__ == "__main__":
    main()