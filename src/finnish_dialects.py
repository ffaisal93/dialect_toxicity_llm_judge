import json
import pickle
import os
from tqdm import tqdm
from murre import supported_dialects, dialectalize_sentences

# Load JSON file into a dictionary
def load_json_to_dict(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'r') as json_file:
            data_dict = json.load(json_file)
    else:
        data_dict = {}
    return data_dict

# Save dictionary to JSON file
def save_dict_to_json(data_dict, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w') as json_file:
        json.dump(data_dict, json_file, indent=4)

# Load Finnish sentences from pickle file
pkl_file_path = 'data/nllb_toxigen_test/fin_Latn.pkl'
with open(pkl_file_path, 'rb') as pkl_file:
    finnish_sentences = pickle.load(pkl_file)

# Load or initialize the dictionary to store dialectalized sentences
output_path = 'data/synthesis/finnish.json'
dialect_dict = load_json_to_dict(output_path)

# Add standard sentences if not already present
if 'standard' not in dialect_dict:
    dialect_dict['standard'] = finnish_sentences

# Get the list of supported dialects
dialects = supported_dialects()

# Process each dialect
batch_size = 64
for dialect in tqdm(dialects, desc="Processing dialects"):
    # Skip if dialect is already processed
    if dialect in dialect_dict:
        continue

    print(f"Processing dialect: {dialect}")
    all_dialectalized_sentences = []

    # Process sentences in batches
    for i in tqdm(range(0, len(finnish_sentences), batch_size), desc=f"Processing batches for {dialect}"):
        batch_sentences = finnish_sentences[i:i + batch_size]
        try:
            dialectalized_sentences = dialectalize_sentences(batch_sentences, dialect)
            all_dialectalized_sentences.extend(dialectalized_sentences)
        except Exception as e:
            print(f"Error processing batch {i} for dialect {dialect}: {e}")
            all_dialectalized_sentences.extend(["" for _ in batch_sentences])

    # Save dialectalized sentences to the dictionary
    dialect_dict[dialect] = all_dialectalized_sentences

    # Save the updated dictionary to JSON file after each dialect
    save_dict_to_json(dialect_dict, output_path)
    print(f"Saved dialectalized sentences for {dialect} to {output_path}")