import os
import pickle
import json

# Directory paths
translated_data_dir = 'data/nllb_toxigen_test'
processed_data_dir = 'data/processed_data'
synthesis_data_path = 'data/synthesis/finnish.json'
english_synthesis_path = 'data/synthesis/english.json'

# Ensure processed data directory exists
os.makedirs(processed_data_dir, exist_ok=True)

# Define dialect grouping for languages
language_groups = {
    'sotho-tswana': ['nso_Latn', 'sot_Latn'],
    'arabic': ['acm_Arab', 'acq_Arab', 'aeb_Arab', 'ajp_Arab', 'apc_Arab', 'arb_Arab', 'ars_Arab', 'ary_Arab', 'arz_Arab'],
    'chinese': ['yue_Hant', 'zho_Hans', 'zho_Hant'],
    'high_german': ['ltg_Latn','lvs_Latn'],
    'common_turkic': ['tur_Latn','azb_Arab','azj_Latn'],
    'norwegian': ['nno_Latn', 'nob_Latn'],
    'kurdish': ['ckb_Arab','kmr_Latn']
}

# Process each language group
for group_name, dialects in language_groups.items():
    group_data = {}
    for dialect in dialects:
        file_path = os.path.join(translated_data_dir, f'{dialect}.pkl')
        if os.path.exists(file_path):
            with open(file_path, 'rb') as pkl_file:
                sentences = pickle.load(pkl_file)
                group_data[dialect] = sentences
        else:
            print(f"Warning: File for {dialect} not found at {file_path}")

    # Save the group data to a JSON file
    output_file_path = os.path.join(processed_data_dir, f'{group_name}.json')
    with open(output_file_path, 'w') as json_file:
        json.dump(group_data, json_file, indent=4)

    print(f"Processed data for {group_name} saved to {output_file_path}")

# Special processing for Finnish
finnish_data = {}

# Load data from fin_Latn.pkl
fin_latn_path = os.path.join(translated_data_dir, 'fin_Latn.pkl')
if os.path.exists(fin_latn_path):
    with open(fin_latn_path, 'rb') as pkl_file:
        finnish_data['fin_Latn'] = pickle.load(pkl_file)
else:
    print(f"Warning: File for fin_Latn not found at {fin_latn_path}")

# Load existing synthesis data from finnish.json
if os.path.exists(synthesis_data_path):
    with open(synthesis_data_path, 'r') as json_file:
        synthesis_data = json.load(json_file)
        synthesis_data.pop('standard', None)
    finnish_data.update(synthesis_data)
else:
    print(f"Warning: Synthesis data file not found at {synthesis_data_path}")

# Save merged Finnish data to processed_data
finnish_output_path = os.path.join(processed_data_dir, 'finnish.json')
with open(finnish_output_path, 'w') as json_file:
    json.dump(finnish_data, json_file, indent=4)

print(f"Processed data for Finnish saved to {finnish_output_path}")

# Special processing for English dialects
dialects_to_work = [
    'standard',
    'SoutheastAmericanEnclaveDialect',
    'AfricanAmericanVernacular',
    'ColloquialSingaporeDialect',
    'ChicanoDialect',
    'NigerianDialect',
    'AppalachianDialect',
    'AustralianDialect',
    'HongKongDialect',
    'IndianDialect',
    'IrishDialect'
]

# Load existing synthesis data from english.json
if os.path.exists(english_synthesis_path):
    with open(english_synthesis_path, 'r') as json_file:
        english_data = json.load(json_file)

    # Filter data based on dialects_to_work
    filtered_english_data = {key: value for key, value in english_data.items() if key in dialects_to_work}

    # Save filtered English data to processed_data
    english_output_path = os.path.join(processed_data_dir, 'english.json')
    with open(english_output_path, 'w') as json_file:
        json.dump(filtered_english_data, json_file, indent=4)

    print(f"Processed data for English saved to {english_output_path}")
else:
    print(f"Warning: Synthesis data file not found at {english_synthesis_path}")