import pandas as pd
import json
import os
from multivalue import Dialects
from tqdm import tqdm

# List of all available dialects in the Dialects module
dialect_classes = [
    # Dialects.SoutheastAmericanEnclaveDialect,
    # Dialects.WhiteSouthAfricanDialect,
    Dialects.WhiteZimbabweanDialect,
    Dialects.ChicanoDialect,
    Dialects.NewZealandDialect,
    Dialects.NewfoundlandDialect,
    Dialects.NigerianDialect,
    Dialects.AboriginalDialect,
    # Dialects.AfricanAmericanVernacular,
    # Dialects.AppalachianDialect,
    # Dialects.AustralianDialect,
    # Dialects.AustralianVernacular,
    # Dialects.BahamianDialect,
    # Dialects.BlackSouthAfricanDialect,
    # Dialects.CameroonDialect,
    # Dialects.CapeFlatsDialect,
    # Dialects.ChannelIslandsDialect,
    # Dialects.ColloquialAmericanDialect,
    # Dialects.ColloquialSingaporeDialect,
    # Dialects.EarlyAfricanAmericanVernacular,
    # Dialects.EastAnglicanDialect,
    # Dialects.FalklandIslandsDialect,
    # Dialects.FijiAcrolect,
    # Dialects.FijiBasilect,
    # Dialects.GhanaianDialect,
    # Dialects.HongKongDialect,
    # Dialects.IndianDialect,
    # Dialects.IndianSouthAfricanDialect,
    # Dialects.IrishDialect,
    # Dialects.JamaicanDialect,
    # Dialects.KenyanDialect,
    # Dialects.LiberianSettlerDialect,
    # Dialects.MalaysianDialect,
    # Dialects.MalteseDialect,
    # Dialects.ManxDialect,
    # Dialects.NorthEnglandDialect,
    # Dialects.OrkneyShetlandDialect,
    # Dialects.OzarkDialect,
    # Dialects.PakistaniDialect,
    # Dialects.PhilippineDialect,
    # Dialects.RuralAfricanAmericanVernacular,
    # Dialects.ScottishDialect,
    # Dialects.SoutheastEnglandDialect,
    # Dialects.SouthwestEnglandDialect,
    # Dialects.SriLankanDialect,
    # Dialects.StHelenaDialect,
    # Dialects.TanzanianDialect,
    # Dialects.TristanDialect,
    # Dialects.UgandanDialect,
    # Dialects.WelshDialect
]

# Load the dataset
df = pd.read_parquet('data/toxigen/test-00000-of-00001.parquet')

# Path to save the transformed data
output_path = 'data/synthesis/english.json'

# Ensure output directory exists
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# Load existing data if file already exists
if os.path.exists(output_path):
    with open(output_path, 'r') as json_file:
        dialect_transformations = json.load(json_file)
else:
    # Initialize with standard sentences if file does not exist
    dialect_transformations = {
        "standard": df['text'].tolist()
    }

# Iterate through each dialect and transform each sentence
for dialect_class in dialect_classes:
    dialect_name = dialect_class.__name__
    
    # Skip if dialect is already processed
    if dialect_name in dialect_transformations:
        continue

    print(f"Processing {dialect_name}...")

    # Instantiate the dialect
    dialect_instance = dialect_class()
    
    # Transform all sentences in df['text'] with try-except
    transformed_sentences = []
    for sentence in tqdm(df['text'], desc=f"Transforming {dialect_name}"):
        try:
            transformed_sentences.append(dialect_instance.transform(sentence))
        except Exception as e:
            transformed_sentences.append('')  # Use empty string as placeholder
    
    # Store the transformed sentences in the dictionary
    dialect_transformations[dialect_name] = transformed_sentences

    # Save the dictionary as JSON after each iteration
    with open(output_path, 'w') as json_file:
        json.dump(dialect_transformations, json_file, indent=4)

    print(f"Saved transformations for {dialect_name}")

print(f"All transformations saved to {output_path}")