from multivalue import Dialects

# List of all available dialects in the Dialects module
dialect_classes = [
    Dialects.SoutheastAmericanEnclaveDialect,
    Dialects.WhiteSouthAfricanDialect,
    Dialects.WhiteZimbabweanDialect,
    Dialects.ChicanoDialect,
    Dialects.NewZealandDialect,
    Dialects.NewfoundlandDialect,
    Dialects.NigerianDialect,
    Dialects.AboriginalDialect,
    Dialects.AfricanAmericanVernacular,
    Dialects.AppalachianDialect,
    Dialects.AustralianDialect,
    Dialects.AustralianVernacular,
    Dialects.BahamianDialect,
    Dialects.BlackSouthAfricanDialect,
    Dialects.CameroonDialect,
    Dialects.CapeFlatsDialect,
    Dialects.ChannelIslandsDialect,
    Dialects.ColloquialAmericanDialect,
    Dialects.ColloquialSingaporeDialect,
    Dialects.EarlyAfricanAmericanVernacular,
    Dialects.EastAnglicanDialect,
    Dialects.FalklandIslandsDialect,
    Dialects.FijiAcrolect,
    Dialects.FijiBasilect,
    Dialects.GhanaianDialect,
    Dialects.HongKongDialect,
    Dialects.IndianDialect,
    Dialects.IndianSouthAfricanDialect,
    Dialects.IrishDialect,
    Dialects.JamaicanDialect,
    Dialects.KenyanDialect,
    Dialects.LiberianSettlerDialect,
    Dialects.MalaysianDialect,
    Dialects.MalteseDialect,
    Dialects.ManxDialect,
    Dialects.NorthEnglandDialect,
    Dialects.OrkneyShetlandDialect,
    Dialects.OzarkDialect,
    Dialects.PakistaniDialect,
    Dialects.PhilippineDialect,
    Dialects.RuralAfricanAmericanVernacular,
    Dialects.ScottishDialect,
    Dialects.SoutheastEnglandDialect,
    Dialects.SouthwestEnglandDialect,
    Dialects.SriLankanDialect,
    Dialects.StHelenaDialect,
    Dialects.TanzanianDialect,
    Dialects.TristanDialect,
    Dialects.UgandanDialect,
    Dialects.WelshDialect
]

# Test example sentence
test_sentence = "I talked with them yesterday"

# Dictionary to store transformed sentences
dialect_transformations = {}

# Iterate through each dialect and test transformation
for dialect_class in dialect_classes:
    # Instantiate the dialect
    dialect_instance = dialect_class()
    
    # Transform the test sentence using the dialect
    transformed_sentence = dialect_instance.transform(test_sentence)
    
    # Store the transformed sentence in the dictionary
    dialect_transformations[dialect_class.__name__] = transformed_sentence

# Print the transformed sentences
for dialect, transformed_sentence in dialect_transformations.items():
    print(f"\nDialect: {dialect}")
    print(f"Original: {test_sentence}")
    print(f"Transformed: {transformed_sentence}")