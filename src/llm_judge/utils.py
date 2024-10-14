import os
import json

def get_token(path, complete=False):
    if complete == False:
        base_path = '/projects/klybarge/muhammad_research/'
        path = os.path.join(base_path,path)
    with open(path, 'r') as file:
        # Read the content of the file
        TOKEN = file.read().strip()  # Stripping any unnecessary whitespace/newlines
    return TOKEN

def update_arguments_with_config(parser, config_data, args):
    """
    Update the argument parser values based on the config file and parsed arguments.

    Args:
        parser (ArgumentParser): The argument parser object.
        config_data (dict): Dictionary of configuration values from the config file.
        args (Namespace): Parsed arguments from the command line.

    Returns:
        Namespace: Updated argument values with proper priority.
    """
    if not config_data:
        return parser.parse_args()
    # Convert parsed arguments to a dictionary
    parsed_args = vars(args)

    # Update argument parser defaults based on config file if command line arguments are not provided
    for key, value in config_data.items():
        if key not in parsed_args or parsed_args[key] is None:
            parser.set_defaults(**{key: value})

    # Re-parse the arguments with updated defaults
    final_args = parser.parse_args()
    
    return final_args

def load_config(config_file_path, complete=False):
    """
    Load the configuration file in JSON format and return it as a dictionary.

    Args:
        config_file_path (str): Path to the configuration file.

    Returns:
        dict: Configuration values from the file.
    """
    if complete == False:
        base_path = '/projects/klybarge/muhammad_research/'
        config_file_path = os.path.join(base_path,config_file_path)

    if not os.path.exists(config_file_path):
        return False
    
    with open(config_file_path, 'r') as file:
        config_data = json.load(file)    
    return config_data

def display_chat_generation(input:str, output:str, shortened_output: bool=True, only_output: bool=False)-> None:
    print("*"*100)
    if not only_output:
        print(f"INPUT\n{input}")
        print("*"*100)
    if shortened_output:
        assert len(output) > len(input)
        output = output[len(input):]
    print(f"OUTPUT\n{output}")
    print("*"*100)

def validate_json_keys_and_values(json_data, valid_keys, valid_options):
    # Check if the keys are valid
    json_keys = json_data.keys()
    invalid_keys = [key for key in json_keys if key not in valid_keys]

    if invalid_keys:
        raise KeyError(f"Invalid keys found: {invalid_keys}")

    # Check if the values for specific keys are valid
    for key, valid_vals in valid_options.items():
        if key in json_data:
            if json_data[key] not in valid_vals:
                raise KeyError(f"Invalid value for {key}: {json_data[key]} (Expected one of {valid_vals})")
    
    return True, None

def convert_string_to_json(input_string):
    try:
        # Step 1: Locate the part of the string that resembles JSON
        start_index = input_string.find('{')
        end_index = input_string.rfind('}') + 1

        # Step 2: Extract the JSON-like substring
        json_part = input_string[start_index:end_index]
        
        # Step 3: Validate if it's properly formatted as JSON
        try:
            # Attempt to load the string as JSON
            json_data = json.loads(json_part)
            valid_keys = ['Input Statement', 'Toxicity', 'Target Group Identification', 'Stereotyping', 'Intent', 'Dialectal Sensitivity Impact']
            valid_options = {
                'Toxicity': ['S1', 'S2', 'S3', 'S4', 'S5'],
                'Target Group Identification': ['T0', 'T1', 'T2', 'T3'],
                'Stereotyping': ['ST0', 'ST1', 'ST2', 'ST3'],
                'Intent': ['I1', 'I2', 'I3'],
                'Dialectal Sensitivity Impact': ['D0', 'D1', 'D2', 'D3']
            }
            validate_json_keys_and_values(json_data, valid_keys, valid_options)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
            return None
        except Exception as e:
            print(f"Error: {e}")
            return None

        # Step 4: Return the valid JSON object if no errors
        return json_data

    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def read_data(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def save_results(outputs, path):
    if os.path.exists(path):
        old_results = read_data(path)
        outputs = old_results + outputs
    # Open the file in write mode and write the JSON data to it
    with open(f"{path}.json", 'w') as f:
        json.dump(outputs, f, indent=4)  # indent=4 for pretty-printing