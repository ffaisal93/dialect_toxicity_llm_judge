import os
os.environ["HF_HOME"] = "./.cache"
os.environ["TORCH_HOME"] = "./.cache"

import argparse
from prompt_support import formulate_prompt
import torch
from utils import (
    load_config, 
    update_arguments_with_config, 
    get_token,
    display_chat_generation,
    convert_string_to_json
)
from llm_support import (
    get_model_and_tokenizer,
    get_model_and_tokenizer_from_checkpoint,
    format_output,
    generate_response
)
from prompt_support import (
    formulate_chat_dict,
    formulate_chat_dict_grouped
)
from functools import partial
from tqdm import tqdm

print(torch.cuda.get_device_name(0))

def main(args):
    print(args)
    HF_TOKEN = get_token(args.HF_TOKEN_PATH)
    
    if args.hf_checkpoint:
        model, tokenizer = get_model_and_tokenizer_from_checkpoint(args.hf_checkpoint)
    else:
        model, tokenizer = get_model_and_tokenizer(args.hf_model_path, args.model_id, HF_TOKEN)
    device = model.device
    
    # sents = ["when i was a little girl, i use to think that she was the only one in the whole who isn't looking everyone world else"]
    
    
    preds = [generate_response(sent, model=model, tokenizer=tokenizer, device=device) for sent in tqdm(sents)]
    print(preds[0])
    print(preds[1])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='toxicity')
    parser.add_argument('--model_id', default='microsoft/Phi-3-mini-4k-instruct')
    parser.add_argument('--hf_model_path', default='/projects/klybarge/muhammad_research/toxic_dialect/try_dialect_toxicity/hf_models')
    parser.add_argument('--config_path', default='/projects/klybarge/muhammad_research/toxic_dialect/try_dialect_toxicity/config.json')
    parser.add_argument('--cache_path', default='./.cache')
    parser.add_argument('--hf_checkpoint', default=None)
    args = parser.parse_args()

    config_file_path = args.config_path  # Replace with your config file path
    config_data = load_config(config_file_path)

    final_args = update_arguments_with_config(parser, config_data, args)

    main(final_args)