import os
os.environ["HF_HOME"] = "./../../.cache"
os.environ["TORCH_HOME"] = "./../../.cache"

import argparse
from prompt_support import formulate_prompt
import torch
from utils import (
    load_config, 
    update_arguments_with_config, 
    get_token,
    display_chat_generation,
    convert_string_to_json,
    read_data,
    save_results,
    combine_results
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
import pandas as pd



def main(args):
    print(args)
    os.chdir(args.current_dir)
    print(os.listdir())

    print(torch.cuda.get_device_name(0))
    device = torch.device("cuda")
    print(device)

    model_name = args.model_id.split('/')[-1]
    HF_TOKEN = get_token(args.HF_TOKEN_PATH)
    
    if args.hf_checkpoint:
        model, tokenizer = get_model_and_tokenizer_from_checkpoint(args.hf_checkpoint)
    else:
        model, tokenizer = get_model_and_tokenizer(args.hf_model_path, args.model_id, HF_TOKEN)
    tokenizer.pad_token = tokenizer.eos_token
    model.to(device)
    # tokenizer.to(device)

    # sents = ["when i was a little girl, i use to think that she was the only one in the whole who isn't looking everyone world else"]
    data = read_data(args.data_path)
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    if args.dialect_list is None:
        args.dialect_list = list(data.keys())

    first_index = args.sample_count_range_start
    if args.sample_count is None:
        args.sample_count = len(data[args.dialect_list[0]])
    last_index = min(len(data[args.dialect_list[0]]),first_index+args.sample_count)

    model_result_path = os.path.join(args.output_dir, model_name)
    if not os.path.exists(model_result_path):
        os.mkdir(model_result_path)

    for dialect in args.dialect_list:
        output_path = os.path.join(model_result_path, dialect)
        if os.path.exists(f"{output_path}.json") and not args.overwrite:
            print(f"Skipping {dialect} for model {model_name}")
            continue
        preds = [generate_response(sent, model=model, tokenizer=tokenizer, device=device, max_new_token=args.max_new_tokens) for sent in tqdm(data[dialect][first_index:last_index])]
        save_results(preds, output_path)
    
    # combine_results(model_result_path, model_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='toxicity')
    parser.add_argument('--model_id', default='microsoft/Phi-3-mini-4k-instruct')
    parser.add_argument('--hf_model_path', default='./../../hf_models')
    parser.add_argument('--config_path', default='./config.json')
    parser.add_argument('--cache_path', default='./../../.cache')
    parser.add_argument('--hf_checkpoint', default=None)
    parser.add_argument('--data_path', default='./../../data')
    parser.add_argument('--output_dir', default='./../../results')
    parser.add_argument('--dialect_list', nargs='+', default=None)
    parser.add_argument('--sample_count_range_start', type=int, default=0)
    parser.add_argument('--sample_count', type=int, default=None)
    parser.add_argument('--current_dir', default='./')
    parser.add_argument('--overwrite', action=argparse.BooleanOptionalAction)
    parser.add_argument('--max_new_tokens', type=int, default=150)
    args = parser.parse_args()

    config_file_path = args.config_path  # Replace with your config file path
    config_data = load_config(config_file_path)

    final_args = update_arguments_with_config(parser, config_data, args)

    # print(final_args.dialect_list)
    main(final_args)