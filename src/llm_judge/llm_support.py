import logging
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    TrainingArguments, 
    Trainer,
    AutoTokenizer
    )
from huggingface_hub import snapshot_download
from utils import (
    display_chat_generation,
    convert_string_to_json,
)
from prompt_support import (
    formulate_chat_dict,
    formulate_input_seq
)
import re
import os

model_to_type = {
    "CohereForAI/aya-101": AutoModelForSeq2SeqLM
}

def get_model_and_tokenizer(hf_model_path:str, model_id: str, token: str, redownload: bool = False) :
    model_path = os.path.join(hf_model_path, model_id)
    # print(f"{model_path} exists: {os.path.exists(model_path)}")
    if (not os.path.exists(model_path)) or redownload:
        print(f"Trying to download model from {model_id}")
        snapshot_download(
            repo_id=model_id, 
            local_dir=model_path,
            # local_dir_use_symlinks=False,
            token=token
        )
    if model_id in model_to_type:
        autoModel = model_to_type[model_id]
    else:
        autoModel = AutoModelForCausalLM

    model = autoModel.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer

def get_model_and_tokenizer_from_checkpoint(hf_checkpoint) :
    model = AutoModelForCausalLM.from_pretrained(hf_checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(hf_checkpoint)
    return model, tokenizer

def format_output(generated_output: str):
    extracted = re.findall(r'\{.*?\}', generated_output)
    if len(extracted) > 0:
        return extracted[0]
    else:
        return generated_output

def generate_response(input_sentence, model, tokenizer: AutoTokenizer, device, max_new_token=100, model_type='causalLM'):

    gen_config = {
                "temperature": 0.7,
                "top_p": 0.1,
                "repetition_penalty": 1.18,
                "top_k": 5,
                "do_sample": True,
                "max_new_tokens": max_new_token,
                "pad_token_id": tokenizer.eos_token_id
                    }

    if model_type == 'causalLM':
        conversation = formulate_chat_dict(input_sentence, rubrics=None)
        tokenized_input = tokenizer.apply_chat_template(conversation, tokenize=True, add_generation_prompt=True, return_tensors="pt").to(device)
        gen_tokens = model.generate(
                tokenized_input,
                **gen_config
            )
        input_seq = tokenizer.decode(tokenized_input[0])
        output_seq = tokenizer.decode(gen_tokens[0])
    elif model_type == 'seq2seqLM':
        conversation = formulate_input_seq(input_sentence, rubrics=None)
        tokenized_input = tokenizer(conversation, padding=True, truncation=True, return_tensors="pt").to(device)
        gen_tokens = model.generate(
                **tokenized_input,
                **gen_config
            )
        # print(f"INPUT SEQ: {tokenized_input}")
        # print(f"GEN TOKENS: {gen_tokens}")
        input_seq = tokenizer.decode(tokenized_input["input_ids"][0], skip_special_tokens=True)
        output_seq = tokenizer.decode(gen_tokens[0], skip_special_tokens=True)

    if len(input_seq) < len(output_seq):
        output_seq = output_seq[len(input_seq):]
    output_dict = convert_string_to_json(output_seq)
    if output_dict:
        return output_dict
    return output_seq