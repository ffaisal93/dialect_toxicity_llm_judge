from utils import read_data, save_results
from glob import glob
import ntpath
import os
from tqdm import tqdm

data_dir = './../../data/processed_data'
data_files = glob(f"{data_dir}/*.json")

language_list = {}
for lang_file in data_files:
    _, language = ntpath.split(lang_file)
    language = language.split('.')[0]
    data = read_data(lang_file)
    language_list[language] = list(data.keys())

# print(language_list)

old_result_dir = './../../results'
new_result_dir = './../../results_final'
if not os.path.exists(new_result_dir):
    os.mkdir(new_result_dir)

model_list = os.listdir(old_result_dir)
# print(model_list)

count = 0
for model in model_list:
    old_model_path = os.path.join(old_result_dir,model)
    new_model_path = os.path.join(new_result_dir,model)
    remaining = {}
    if not os.path.exists(new_model_path):
        os.mkdir(new_model_path)
    for lang in language_list:
        remaining[lang] = []
        # print(f"Running for {model} and {lang}")
        dialects = language_list[lang]
        result = {}
        for dialect in dialects:
            old_dialect_path = os.path.join(old_model_path,f"{dialect}.json")
            if os.path.exists(old_dialect_path):
                count += 1
                result[dialect] = read_data(old_dialect_path)
            else:
                remaining[lang].append(dialect)

        save_path = os.path.join(new_model_path,f"{lang}")
        save_results(result, save_path)

    print(f"Remaining dialects for {model}:\n{remaining}")
# print(count)
# print(len(glob(f"{old_result_dir}/*/*.json")))