#!/bin/bash
#SBATCH --partition=contrib-gpuq                    # need to set 'gpuq' or 'contrib-gpuq'  partition
#SBATCH --qos=gpu                           # need to select 'gpu' QOS or other relvant QOS
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=12                  # number of cores needed
#SBATCH --gres=gpu:A100.80gb:1                # up to 8; only request what you need
#SBATCH --mem=100GB
#SBATCH --export=ALL
#SBATCH --time=0-23:00:00                   # set to 2hr; please choose carefully

set echo
umask 0027

# to see ID and state of GPUs assigned
nvidia-smi

module load gnu10
module load python

cd ..
cd ..
cd ..

source vnv/llm_judge/bin/activate

cd src/llm_judge

python main.py \
    --data_path ./../../data/processed_data/arabic.json \
    --model_id ${MODEL_ID} \
    --current_dir /scratch/mrahma45/dialect_toxicity_llm_judge/src/llm_judge

python main.py \
    --data_path ./../../data/processed_data/chinese.json \
    --model_id ${MODEL_ID} \
    --current_dir /scratch/mrahma45/dialect_toxicity_llm_judge/src/llm_judge

python main.py \
    --data_path ./../../data/processed_data/common_turkic.json \
    --model_id ${MODEL_ID} \
    --current_dir /scratch/mrahma45/dialect_toxicity_llm_judge/src/llm_judge

python main.py \
    --data_path ./../../data/processed_data/high_german.json \
    --model_id ${MODEL_ID} \
    --current_dir /scratch/mrahma45/dialect_toxicity_llm_judge/src/llm_judge

python main.py \
    --data_path ./../../data/processed_data/kurdish.json \
    --model_id ${MODEL_ID} \
    --current_dir /scratch/mrahma45/dialect_toxicity_llm_judge/src/llm_judge

python main.py \
    --data_path ./../../data/processed_data/norwegian.json \
    --model_id ${MODEL_ID} \
    --current_dir /scratch/mrahma45/dialect_toxicity_llm_judge/src/llm_judge

python main.py \
    --data_path ./../../data/processed_data/sotho-tswana.json \
    --model_id ${MODEL_ID} \
    --current_dir /scratch/mrahma45/dialect_toxicity_llm_judge/src/llm_judge

deactivate