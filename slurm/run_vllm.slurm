#!/bin/sh
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --gres=gpu:A100.80gb:1
#SBATCH --time=08:00:00
#SBATCH --partition=contrib-gpuq
#SBATCH --qos=gpu

# -------------------------------
# Dynamic Job Info Based on Model
# -------------------------------

MODEL_ID=$1

echo "Running VLLM toxicity detection"
echo "Model ID: $MODEL_ID"
echo "Job ID: $SLURM_JOB_ID"
echo "Started at: $(date)"

# -------------------------------
# Run Python Script
# -------------------------------

source /scratch/ffaisal/test/vnv/bin/activate

cd /scratch/ffaisal/dialect_toxicity_llm_judge/

python src/vllm_predict.py \
  --data_folder /scratch/ffaisal/dialect_toxicity_llm_judge/data/processed_data_gpt_assisted \
  --model_id "$MODEL_ID" \
  --output_dir /scratch/ffaisal/dialect_toxicity_llm_judge/vllm_results_gpt_assisted \
  --num_votes 5 \
  --overwrite

echo "Finished at: $(date)"

deactivate