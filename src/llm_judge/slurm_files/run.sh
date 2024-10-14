export JOB_NAME=english-phi
export DATA_PATH=/projects/klybarge/muhammad_research/toxic_dialect/dialect_toxicity_llm_judge/data/processed_data/english.json
export MODEL_ID=microsoft/Phi-3-mini-4k-instruct

sbatch bashRun.SLURM

export JOB_NAME=english-mistral
export MODEL_ID=mistralai/Mistral-Nemo-Instruct-2407

sbatch bashRun.SLURM