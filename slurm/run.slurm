#!/bin/bash
#SBATCH --partition=contrib-gpuq
#SBATCH --qos=gpu
#SBATCH --output=/scratch/%u/temp_gpt.out # Output file
#SBATCH --error=/scratch/%u/temp_gpt.err # Error file
#SBATCH --mail-type=FAIL    # ALL, NONE, BEGIN, END, FAIL, REQUEUE,..
#SBATCH --mail-user=<GMUnetID>@gmu.edu # Put your GMU email address here
#SBATCH --mem=48GB # Total memory needed per task (units: K,M,G,T)
#SBATCH --cpus-per-task 1 # Request extra CPUs for threads
#SBATCH --ntasks 1 # Number of processes you plan to launch
#SBATCH --gres=gpu:A100.80gb:1
#SBATCH --time=3-01:00:00                   # set to 1hr; please choose carefully

cd /scratch/ffaisal/dialect_toxicity_llm_judge

# source vnv/vnv_tf/bin/activate
# python src/finnish_dialects.py
# deactivate


# ./run.sh --task gpt4_inference
# deactivate

# ./run.sh --task translation_gpt_assisted_inference

./run.sh --task gpt4_assisted_inference