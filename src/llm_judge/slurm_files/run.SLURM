#!/bin/bash
#SBATCH --partition=contrib-gpuq                    # need to set 'gpuq' or 'contrib-gpuq'  partition
#SBATCH --qos=gpu                           # need to select 'gpu' QOS or other relvant QOS
#SBATCH --job-name=llama-trial-run-2
#SBATCH --output=/projects/klybarge/muhammad_research/toxic_dialect/dialect_toxicity_llm_judge/src/llm_judge/slurm_files/outputs/run1.out   # Output file
#SBATCH --error=/projects/klybarge/muhammad_research/toxic_dialect/dialect_toxicity_llm_judge/src/llm_judge/slurm_files/outputs/run1.err    # Error file
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=12                  # number of cores needed
#SBATCH --gres=gpu:3g.40gb:1                # up to 8; only request what you need
#SBATCH --mem=100GB
#SBATCH --export=ALL
#SBATCH --time=0-10:00:00                   # set to 2hr; please choose carefully

set echo
umask 0027

# to see ID and state of GPUs assigned
nvidia-smi

module load gnu10
module load python

source /projects/klybarge/muhammad_research/toxic_dialect/try_dialect_toxicity/vnv/llm_judge/bin/activate

python /projects/klybarge/muhammad_research/toxic_dialect/dialect_toxicity_llm_judge/src/llm_judge/main.py --dialect_list standard NigerianDialect --sample_count 10

deactivate