#!/bin/bash
#SBATCH --partition=gpuq                    # the DGX only belongs in the 'gpu'  partition
#SBATCH --qos=gpu                           # need to select 'gpu' QoS
#SBATCH --job-name=python-gpu
##SBATCH --output=../tr_output/udp_all.out
##SBATCH --error=../tr_output/udp_all.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1                 # up to 128;
#SBATCH --gres=gpu:3g.40gb:1              # up to 8; only request what you need
#SBATCH --mem-per-cpu=90G                 # memory per CORE; total memory is 1 TB (1,000,000 MB)
#SBATCH --export=ALL
#SBATCH --time=3-01:00:00                   # set to 1hr; please choose carefully

cd /scratch/ffaisal/dialect_toxicity_llm_judge

source vnv/vnv_tf/bin/activate
python src/finnish_dialects.py
deactivate
