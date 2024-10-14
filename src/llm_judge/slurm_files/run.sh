gpu_big="gpu:A100.80gb:1"
gpu_small="gpu:3g.40gb:1"

job_name="phi3-eng"
err_file="outputs/${job_name}.err"
out_file="outputs/${job_name}.out"
export DATA_PATH=english.json
export MODEL_ID=microsoft/Phi-3-mini-4k-instruct

sbatch -J ${job_name} -o ${out_file} -e ${err_file} --gres ${gpu_small} bashRun.SLURM

job_name="nemo-eng"
err_file="outputs/${job_name}.err"
out_file="outputs/${job_name}.out"
export DATA_PATH=english.json
export MODEL_ID=nvidia/Mistral-NeMo-Minitron-8B-Instruct

sbatch -J ${job_name} -o ${out_file} -e ${err_file} --gres ${gpu_big} bashRun.SLURM

job_name="phi3-fin"
err_file="outputs/${job_name}.err"
out_file="outputs/${job_name}.out"
export DATA_PATH=finnish.json
export MODEL_ID=microsoft/Phi-3-mini-4k-instruct

sbatch -J ${job_name} -o ${out_file} -e ${err_file} --gres ${gpu_small} bashRun.SLURM

job_name="nemo-fin"
err_file="outputs/${job_name}.err"
out_file="outputs/${job_name}.out"
export DATA_PATH=finnish.json
export MODEL_ID=nvidia/Mistral-NeMo-Minitron-8B-Instruct

sbatch -J ${job_name} -o ${out_file} -e ${err_file} --gres ${gpu_big} bashRun.SLURM