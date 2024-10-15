gpu_big="gpu:A100.80gb:1"
gpu_small="gpu:3g.40gb:1"

# job_name="phi3-eng"
# err_file="outputs/${job_name}.err"
# out_file="outputs/${job_name}.out"
# export DATA_PATH=english.json
# export MODEL_ID=microsoft/Phi-3-mini-4k-instruct

# sbatch -J ${job_name} -o ${out_file} -e ${err_file} --gres ${gpu_small} bashRunAll.SLURM

# job_name="nemo-eng"
# err_file="outputs/${job_name}.err"
# out_file="outputs/${job_name}.out"
# export DATA_PATH=english.json
# export MODEL_ID=nvidia/Mistral-NeMo-Minitron-8B-Instruct

# sbatch -J ${job_name} -o ${out_file} -e ${err_file} --gres ${gpu_big} bashRunAll.SLURM

# job_name="aya-eng"
# err_file="outputs/${job_name}.err"
# out_file="outputs/${job_name}.out"
# export DATA_PATH=english.json
# export MODEL_ID=CohereForAI/aya-23-8B

# sbatch -J ${job_name} -o ${out_file} -e ${err_file} --gres ${gpu_big} bashRunAll.SLURM

# job_name="phi3-fin"
# err_file="outputs/${job_name}.err"
# out_file="outputs/${job_name}.out"
# export DATA_PATH=finnish.json
# export MODEL_ID=microsoft/Phi-3-mini-4k-instruct

# sbatch -J ${job_name} -o ${out_file} -e ${err_file} --gres ${gpu_small} bashRunAll.SLURM

# job_name="nemo-fin"
# err_file="outputs/${job_name}.err"
# out_file="outputs/${job_name}.out"
# export DATA_PATH=finnish.json
# export MODEL_ID=nvidia/Mistral-NeMo-Minitron-8B-Instruct

# sbatch -J ${job_name} -o ${out_file} -e ${err_file} --gres ${gpu_big} bashRunAll.SLURM

# job_name="aya-fin"
# err_file="outputs/${job_name}.err"
# out_file="outputs/${job_name}.out"
# export DATA_PATH=finnish.json
# export MODEL_ID=CohereForAI/aya-23-8B

# sbatch -J ${job_name} -o ${out_file} -e ${err_file} --gres ${gpu_big} bashRunAll.SLURM


# job_name="phi3-rest"
# err_file="outputs/${job_name}.err"
# out_file="outputs/${job_name}.out"
# export MODEL_ID=microsoft/Phi-3-mini-4k-instruct

# sbatch -J ${job_name} -o ${out_file} -e ${err_file} --gres ${gpu_small} bashRun.SLURM

# job_name="nemo-rest"
# err_file="outputs/${job_name}.err"
# out_file="outputs/${job_name}.out"
# export MODEL_ID=nvidia/Mistral-NeMo-Minitron-8B-Instruct

# sbatch -J ${job_name} -o ${out_file} -e ${err_file} --gres ${gpu_big} bashRun.SLURM

# job_name="aya-rest"
# err_file="outputs/${job_name}.err"
# out_file="outputs/${job_name}.out"
# export MODEL_ID=CohereForAI/aya-23-8B

# sbatch -J ${job_name} -o ${out_file} -e ${err_file} --gres ${gpu_big} bashRun.SLURM

job_name="phi3-bn"
err_file="outputs/${job_name}.err"
out_file="outputs/${job_name}.out"
export DATA_PATH=bengali.json
export MODEL_ID=microsoft/Phi-3-mini-4k-instruct

sbatch -J ${job_name} -o ${out_file} -e ${err_file} --gres ${gpu_big} bashRunAll.SLURM

job_name="nemo-bn"
err_file="outputs/${job_name}.err"
out_file="outputs/${job_name}.out"
export DATA_PATH=bengali.json
export MODEL_ID=nvidia/Mistral-NeMo-Minitron-8B-Instruct

sbatch -J ${job_name} -o ${out_file} -e ${err_file} --gres ${gpu_big} bashRunAll.SLURM

job_name="aya-bn"
err_file="outputs/${job_name}.err"
out_file="outputs/${job_name}.out"
export DATA_PATH=bengali.json
export MODEL_ID=CohereForAI/aya-23-8B

sbatch -J ${job_name} -o ${out_file} -e ${err_file} --gres ${gpu_big} bashRunAll.SLURM