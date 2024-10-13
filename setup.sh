#!/bin/bash
task=${task:-none}

while [ $# -gt 0 ]; do

	if [[ $1 == *"--"* ]]; then
		param="${1/--/}"
		declare $param="$2"
		echo $1 $2 #Optional to see the parameter:value result
	fi

	shift
done

if [[ "$task" = "install_llm_judge" ]]; then
	module load gnu10/10.3.0-ya
	module avail python
    module load python/3.10.1-5r
	module load git
	rm -rfd vnv/llm_judge
	python -m venv vnv/llm_judge
	source vnv/llm_judge/bin/activate
	pip install --upgrade pip
    module unload python
    pip3 install torch==2.4.0+cu118 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    pip install -r requirements.txt
    # git clone https://github.com/vllm-project/vllm.git
    # cd vllm
    # python use_existing_torch.py
    # pip install -r requirements-build.txt
    # pip install -e . --no-build-isolation
    # cd ..
    # rm -rf vllm
	# deactivate

fi
