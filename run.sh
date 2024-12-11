#!/bin/bash
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

if [[ "$task" = "install_mvalue" ]]; then
	module load hosts/hopper
	module load gnu10/10.3.0-ya
	module load python/3.10.1-5r
	rm -rf vnv/mvalue
	python -m venv vnv/mvalue
	source vnv/mvalue/bin/activate
	pip install --upgrade pip
	pip install torch==2.0.1+cu117 torchvision==0.15.2+cu117 -f https://download.pytorch.org/whl/torch_stable.html
	pip install -r requirements.txt
	pip install value-nlp
	deactivate
fi

if [[ "$task" = "install_dep" ]]; then
	pip install -r requirements.txt
fi

if [[ "$task" = "eng_dialects" ]]; then
	python src/english_dialects.py
fi

if [[ "$task" = "install_translation" || "$task" = "all" ]]; then
	module load python/3.8.6-ff
	module load git
	rm -rf vnv/vnv_translation
	python -m venv vnv/vnv_translation
	source vnv/vnv_translation/bin/activate
	pip install --upgrade pip
	pip install torch==2.0.1+cu117 torchvision==0.15.2+cu117 -f https://download.pytorch.org/whl/torch_stable.html
	pip install transformers==4.34.0
	pip install ctranslate2 >=3.16.0
	cd ${ROOT_DIR}
	deactivate

fi

if [[ "$task" = "install_tf" || "$task" = "all" ]]; then
	module load python/3.8.6-ff
	module load git
	rm -rf vnv/vnv_tf
	python -m venv vnv/vnv_tf
	source vnv/vnv_tf/bin/activate
	pip install --upgrade pip
	pip3 install tensorflow==2.4.0
	pip install numpy==1.19.2
	pip install spacy==3.1.0
	pip install murre
	python3 -m murre.download

	deactivate

fi

if [[ "$task" = "run_murre_slurm" || "$task" = "all" ]]; then
	mkdir tr_output
	err_file="tr_output/murre.err"
	out_file="tr_output/murre.out"
	echo ${err_file}
	echo ${out_file}
	sbatch -o ${out_file} -e ${err_file} slurm/run.slurm
fi

if [[ "$task" = "gpt4_inference" ]]; then

    # Configuration
    API_KEY_FILE="metadata/my_key.txt"  # Path to the file containing the OpenAI API key
    PROMPT_FILE="src/gpt_inference/prompts.json"              # Path to the prompts file
    INPUT_DIR="data/processed_data"         # Directory containing input files
    OUTPUT_DIR="results_final"              # Directory to save results
    PROMPT_TYPE="oneshot_eng"               # Example: 'oneshot_eng'
    MODEL_NAME="gpt-4o-2024-08-06"                     # Model name for organizing results
	CUT_OFF="2"

    # Dialects to process
    DIALECTS=("english")

    # Loop through each dialect and run the Python script
    for DIALECT in "${DIALECTS[@]}"; do
        INPUT_FILE="${INPUT_DIR}/${DIALECT}.json"  # Path to the input file for the dialect
        echo "Processing dialect: ${DIALECT}"
        
        python3 src/gpt_inference/inference_gpt.py \
            --api_key_file "${API_KEY_FILE}" \
            --prompt_file "${PROMPT_FILE}" \
            --input_file "${INPUT_FILE}" \
            --output_dir "${OUTPUT_DIR}" \
            --prompt_type "${PROMPT_TYPE}" \
            --model_name "${MODEL_NAME}" \
			--cut_off "${CUT_OFF}"
        
        if [ $? -ne 0 ]; then
            echo "Error processing ${DIALECT}. Skipping..."
            continue
        fi
        
        echo "Finished processing ${DIALECT}"
    done

    echo "All specified dialects processed!"

fi


