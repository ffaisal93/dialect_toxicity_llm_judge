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
	pip install evaluate
	pip install sacrebleu
	cd ${ROOT_DIR}
	deactivate

fi
if [[ "$task" = "install_vllm" || "$task" = "all" ]]; then
	module load gnu12/12.3.0
	module load python/3.12.1-33
    rm -rf vnv/vnv_vllm
	python -m venv vnv/vnv_vllm
	source vnv/vnv_vllm/bin/activate
	pip install vllm
	pip install --upgrade pip
	pip install flashinfer-python==0.2.2 -i https://flashinfer.ai/whl/cu124/torch2.6/
	pip install pandas
	pip install ipykernel
	python -m ipykernel install --user --name vnv_copa --display-name "vnv_vllm"
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
    OUTPUT_DIR="vllm_results"              # Directory to save results
    PROMPT_TYPE="zeroshot"               # Example: 'oneshot_eng'
    MODEL_NAME="gpt-4.1-2025-04-14"                     # Model name for organizing results
	CUT_OFF="2"

    source /scratch/ffaisal/test/vnv/bin/activate
    # Dialects to process
    DIALECTS=("english" "arabic" "bengali" "chinese" "common_turkic" "finnish" "high_german" "kurdish" "norwegian" "sotho-tswana")

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
            --model_name "${MODEL_NAME}"
			# --cut_off "${CUT_OFF}"
        
        if [ $? -ne 0 ]; then
            echo "Error processing ${DIALECT}. Skipping..."
            continue
        fi
        
        echo "Finished processing ${DIALECT}"
    done

    deactivate

    echo "All specified dialects processed!"

fi

if [[ "$task" = "gpt4_assisted_inference" ]]; then

    # Configuration
    API_KEY_FILE="metadata/my_key.txt"  # Path to the file containing the OpenAI API key
    PROMPT_FILE="src/gpt_inference/prompts.json"              # Path to the prompts file
    INPUT_DIR="data/processed_data_gpt_assisted"         # Directory containing input files
    OUTPUT_DIR="vllm_results_gpt_assisted"              # Directory to save results
    PROMPT_TYPE="zeroshot"               # Example: 'oneshot_eng'
    MODEL_NAME="gpt-4.1-2025-04-14"                     # Model name for organizing results
	CUT_OFF="2"

    source /scratch/ffaisal/test/vnv/bin/activate
    # Dialects to process
    DIALECTS=("arabic" "bengali" "chinese" "common_turkic" "high_german" "kurdish" "norwegian" "sotho-tswana")

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
            --model_name "${MODEL_NAME}"
			# --cut_off "${CUT_OFF}"
        
        if [ $? -ne 0 ]; then
            echo "Error processing ${DIALECT}. Skipping..."
            continue
        fi
        
        echo "Finished processing ${DIALECT}"
    done

    deactivate

    echo "All specified dialects processed!"

fi

if [[ "$task" = "translation_gpt_assisted_inference" ]]; then

    # Configuration
    API_KEY_FILE="metadata/my_key.txt"                            # Path to OpenAI API key
    DATA_PATH="data/processed_data"                               # Input directory
    OUTPUT_PATH="data/processed_data_gpt_assisted"                # Output directory
    LANG_MAP_PATH="metadata/lang_mapping.json"                    # Dialect mapping file
    MODEL_NAME="gpt-4.1-mini-2025-04-14"                           # Model name to use
	MODEL_NAME="gpt-4.1-2025-04-14"

    echo "🔁 Starting translation correction with GPT model: $MODEL_NAME"

	source /scratch/ffaisal/test/vnv/bin/activate

    python3 src/gpt_inference/translation_gpt.py \
        --api_key_file "$API_KEY_FILE" \
        --data_path "$DATA_PATH" \
        --output_path "$OUTPUT_PATH" \
        --lang_map_path "$LANG_MAP_PATH" \
        --model_name "$MODEL_NAME"
	
	deactivate

    if [ $? -ne 0 ]; then
        echo "❌ Error occurred during GPT translation correction"
        exit 1
    fi

    echo "✅ Translation correction completed successfully and saved to: $OUTPUT_PATH"
fi


if [[ "$task" = "nllb_translation_to_english" ]]; then

    # Choose between GPT-assisted or base
    USE_GPT_ASSISTED=false  # set to false for base inference

    if [ "$USE_GPT_ASSISTED" = true ]; then
        DATA_PATH="data/processed_data_gpt_assisted"
        OUTPUT_FILE_NAME="nllb_gpt_assisted.json"
    else
        DATA_PATH="data/processed_data"
        OUTPUT_FILE_NAME="nllb.json"
    fi

    # Shared config
    OUTPUT_PATH="data/translated_to_english"
    MODEL_PATH="/scratch/ffaisal/dialect-copa/models/ct2fast-nllb-200-3.3B"
    DEVICE="cuda"

    echo "🔁 Starting NLLB -> English translation with CTranslate2"
    echo "📂 Input: $DATA_PATH"
    echo "📄 Output file: $OUTPUT_PATH/$OUTPUT_FILE_NAME"

    # Activate virtual environment
    source vnv/vnv_translation/bin/activate

    python3 src/ct2_translate_to_english.py \
        --data_path "$DATA_PATH" \
        --output_path "$OUTPUT_PATH" \
        --output_file_name "$OUTPUT_FILE_NAME" \
        --model_path "$MODEL_PATH" \
        --device "$DEVICE"

    deactivate

    if [ $? -ne 0 ]; then
        echo "❌ Error occurred during NLLB translation"
        exit 1
    fi

    echo "✅ NLLB translation completed successfully and saved to: $OUTPUT_PATH/$OUTPUT_FILE_NAME"
fi


if [[ "$task" = "bleu_evaluation" ]]; then

    # Set to true if you're evaluating the GPT-assisted version
    USE_GPT_ASSISTED=true

    if [ "$USE_GPT_ASSISTED" = true ]; then
        TRANSLATED_FILE="data/translated_to_english/nllb_gpt_assisted.json"
        RESULT_FILE="evaluation_scores/nllb_gpt_assisted_scores.json"
    else
        TRANSLATED_FILE="data/translated_to_english/nllb.json"
        RESULT_FILE="evaluation_scores/nllb_scores.json"
    fi

    REFERENCE_FILE="data/processed_data/english.json"

    echo "🔍 Evaluating BLEU score for translations in: $TRANSLATED_FILE"
    echo "📘 Using gold references from: $REFERENCE_FILE"

    source vnv/vnv_translation/bin/activate

    python3 src/evaluate_bleu_score.py \
        --translated_file "$TRANSLATED_FILE" \
        --reference_file "$REFERENCE_FILE" \
        --result_file "$RESULT_FILE"

    deactivate

    if [ $? -ne 0 ]; then
        echo "❌ BLEU score evaluation failed"
        exit 1
    fi

    echo "✅ BLEU score evaluation completed successfully and saved to: $RESULT_FILE"
fi


if [[ "$task" = "judge_translation_eval" ]]; then

    # Configuration
    MODEL_PATH="/scratch/ffaisal/test/models/Unbabel/M-Prometheus-7B"
    GOLD_FILE="data/processed_data/english.json"
    TRANSLATION_A_FILE="data/translated_to_english/nllb.json"
    TRANSLATION_B_FILE="data/translated_to_english/nllb_gpt_assisted.json"

    echo "🧠 Running translation quality judgment using: $MODEL_PATH"
    echo "📄 A: $TRANSLATION_A_FILE"
    echo "📄 B: $TRANSLATION_B_FILE"
    echo "🎯 Gold: $GOLD_FILE"

    source /scratch/ffaisal/test/vnv/bin/activate

    python3 src/judge_translation_vllm.py \
        --model_path "$MODEL_PATH" \
        --gold_file "$GOLD_FILE" \
        --translation_a_file "$TRANSLATION_A_FILE" \
        --translation_b_file "$TRANSLATION_B_FILE"

    deactivate

    if [ $? -ne 0 ]; then
        echo "❌ Judge evaluation failed"
        exit 1
    fi

    echo "✅ Judge evaluation completed successfully"
fi





if [[ "$task" = "vllm_inference" ]]; then

	# Root directory for models
	model_root="/scratch/ffaisal/test/models"

	# List of model subpaths (relative to model_root)
	models=(
	"Llama-3.1-8B"
	"Qwen2.5-7B-Instruct"
	"Mistral-Nemo-Instruct-2407"
	"google/gemma-3-12b-it"
	"Unbabel/M-Prometheus-7B"
	"Unbabel/TowerInstruct-Mistral-7B-v0.2"
	)

	# Print numbered menu
	echo "Choose a model to run:"
	for i in "${!models[@]}"; do
	printf "  %d) %s\n" $((i+1)) "${models[$i]}"
	done

	# Get user selection
	read -p "Enter the number corresponding to your choice: " selection

	# Validate input
	if ! [[ "$selection" =~ ^[1-6]$ ]]; then
	echo "❌ Invalid selection. Please choose a number between 1 and 4."
	exit 1
	fi

	# Resolve full model ID and safe name
	model_rel_path="${models[$((selection-1))]}"
	model_id="${model_root}/${model_rel_path}"
	model_safe_name=$(basename "$model_rel_path" | tr '/' '_')

	# Ensure logs folder exists
	mkdir -p logs

	# Submit SLURM job with dynamic job name and log paths
	echo "🔁 Submitting SLURM job for model: $model_id"
    echo "logs/vllm_${model_safe_name}.out"
    echo "logs/vllm_${model_safe_name}.err"
	sbatch --job-name="vllm_${model_safe_name}" \
		--output="logs/vllm_${model_safe_name}.out" \
		--error="logs/vllm_${model_safe_name}.err" \
		slurm/run_vllm.slurm "$model_id"


fi


if [[ "$task" = "test" ]]; then


	python test.py \
  --api_key_file metadata/my_key.txt \
  --input_json batch_input.json \
  --output_json corrected_output.json \
  --model_name "gpt-4o-2024-08-06" \
  --max_concurrency 10




  fi
