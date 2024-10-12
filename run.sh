#!/bin/bash
#!/bin/bash
task=${task:-none}

mkdir ${RESULT_FOLDER}

while [ $# -gt 0 ]; do

	if [[ $1 == *"--"* ]]; then
		param="${1/--/}"
		declare $param="$2"
		echo $1 $2 #Optional to see the parameter:value result
	fi

	shift
done

if [[ "$task" = "install_dep" ]]; then
	pip install -r requirements.txt
fi

if [[ "$task" = "eng_dialects" ]]; then
	python src/english_dialects.py
fi
