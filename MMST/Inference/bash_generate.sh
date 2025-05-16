#!/bin/bash
# Ensure the script fails if any command fail
set -e

# Benchmark type
# standard
# contextual
BENCH_TYPE="standard"
INPUT_FILE="../Datasets/sample_bench/sample_${BENCH_TYPE}_benchmark.json"

################### Close Source Models ###################
# gpt-4o
# gpt-4o-mini
# gpt-4.1
# gpt-4.1-mini

################### Open Source Models ###################
# Qwen/Qwen2.5-VL-3B-Instruct

MODEL_NAME='gpt-4o-mini'
MODEL_NAME_CLEANED=$(echo "$MODEL_NAME" | sed 's|.*/||')

# You can use VLLM to launch the Open Source Models, remember to change the OPENAI_API_BASE
OPENAI_API_BASE="None"

NUM_PROCESSES=10

echo "Inference $MODEL_NAME on $BENCH_TYPE Benchmark"

# Inference results will be saved in the following directory
OUTPUT_DIR="results/${BENCH_TYPE}"

mkdir -p $OUTPUT_DIR

OUTPUT_FILE="${OUTPUT_DIR}/${MODEL_NAME_CLEANED}.json"
# Run Python script
python generate.py \
    --input_file $INPUT_FILE \
    --output_file $OUTPUT_FILE \
    --model_name $MODEL_NAME \
    --openai_api_base $OPENAI_API_BASE \
    --num_processes $NUM_PROCESSES

################# Split Inference Results #################

SPLIT_OUTPUT_FILE="../Datasets/sample_inference"

python split.py \
    --bench_type $BENCH_TYPE \
    --model_name $MODEL_NAME_CLEANED \
    --raw_data_path $INPUT_FILE \
    --results_dir $OUTPUT_DIR \
    --output_dir $SPLIT_OUTPUT_FILE