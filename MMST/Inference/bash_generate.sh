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

# WebAgent configuration (optional)
# Set USE_WEB_AGENT to "true" to enable web context enhancement
USE_WEB_AGENT="false"
SERPAPI_KEY=""  # Your SerpAPI key (required if USE_WEB_AGENT is true)
WEB_LOCATION="United States"  # Location for web search
WEB_NUM_RESULTS=10  # Number of web search results to retrieve
WEB_AGENT_MODEL_NAME=""  # Optional: different model for WebAgent (default: same as MODEL_NAME)
WEB_AGENT_API_BASE=""  # Optional: different API base for WebAgent (default: same as OPENAI_API_BASE)

echo "Inference $MODEL_NAME on $BENCH_TYPE Benchmark"

# Inference results will be saved in the following directory
OUTPUT_DIR="results/${BENCH_TYPE}"

mkdir -p $OUTPUT_DIR

OUTPUT_FILE="${OUTPUT_DIR}/${MODEL_NAME_CLEANED}.json"

# Build command arguments
CMD_ARGS="--input_file $INPUT_FILE \
    --output_file $OUTPUT_FILE \
    --model_name $MODEL_NAME \
    --openai_api_base $OPENAI_API_BASE \
    --num_processes $NUM_PROCESSES"

# Add WebAgent arguments if enabled
if [ "$USE_WEB_AGENT" = "true" ]; then
    if [ -z "$SERPAPI_KEY" ]; then
        echo "ERROR: USE_WEB_AGENT is true but SERPAPI_KEY is not set. Please set SERPAPI_KEY."
        exit 1
    fi
    CMD_ARGS="$CMD_ARGS --use_web_agent --serpapi_key $SERPAPI_KEY"
    CMD_ARGS="$CMD_ARGS --web_location $WEB_LOCATION --web_num_results $WEB_NUM_RESULTS"
    if [ -n "$WEB_AGENT_MODEL_NAME" ]; then
        CMD_ARGS="$CMD_ARGS --web_agent_model_name $WEB_AGENT_MODEL_NAME"
    fi
    if [ -n "$WEB_AGENT_API_BASE" ]; then
        CMD_ARGS="$CMD_ARGS --web_agent_api_base $WEB_AGENT_API_BASE"
    fi
    echo "WebAgent enabled: Will enhance prompts with web context"
fi

# Run Python script
python generate.py $CMD_ARGS

################# Split Inference Results #################

SPLIT_OUTPUT_FILE="../Datasets/sample_inference"

python split.py \
    --bench_type $BENCH_TYPE \
    --model_name $MODEL_NAME_CLEANED \
    --raw_data_path $INPUT_FILE \
    --results_dir $OUTPUT_DIR \
    --output_dir $SPLIT_OUTPUT_FILE