#!/bin/bash
# =============================================================================
# MMST Inference Generation Script
# =============================================================================
# This script runs inference on the MMST benchmark and optionally splits results
# Usage: bash bash_generate.sh
# =============================================================================

# Ensure the script fails if any command fails
set -e

# =============================================================================
# CONFIGURATION - Edit these variables as needed
# =============================================================================

# Benchmark type: "standard" or "contextual"
BENCH_TYPE="standard"
INPUT_FILE="../Datasets/sample_bench/sample_${BENCH_TYPE}_benchmark.json"

# Model configuration
# For OpenAI models: "gpt-4o", "gpt-4o-mini", "gpt-4.1", "gpt-4.1-mini"
# For open-source models (vLLM): "Qwen/Qwen2.5-14B-Instruct", "Qwen/Qwen2.5-VL-3B-Instruct", etc.
# 
# Example for Qwen/Qwen2.5-14B-Instruct:
#   MODEL_NAME='Qwen/Qwen2.5-14B-Instruct'
#   OPENAI_API_BASE='http://127.0.0.1:8000/v1'  # Your vLLM server URL
#
MODEL_NAME='Qwen/Qwen2.5-14B-Instruct'
MODEL_NAME_CLEANED=$(echo "$MODEL_NAME" | sed 's|.*/||')

# API configuration
# For OpenAI models: leave as "None" or empty string ""
# For vLLM/open-source: set to your vLLM server URL
#   Example: "http://127.0.0.1:8000/v1"
#   Make sure your vLLM server is running before starting inference!
#   To start vLLM: python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen2.5-14B-Instruct --port 8000
OPENAI_API_BASE="http://127.0.0.1:8000/v1"

# Parallel processing
NUM_PROCESSES=10

# =============================================================================
# WebAgent Configuration (Optional)
# =============================================================================
# WebAgent enhances prompts with web search context for better accuracy
# Get your SerpAPI key from: https://serpapi.com/
#
# NOTE: By default, WebAgent uses the same MODEL_NAME and OPENAI_API_BASE as above.
#       Only set WEB_AGENT_MODEL_NAME and WEB_AGENT_API_BASE if you want to use
#       a different model/API for WebAgent (e.g., a smaller/faster model for keyword extraction).

USE_WEB_AGENT="false"  # Set to "true" to enable WebAgent
SERPAPI_KEY=""  # Your SerpAPI key (required if USE_WEB_AGENT is true)
WEB_LOCATION="United States"  # Location for web search (affects search results)
WEB_NUM_RESULTS=10  # Number of web search results to retrieve (5-20 recommended)

# Optional: Use different model/API for WebAgent
# If left empty, WebAgent will use MODEL_NAME and OPENAI_API_BASE from above
WEB_AGENT_MODEL_NAME=""  # Leave empty to use MODEL_NAME, or set to e.g., "Qwen/Qwen2.5-14B-Instruct"
WEB_AGENT_API_BASE=""  # Leave empty to use OPENAI_API_BASE, or set to e.g., "http://127.0.0.1:8000/v1"

# =============================================================================
# Script Options
# =============================================================================
SKIP_SPLIT="false"  # Set to "true" to skip the split step after inference

# =============================================================================
# Validation and Setup
# =============================================================================

echo "=============================================================================="
echo "MMST Inference Generation"
echo "=============================================================================="
echo "Benchmark Type: $BENCH_TYPE"
echo "Model: $MODEL_NAME"
echo "Input File: $INPUT_FILE"
if [ "$OPENAI_API_BASE" != "None" ] && [ -n "$OPENAI_API_BASE" ]; then
    echo "API Base: $OPENAI_API_BASE"
else
    echo "API Base: OpenAI (default)"
fi
echo ""

# Check if input file exists
if [ ! -f "$INPUT_FILE" ]; then
    echo "ERROR: Input file not found: $INPUT_FILE"
    echo "Please check the BENCH_TYPE and ensure the dataset file exists."
    exit 1
fi

# Validate API configuration for open-source models
# Check if model name contains "/" (indicates open-source model from HuggingFace)
if [[ "$MODEL_NAME" == *"/"* ]]; then
    if [ "$OPENAI_API_BASE" = "None" ] || [ -z "$OPENAI_API_BASE" ]; then
        echo "ERROR: Open-source model detected ($MODEL_NAME) but OPENAI_API_BASE is not set."
        echo "Please set OPENAI_API_BASE to your vLLM server URL (e.g., 'http://127.0.0.1:8000/v1')."
        echo ""
        echo "To start a vLLM server, run:"
        echo "  python -m vllm.entrypoints.openai.api_server --model $MODEL_NAME --port 8000"
        exit 1
    fi
    
    # Optional: Check if vLLM server is accessible (uncomment to enable)
    # echo "Checking vLLM server connectivity..."
    # if ! curl -s "$OPENAI_API_BASE/models" > /dev/null 2>&1; then
    #     echo "WARNING: Could not reach vLLM server at $OPENAI_API_BASE"
    #     echo "Make sure your vLLM server is running before starting inference."
    # fi
fi

# Validate WebAgent configuration
if [ "$USE_WEB_AGENT" = "true" ]; then
    if [ -z "$SERPAPI_KEY" ]; then
        echo "ERROR: USE_WEB_AGENT is set to 'true' but SERPAPI_KEY is empty."
        echo "Please set your SerpAPI key or disable WebAgent."
        exit 1
    fi
    echo "WebAgent: ENABLED"
    echo "  - Location: $WEB_LOCATION"
    echo "  - Results per query: $WEB_NUM_RESULTS"
    if [ -n "$WEB_AGENT_MODEL_NAME" ]; then
        echo "  - WebAgent Model: $WEB_AGENT_MODEL_NAME (custom)"
    else
        echo "  - WebAgent Model: $MODEL_NAME (same as main model)"
    fi
    if [ -n "$WEB_AGENT_API_BASE" ]; then
        echo "  - WebAgent API Base: $WEB_AGENT_API_BASE (custom)"
    else
        if [ "$OPENAI_API_BASE" != "None" ] && [ -n "$OPENAI_API_BASE" ]; then
            echo "  - WebAgent API Base: $OPENAI_API_BASE (same as main model)"
        else
            echo "  - WebAgent API Base: OpenAI (same as main model)"
        fi
    fi
else
    echo "WebAgent: DISABLED"
fi
echo ""

# =============================================================================
# Prepare Output Directory
# =============================================================================

OUTPUT_DIR="results/${BENCH_TYPE}"
mkdir -p $OUTPUT_DIR
OUTPUT_FILE="${OUTPUT_DIR}/${MODEL_NAME_CLEANED}.json"

echo "Output will be saved to: $OUTPUT_FILE"
echo ""

# =============================================================================
# Build Command Arguments
# =============================================================================

# Base arguments
CMD_ARGS="--input_file $INPUT_FILE"
CMD_ARGS="$CMD_ARGS --output_file $OUTPUT_FILE"
CMD_ARGS="$CMD_ARGS --model_name $MODEL_NAME"
CMD_ARGS="$CMD_ARGS --num_processes $NUM_PROCESSES"

# Handle API base (convert "None" to empty string for Python)
if [ "$OPENAI_API_BASE" = "None" ] || [ -z "$OPENAI_API_BASE" ]; then
    CMD_ARGS="$CMD_ARGS --openai_api_base \"\""
else
    CMD_ARGS="$CMD_ARGS --openai_api_base $OPENAI_API_BASE"
fi

# Add WebAgent arguments if enabled
if [ "$USE_WEB_AGENT" = "true" ]; then
    CMD_ARGS="$CMD_ARGS --use_web_agent"
    CMD_ARGS="$CMD_ARGS --serpapi_key $SERPAPI_KEY"
    CMD_ARGS="$CMD_ARGS --web_location \"$WEB_LOCATION\""
    CMD_ARGS="$CMD_ARGS --web_num_results $WEB_NUM_RESULTS"
    
    if [ -n "$WEB_AGENT_MODEL_NAME" ]; then
        CMD_ARGS="$CMD_ARGS --web_agent_model_name $WEB_AGENT_MODEL_NAME"
    fi
    
    if [ -n "$WEB_AGENT_API_BASE" ]; then
        CMD_ARGS="$CMD_ARGS --web_agent_api_base $WEB_AGENT_API_BASE"
    fi
fi

# =============================================================================
# Run Inference
# =============================================================================

echo "=============================================================================="
echo "Starting Inference..."
echo "=============================================================================="
echo ""

# Run Python script
eval python generate.py $CMD_ARGS

INFERENCE_EXIT_CODE=$?

if [ $INFERENCE_EXIT_CODE -ne 0 ]; then
    echo ""
    echo "ERROR: Inference failed with exit code $INFERENCE_EXIT_CODE"
    exit $INFERENCE_EXIT_CODE
fi

echo ""
echo "=============================================================================="
echo "Inference completed successfully!"
echo "=============================================================================="
echo ""

# =============================================================================
# Split Inference Results (Optional)
# =============================================================================

if [ "$SKIP_SPLIT" = "true" ]; then
    echo "Skipping split step (SKIP_SPLIT=true)"
    exit 0
fi

echo "=============================================================================="
echo "Splitting Inference Results..."
echo "=============================================================================="
echo ""

SPLIT_OUTPUT_FILE="../Datasets/sample_inference"

python split.py \
    --bench_type $BENCH_TYPE \
    --model_name $MODEL_NAME_CLEANED \
    --raw_data_path $INPUT_FILE \
    --results_dir $OUTPUT_DIR \
    --output_dir $SPLIT_OUTPUT_FILE

SPLIT_EXIT_CODE=$?

if [ $SPLIT_EXIT_CODE -ne 0 ]; then
    echo ""
    echo "WARNING: Split step failed with exit code $SPLIT_EXIT_CODE"
    echo "Inference results are still available at: $OUTPUT_FILE"
    exit $SPLIT_EXIT_CODE
fi

echo ""
echo "=============================================================================="
echo "All steps completed successfully!"
echo "=============================================================================="
echo "Inference results: $OUTPUT_FILE"
echo "Split results: $SPLIT_OUTPUT_FILE"
echo ""