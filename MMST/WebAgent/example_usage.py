# ==========================================
# example_usage.py
# ==========================================
"""
Example usage scripts for DatasetProcessor.

This file contains two example scripts that demonstrate how to use
the DatasetProcessor class for both modes:
1. Script 1: Scraped data mode
2. Script 2: Summary mode
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from WebAgent.dataset_processor import DatasetProcessor

# ==========================================
# Configuration
# ==========================================
API_KEY = "your_you_com_api_key_here"  # Replace with your You.com API key
MODEL_NAME = "Qwen/Qwen2.5-14B-Instruct"
API_BASE = "http://127.0.0.1:8000/v1"
API_LIMIT = 2000  # Adjust based on your You.com plan (2000 or 2500)
SAVE_INTERVAL = 100  # Save every 100 entries


# ==========================================
# Script 1: Scraped Data Mode
# ==========================================
def script1_scraped_data():
    """
    Script 1: Process dataset with full scraped content appended to questions.
    This matches your original Script1 functionality.
    """
    input_path = "/content/standard_training.json"
    output_path = "/content/standard_training_with_scraped_data.json"
    
    # Initialize processor
    processor = DatasetProcessor(
        api_key=API_KEY,
        model_name=MODEL_NAME,
        openai_api_base=API_BASE,
        api_limit=API_LIMIT,
        save_interval=SAVE_INTERVAL
    )
    
    # Process dataset
    stats = processor.process_dataset(
        input_path=input_path,
        output_path=output_path,
        mode="scraped_data"
    )
    
    print("\n" + "="*50)
    print("Script 1 Complete - Scraped Data Mode")
    print("="*50)
    print(f"Output saved to: {output_path}")
    return stats


# ==========================================
# Script 2: Summary Mode
# ==========================================
def script2_summary():
    """
    Script 2: Process dataset with summarized content appended to questions.
    This matches your original Script2 functionality.
    """
    input_path = "/content/sample_standard_benchmark_MG_gpt-4o-mini.json"
    output_path = "/content/sample_standard_benchmark_MG_gpt-4o-mini_with_summary.json"
    
    # Initialize processor
    processor = DatasetProcessor(
        api_key=API_KEY,
        model_name=MODEL_NAME,
        openai_api_base=API_BASE,
        api_limit=API_LIMIT,
        save_interval=SAVE_INTERVAL
    )
    
    # Process dataset
    stats = processor.process_dataset(
        input_path=input_path,
        output_path=output_path,
        mode="summary"
    )
    
    print("\n" + "="*50)
    print("Script 2 Complete - Summary Mode")
    print("="*50)
    print(f"Output saved to: {output_path}")
    return stats


# ==========================================
# Resume Processing Example
# ==========================================
def resume_processing():
    """
    Example of how to resume processing from where it left off.
    Useful when API limit is reached or processing is interrupted.
    """
    input_path = "/content/standard_training.json"
    output_path = "/content/standard_training_with_scraped_data.json"
    resume_file = output_path  # Resume from existing output
    
    processor = DatasetProcessor(
        api_key=API_KEY,
        model_name=MODEL_NAME,
        openai_api_base=API_BASE,
        api_limit=API_LIMIT,
        save_interval=SAVE_INTERVAL
    )
    
    # Process dataset with resume
    stats = processor.process_dataset(
        input_path=input_path,
        output_path=output_path,
        mode="scraped_data",
        resume_file=resume_file
    )
    
    return stats


# ==========================================
# Main Execution
# ==========================================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Process dataset with web search")
    parser.add_argument("--script", type=int, choices=[1, 2], required=True,
                       help="Which script to run: 1 (scraped_data) or 2 (summary)")
    parser.add_argument("--input", type=str, help="Input JSON file path (overrides default)")
    parser.add_argument("--output", type=str, help="Output JSON file path (overrides default)")
    parser.add_argument("--api_key", type=str, help="You.com API key (overrides default)")
    parser.add_argument("--api_limit", type=int, default=2000, help="API call limit")
    parser.add_argument("--resume", action="store_true", help="Resume from existing output file")
    
    args = parser.parse_args()
    
    # Override API key if provided
    if args.api_key:
        API_KEY = args.api_key
    
    # Override API limit if provided
    if args.api_limit:
        API_LIMIT = args.api_limit
    
    if args.script == 1:
        if args.input and args.output:
            # Custom paths
            processor = DatasetProcessor(
                api_key=API_KEY,
                model_name=MODEL_NAME,
                openai_api_base=API_BASE,
                api_limit=API_LIMIT,
                save_interval=SAVE_INTERVAL
            )
            stats = processor.process_dataset(
                input_path=args.input,
                output_path=args.output,
                mode="scraped_data",
                resume_file=args.output if args.resume else None
            )
        else:
            stats = script1_scraped_data()
    else:  # script == 2
        if args.input and args.output:
            # Custom paths
            processor = DatasetProcessor(
                api_key=API_KEY,
                model_name=MODEL_NAME,
                openai_api_base=API_BASE,
                api_limit=API_LIMIT,
                save_interval=SAVE_INTERVAL
            )
            stats = processor.process_dataset(
                input_path=args.input,
                output_path=args.output,
                mode="summary",
                resume_file=args.output if args.resume else None
            )
        else:
            stats = script2_summary()
    
    print("\n" + "="*50)
    print("Final Statistics:")
    print("="*50)
    for key, value in stats.items():
        print(f"  {key}: {value}")

