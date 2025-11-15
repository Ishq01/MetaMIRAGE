# Quick Start Guide

## Summary

I've created a unified `DatasetProcessor` class that combines both your scripts into one optimized solution with rate limiting and incremental saves.

## Files Created

1. **`dataset_processor.py`** - Main processor class
2. **`example_usage.py`** - Example scripts showing how to use it
3. **`README_DATASET_PROCESSOR.md`** - Full documentation
4. **Updated `WebAgent.py`** - Added rate limit detection

## Quick Usage

### Option 1: Python Script

```python
from WebAgent.dataset_processor import DatasetProcessor

# Initialize
processor = DatasetProcessor(
    api_key="your_you_com_api_key",
    api_limit=2000,  # Adjust based on your plan
    save_interval=100
)

# Script 1: Scraped data mode
processor.process_dataset(
    input_path="/content/standard_training.json",
    output_path="/content/standard_training_with_scraped_data.json",
    mode="scraped_data"
)

# Script 2: Summary mode
processor.process_dataset(
    input_path="/content/sample_standard_benchmark_MG_gpt-4o-mini.json",
    output_path="/content/sample_standard_benchmark_MG_gpt-4o-mini_with_summary.json",
    mode="summary"
)
```

### Option 2: Command Line

```bash
# Script 1
python WebAgent/dataset_processor.py \
    --input /content/standard_training.json \
    --output /content/standard_training_with_scraped_data.json \
    --mode scraped_data \
    --api_key YOUR_API_KEY \
    --api_limit 2000

# Script 2
python WebAgent/dataset_processor.py \
    --input /content/sample_standard_benchmark_MG_gpt-4o-mini.json \
    --output /content/sample_standard_benchmark_MG_gpt-4o-mini_with_summary.json \
    --mode summary \
    --api_key YOUR_API_KEY \
    --api_limit 2000
```

## Key Features

✅ **Rate Limit Handling**: Automatically stops at API limit (2000/2500) and saves progress  
✅ **Incremental Saves**: Saves every 100 entries (configurable)  
✅ **Resume Support**: Can resume from where it stopped  
✅ **Error Handling**: Continues processing even if some entries fail  
✅ **Progress Tracking**: Shows progress and API call count  

## For 25k Records

The combined approach is **recommended** because:
- Better resource management
- Incremental saves prevent data loss
- Can process in batches with resume
- Single codebase to maintain

## Resuming After API Limit

When API limit is reached, the file is saved. To resume:

```python
processor.process_dataset(
    input_path="/path/to/input.json",
    output_path="/path/to/output.json",
    mode="scraped_data",
    resume_file="/path/to/output.json"  # Resume from this file
)
```

Or use `--resume` flag in command line.

## Configuration

- **`api_limit`**: Set to 2000 or 2500 based on your You.com plan
- **`save_interval`**: How often to save (default: 100 entries)
- **`mode`**: "scraped_data" or "summary"

See `README_DATASET_PROCESSOR.md` for full documentation.

