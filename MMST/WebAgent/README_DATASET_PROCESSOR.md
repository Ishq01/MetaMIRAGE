# Dataset Processor Documentation

## Overview

The `DatasetProcessor` class provides a unified solution for enriching JSON datasets with web search data. It combines the functionality of your two original scripts into a single, optimized class with rate limiting and incremental saves.

## Features

- **Two Processing Modes**:
  - `scraped_data`: Appends full scraped content to questions (Script 1)
  - `summary`: Appends summarized content to questions (Script 2)

- **API Rate Limit Handling**:
  - Tracks API calls and stops when limit is reached (2000 or 2500)
  - Detects HTTP 429 errors from You.com API
  - Saves progress before stopping

- **Incremental Saves**:
  - Saves progress every N entries (default: 100)
  - Prevents data loss on interruption
  - Can resume from existing output file

- **Error Handling**:
  - Graceful error handling for individual entries
  - Continues processing even if some entries fail
  - Tracks errors in statistics

- **Optimized for Large Datasets**:
  - Efficient processing for 25k+ records
  - Progress tracking and status updates
  - Memory-efficient incremental saves

## Installation

No additional dependencies required - uses existing codebase components:
- `WebAgent` class
- `KeywordExtractor`
- `SimpleWebScraper`

## Usage

### Basic Usage

```python
from WebAgent.dataset_processor import DatasetProcessor

# Initialize processor
processor = DatasetProcessor(
    api_key="your_you_com_api_key",
    model_name="Qwen/Qwen2.5-14B-Instruct",
    openai_api_base="http://127.0.0.1:8000/v1",
    api_limit=2000,  # Adjust based on your plan
    save_interval=100  # Save every 100 entries
)

# Process dataset in scraped_data mode
stats = processor.process_dataset(
    input_path="/path/to/input.json",
    output_path="/path/to/output.json",
    mode="scraped_data"
)

# Process dataset in summary mode
stats = processor.process_dataset(
    input_path="/path/to/input.json",
    output_path="/path/to/output.json",
    mode="summary"
)
```

### Command Line Usage

```bash
# Script 1: Scraped data mode
python WebAgent/dataset_processor.py \
    --input /content/standard_training.json \
    --output /content/standard_training_with_scraped_data.json \
    --mode scraped_data \
    --api_key YOUR_API_KEY \
    --api_limit 2000 \
    --save_interval 100

# Script 2: Summary mode
python WebAgent/dataset_processor.py \
    --input /content/sample_standard_benchmark_MG_gpt-4o-mini.json \
    --output /content/sample_standard_benchmark_MG_gpt-4o-mini_with_summary.json \
    --mode summary \
    --api_key YOUR_API_KEY \
    --api_limit 2000 \
    --save_interval 100
```

### Resuming Processing

If processing is interrupted (e.g., API limit reached), you can resume:

```python
# Resume from existing output file
stats = processor.process_dataset(
    input_path="/path/to/input.json",
    output_path="/path/to/output.json",
    mode="scraped_data",
    resume_file="/path/to/output.json"  # Resume from this file
)
```

Or via command line:
```bash
python WebAgent/dataset_processor.py \
    --input /path/to/input.json \
    --output /path/to/output.json \
    --mode scraped_data \
    --api_key YOUR_API_KEY \
    --resume
```

## Processing Modes

### Mode 1: `scraped_data`

Appends full scraped content from web sources to each question:

```json
{
  "question": "Original question\n\n--- Source 1: https://example.com (Title) ---\nFull scraped text...\n\n--- Source 2: ...",
  "keywords": ["keyword1", "keyword2"],
  "search_output": [...]
}
```

### Mode 2: `summary`

Appends a summarized version of web sources to each question:

```json
{
  "question": "Original question\n\n--- Summary of Web Sources for Query: 'keywords' ---\nSummarized content...",
  "keywords": ["keyword1", "keyword2"],
  "search_output": [...]
}
```

## API Rate Limiting

The processor automatically handles You.com API rate limits:

1. **Tracking**: Counts each API call made via `agent.search()`
2. **Limit Detection**: Stops when `api_calls_made >= api_limit`
3. **HTTP 429 Handling**: Catches rate limit errors from API responses
4. **Graceful Stop**: Saves progress before stopping

**Important**: Set `api_limit` based on your You.com plan:
- Free tier: Usually 2000-2500 calls
- Check your plan's exact limit

## Configuration Options

### `DatasetProcessor` Parameters

- `api_key` (str, required): You.com API key
- `model_name` (str, default: "Qwen/Qwen2.5-14B-Instruct"): LLM model name
- `openai_api_base` (str, default: "http://127.0.0.1:8000/v1"): API base URL
- `api_limit` (int, default: 2000): Maximum API calls before stopping
- `save_interval` (int, default: 100): Save progress every N entries

### `process_dataset` Parameters

- `input_path` (str, required): Path to input JSON file
- `output_path` (str, required): Path to output JSON file
- `mode` (str, required): "scraped_data" or "summary"
- `start_from` (int, default: 0): Start processing from this index
- `resume_file` (str, optional): Resume from existing output file

## Output Statistics

The `process_dataset` method returns a dictionary with statistics:

```python
{
    "total_entries": 25000,
    "processed": 2000,
    "skipped": 5,
    "errors": 2,
    "api_calls_made": 2000,
    "rate_limit_reached": True,
    "output_path": "/path/to/output.json"
}
```

## Error Handling

- **Individual Entry Errors**: Logged but don't stop processing
- **API Errors**: Handled gracefully, entry marked with error
- **Rate Limit**: Stops processing, saves progress
- **File Errors**: Raises exception with clear message

## Performance Tips

1. **Save Interval**: Adjust based on dataset size:
   - Small datasets (< 1k): `save_interval=50`
   - Medium datasets (1k-10k): `save_interval=100`
   - Large datasets (10k+): `save_interval=200`

2. **API Limit**: Set conservatively (e.g., 1900 instead of 2000) to avoid hitting exact limit

3. **Resume**: Use resume feature if processing is interrupted

4. **Batch Processing**: For very large datasets, process in batches:
   ```python
   # Process first 2000 entries
   processor.process_dataset(..., start_from=0)
   # Then resume for next batch
   processor.process_dataset(..., resume_file=output_path)
   ```

## Example: Processing 25k Records

```python
processor = DatasetProcessor(
    api_key="your_key",
    api_limit=2000,
    save_interval=100
)

# First batch: 0-2000
stats1 = processor.process_dataset(
    input_path="large_dataset.json",
    output_path="output_batch1.json",
    mode="scraped_data"
)

# Second batch: Resume from batch1
processor2 = DatasetProcessor(
    api_key="your_key",
    api_limit=2000,
    save_interval=100
)
stats2 = processor2.process_dataset(
    input_path="large_dataset.json",
    output_path="output_batch2.json",
    mode="scraped_data",
    resume_file="output_batch1.json"
)

# Continue for remaining batches...
```

## Troubleshooting

### Rate Limit Reached Early
- Check your You.com API plan limits
- Reduce `api_limit` to be conservative
- Check for duplicate API calls

### Processing Too Slow
- Increase `save_interval` to reduce I/O
- Check network connectivity
- Verify local LLM is running efficiently

### Resume Not Working
- Ensure output file exists and is valid JSON
- Check that output file structure matches input
- Verify `resume_file` path is correct

## Comparison: Combined vs Separate Scripts

**Advantages of Combined Approach:**
- ✅ Single codebase to maintain
- ✅ Shared rate limiting logic
- ✅ Consistent error handling
- ✅ Easier to optimize
- ✅ Resume functionality built-in

**For 25k Records:**
- ✅ Better resource management
- ✅ Incremental saves prevent data loss
- ✅ Can process in batches with resume
- ✅ Single configuration point

The combined approach is recommended for your use case.

