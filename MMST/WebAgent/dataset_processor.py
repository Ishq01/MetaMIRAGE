# ==========================================
# dataset_processor.py
# ==========================================
"""
Dataset processor for enriching JSON datasets with web search data.
Supports two modes:
1. 'scraped_data': Appends full scraped content to questions
2. 'summary': Appends summarized content to questions

Features:
- API rate limit tracking and handling
- Incremental saves to prevent data loss
- Error handling and recovery
- Progress tracking for large datasets
"""
import sys
import os
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from WebAgent.WebAgent import WebAgent, RateLimitError


class DatasetProcessor:
    """Processes JSON datasets by enriching entries with web search data."""
    
    def __init__(
        self,
        api_key: str,
        model_name: str = "Qwen/Qwen2.5-14B-Instruct",
        openai_api_base: str = "http://127.0.0.1:8000/v1",
        api_limit: int = 2000,
        save_interval: int = 100
    ):
        """
        Initialize the dataset processor.
        
        Args:
            api_key: You.com API key
            model_name: Model name for LLM operations
            openai_api_base: API base URL for local models
            api_limit: Maximum number of API calls before stopping (default: 2000)
            save_interval: Number of entries to process before saving (default: 100)
        """
        self.agent = WebAgent(api_key=api_key, model_name=model_name, openai_api_base=openai_api_base)
        self.api_limit = api_limit
        self.api_calls_made = 0
        self.save_interval = save_interval
        self.rate_limit_reached = False
        
    def process_entry_scraped_data(self, entry: Dict[str, Any], query: str) -> Dict[str, Any]:
        """
        Process a single entry in 'scraped_data' mode.
        Appends full scraped content to the question field.
        """
        question_text = entry.get("question", "")
        
        try:
            # Perform web search
            search_output = self.agent.search(query)
            self.api_calls_made += 1
            
            # Check rate limit
            if self.api_calls_made >= self.api_limit:
                print(f"⚠️  API call limit reached ({self.api_limit}). Stopping processing.")
                self.rate_limit_reached = True
                return entry
            
            entry["search_output"] = search_output
            
            # Extract URLs from search results
            urls = [item.get("link") for item in search_output if item.get("link")]
            
            # Scrape webpage content
            scraped_data = self.agent.web_scraper.scrape(urls)
            
            # Append scraped text directly to the question field
            appended_text = question_text.strip() + "\n\n"
            
            for idx, url in enumerate(urls, start=1):
                if url in scraped_data:
                    title = scraped_data[url].get("title", "No Title")
                    text = scraped_data[url].get("text", "")
                    if text.strip():
                        appended_text += f"--- Source {idx}: {url} ({title}) ---\n{text}\n\n"
            
            entry["question"] = appended_text.strip()
            
        except RateLimitError:
            print(f"⚠️  Rate limit error detected. Stopping processing.")
            self.rate_limit_reached = True
            entry["search_output"] = "ERROR: Rate limit exceeded"
        except Exception as e:
            print(f"[Error] Failed to process entry: {e}")
            entry["search_output"] = f"ERROR: {e}"
        
        return entry
    
    def process_entry_summary(self, entry: Dict[str, Any], query: str) -> Dict[str, Any]:
        """
        Process a single entry in 'summary' mode.
        Appends summarized content to the question field.
        """
        question_text = entry.get("question", "")
        
        try:
            # Perform web search
            search_output = self.agent.search(query)
            self.api_calls_made += 1
            
            # Check rate limit
            if self.api_calls_made >= self.api_limit:
                print(f"⚠️  API call limit reached ({self.api_limit}). Stopping processing.")
                self.rate_limit_reached = True
                return entry
            
            entry["search_output"] = search_output
            
            # Extract URLs from search results
            urls = [item.get("link") for item in search_output if item.get("link")]
            
            # Scrape webpages to populate agent's combined_text cache
            # This is needed for summarize_results to work
            for url in urls:
                try:
                    text_data = self.agent.web_scraper.scrape([url])
                    if text_data and url in text_data:
                        # Store in combined_text using query as key
                        if query not in self.agent.combined_text:
                            self.agent.combined_text[query] = ""
                        self.agent.combined_text[query] += text_data[url].get("text", "") + "\n"
                except Exception as e:
                    print(f"[Warning] Failed to scrape {url}: {e}")
            
            # Generate summary using the agent's built-in summarizer
            summary_text = self.agent.summarize_results(query)
            
            # Append the summary to the question text
            appended_text = question_text.strip() + "\n\n"
            appended_text += f"--- Summary of Web Sources for Query: '{query}' ---\n"
            appended_text += summary_text.strip()
            
            entry["question"] = appended_text.strip()
            
        except RateLimitError:
            print(f"⚠️  Rate limit error detected. Stopping processing.")
            self.rate_limit_reached = True
            entry["search_output"] = "ERROR: Rate limit exceeded"
            entry["summary_error"] = "Rate limit exceeded"
        except Exception as e:
            print(f"[Error] Failed to process entry: {e}")
            entry["search_output"] = f"ERROR: {e}"
            entry["summary_error"] = str(e)
        
        return entry
    
    def process_dataset(
        self,
        input_path: str,
        output_path: str,
        mode: str = "scraped_data",
        start_from: int = 0,
        resume_file: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process the entire dataset.
        
        Args:
            input_path: Path to input JSON file
            output_path: Path to output JSON file
            mode: Processing mode - 'scraped_data' or 'summary'
            start_from: Index to start processing from (for resuming)
            resume_file: Optional path to resume from existing output file
            
        Returns:
            Dictionary with processing statistics
        """
        if mode not in ["scraped_data", "summary"]:
            raise ValueError(f"Invalid mode: {mode}. Must be 'scraped_data' or 'summary'")
        
        # Load input data
        print(f"📂 Loading dataset from: {input_path}")
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        total_entries = len(data)
        print(f"📊 Total entries to process: {total_entries}")
        
        # Resume from existing file if provided
        processed_data = []
        initial_processed = 0
        if resume_file and os.path.exists(resume_file):
            print(f"📂 Resuming from: {resume_file}")
            with open(resume_file, "r", encoding="utf-8") as f:
                processed_data = json.load(f)
            initial_processed = len(processed_data)
            start_from = initial_processed
            print(f"📊 Resuming from entry {start_from} (already processed: {initial_processed})")
        
        # Process entries
        processed_count = 0
        skipped_count = 0
        error_count = 0
        
        for idx, entry in enumerate(data[start_from:], start=start_from):
            # Check if we've hit the rate limit
            if self.rate_limit_reached or self.api_calls_made >= self.api_limit:
                print(f"\n⚠️  API limit reached at entry {idx}/{total_entries}")
                print(f"   Processed: {processed_count}, Skipped: {skipped_count}, Errors: {error_count}")
                print(f"   API calls made: {self.api_calls_made}/{self.api_limit}")
                break
            
            question_text = entry.get("question", "")
            if not question_text:
                print(f"[Warning] Entry {idx} has no question, skipping...")
                skipped_count += 1
                processed_data.append(entry)
                continue
            
            # Extract keywords
            try:
                keywords = self.agent.keyword_extractor.extract_keywords(question_text)
                entry["keywords"] = keywords
                query = " ".join(keywords)
            except Exception as e:
                print(f"[Error] Keyword extraction failed for entry {idx}: {e}")
                entry["keywords"] = []
                query = question_text  # Fallback to original question
            
            # Process entry based on mode
            try:
                if mode == "scraped_data":
                    entry = self.process_entry_scraped_data(entry, query)
                else:  # summary mode
                    entry = self.process_entry_summary(entry, query)
                
                processed_count += 1
                
            except Exception as e:
                print(f"[Error] Processing failed for entry {idx}: {e}")
                error_count += 1
                entry["processing_error"] = str(e)
            
            processed_data.append(entry)
            
            # Incremental save
            if (idx + 1) % self.save_interval == 0:
                self._save_progress(output_path, processed_data)
                print(f"💾 Progress saved: {idx + 1}/{total_entries} entries processed")
                print(f"   API calls: {self.api_calls_made}/{self.api_limit}")
            
            # Progress update
            if (idx + 1) % 10 == 0:
                print(f"⏳ Progress: {idx + 1}/{total_entries} | API calls: {self.api_calls_made}/{self.api_limit}")
        
        # Final save
        print(f"\n💾 Saving final results...")
        self._save_progress(output_path, processed_data)
        
        stats = {
            "total_entries": total_entries,
            "processed": processed_count,
            "skipped": skipped_count,
            "errors": error_count,
            "initial_processed": initial_processed,
            "total_processed": initial_processed + processed_count,
            "api_calls_made": self.api_calls_made,
            "rate_limit_reached": self.rate_limit_reached,
            "output_path": output_path
        }
        
        print(f"\n✅ Processing complete!")
        print(f"   Newly processed: {processed_count}")
        print(f"   Previously processed: {initial_processed}")
        print(f"   Total processed: {initial_processed + processed_count}")
        print(f"   Skipped: {skipped_count}")
        print(f"   Errors: {error_count}")
        print(f"   API calls made: {self.api_calls_made}/{self.api_limit}")
        print(f"   Output saved to: {output_path}")
        
        return stats
    
    def _save_progress(self, output_path: str, processed_data: List[Dict]):
        """Save progress to output file."""
        # Write to file
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(processed_data, f, indent=2, ensure_ascii=False)


def main():
    """Example usage of DatasetProcessor."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Process dataset with web search enrichment")
    parser.add_argument("--input", type=str, required=True, help="Input JSON file path")
    parser.add_argument("--output", type=str, required=True, help="Output JSON file path")
    parser.add_argument("--mode", type=str, choices=["scraped_data", "summary"], default="scraped_data",
                       help="Processing mode: 'scraped_data' or 'summary'")
    parser.add_argument("--api_key", type=str, required=True, help="You.com API key")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-14B-Instruct",
                       help="Model name for LLM operations")
    parser.add_argument("--api_base", type=str, default="http://127.0.0.1:8000/v1",
                       help="API base URL for local models")
    parser.add_argument("--api_limit", type=int, default=2000,
                       help="Maximum API calls before stopping (default: 2000)")
    parser.add_argument("--save_interval", type=int, default=100,
                       help="Save progress every N entries (default: 100)")
    parser.add_argument("--resume", type=str, default=None,
                       help="Resume from existing output file")
    parser.add_argument("--start_from", type=int, default=0,
                       help="Start processing from this index")
    
    args = parser.parse_args()
    
    # Initialize processor
    processor = DatasetProcessor(
        api_key=args.api_key,
        model_name=args.model_name,
        openai_api_base=args.api_base,
        api_limit=args.api_limit,
        save_interval=args.save_interval
    )
    
    # Process dataset
    stats = processor.process_dataset(
        input_path=args.input,
        output_path=args.output,
        mode=args.mode,
        start_from=args.start_from,
        resume_file=args.resume
    )
    
    print("\n" + "="*50)
    print("Processing Statistics:")
    print("="*50)
    for key, value in stats.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()

