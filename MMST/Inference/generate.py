import sys
sys.path.append('../')
from chat_models.OpenAI_Chat import OpenAI_Chat
from chat_models.Client import Client
from pydantic import BaseModel
import json
import multiprocessing
import os
from tqdm import tqdm
import argparse
import time

class Generate:
    def __init__(self, raw_data_file, output_file, model_name="gpt-4o", openai_api_base="", num_processes=None,
                 use_web_agent=False, serpapi_key=None, web_location="United States", web_num_results=10,
                 web_agent_model_name=None, web_agent_api_base=None):
        self.raw_data_file = raw_data_file
        self.output_file = output_file
        self.offline_model = model_name
        self.model_name = model_name.split("/")[-1]
        self.openai_api_base = openai_api_base
        # If the number of processes is not specified, use the number of CPU cores
        self.num_processes = num_processes if num_processes is not None else os.cpu_count()
        
        # WebAgent configuration
        self.use_web_agent = use_web_agent
        self.serpapi_key = serpapi_key
        self.web_location = web_location
        self.web_num_results = web_num_results
        # Use same model/api_base for WebAgent if not specified
        self.web_agent_model_name = web_agent_model_name if web_agent_model_name else model_name
        self.web_agent_api_base = web_agent_api_base if web_agent_api_base else openai_api_base

    def get_prompt(self, item, web_agent=None, web_config=None):
        """Generate prompt for an item, optionally enhanced with web context."""
        question = item["question"]
        user_prompt = f"{question}"
        
        # Optionally enhance prompt with web context
        if web_agent is not None and web_config and web_config.get('use_web_agent'):
            try:
                web_context = web_agent.get_web_context(
                    query=question,
                    location=web_config.get('web_location', 'United States'),
                    num_results=web_config.get('web_num_results', 10)
                )
                if web_context:
                    user_prompt = f"""{question}

Additional context from web search:
{web_context}"""
                    print(f"[WebAgent] Enhanced prompt for item {item.get('id', 'unknown')}")
            except Exception as e:
                print(f"[Warning] WebAgent failed for item {item.get('id', 'unknown')}: {e}. Using original prompt.")
        
        images = item.get("images", [])
        new_images = []
        for i in range(len(images)):
            dir_path = os.path.dirname(os.path.abspath(self.raw_data_file))  
            new_path = dir_path + "/" + images[i]
            if not os.path.exists(new_path):
                print(f"Image path {new_path} does not exist. Please check the input data.")
                continue
            new_images.append(new_path)
        return {"user": user_prompt, "images": new_images}

    # Function to handle item processing
    def process_item(self, args):
        item, model_name, output_file, lock, web_agent_config, web_config = args
        
        # Ensure sys.path is set up for this process (needed for multiprocessing)
        import sys
        import os
        if '../' not in sys.path:
            sys.path.append('../')
        
        # Initialize WebAgent in this process if enabled
        web_agent = None
        if web_config and web_config.get('use_web_agent') and web_agent_config:
            try:
                from WebAgent.WebAgent import WebAgent
                web_agent = WebAgent(
                    api_key=web_agent_config['serpapi_key'],
                    model_name=web_agent_config['model_name'],
                    openai_api_base=web_agent_config['api_base']
                )
            except Exception as e:
                print(f"[Warning] Failed to initialize WebAgent in process: {e}")
        
        # Get prompt with optional web context using the get_prompt method
        prompt = self.get_prompt(item, web_agent=web_agent, web_config=web_config)
        response = None
        last_exception = None
        self.max_retries = 5
        self.retry_delay = 5  # seconds
        item_id = item.get('id', 'unknown')
        for attempt in range(self.max_retries):
            try:
                # Initialize the client based on the model name
                if self.model_name.startswith("gpt"):
                    client = OpenAI_Chat(model_name=model_name, messages=[])
                else:
                    client = Client(model_name=self.offline_model, openai_api_base=self.openai_api_base, messages=[])
                
                response = client.chat(prompt=prompt["user"], images=prompt["images"])
                item[model_name] = response
                # item["info"] = client.info() # Uncomment if needed
                item["history"] = client.get_history()
                break # Exit retry loop on success

            except Exception as e:
                last_exception = e # Store the exception
                print(f"Attempt {attempt + 1}/{self.max_retries} failed for item {item_id}: {e}")
                if attempt < self.max_retries - 1:
                    
                    print(f"Waiting {self.retry_delay} seconds before retrying...")
                    time.sleep(self.retry_delay)
                else:
                    # Max retries reached
                    print(f"Max retries ({self.max_retries}) reached for item {item_id}. Marking as failed.")
                    item[model_name] = -1 # Mark as failed after all retries
 
        with lock:
            with open(output_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        return item.get('id')

    def generate(self):
        # Read the raw data file
        with open(self.raw_data_file, "r", encoding='utf-8') as f:
            data = json.load(f)

        # Check if the output file exists and read processed items
        processed_ids = set()
        if os.path.exists(self.output_file):
            with open(self.output_file, "r", encoding='utf-8') as f:
                for line in f:
                    try:
                        item = json.loads(line)
                        if self.model_name in item and item[self.model_name] != -1 and item[self.model_name] != None:
                            processed_ids.add(item['id'])
                    except json.JSONDecodeError:
                        # Handle potentially corrupt JSON lines
                        continue
                    
        items_to_process = [item for item in data if item.get('id') not in processed_ids]
        print(f"Processing {len(items_to_process)} items.")
        
        if items_to_process:
            manager = multiprocessing.Manager()
            lock = manager.Lock()
            # Prepare WebAgent config to pass to each process
            web_agent_config = None
            web_config = None
            if self.use_web_agent:
                if not self.serpapi_key:
                    print("[Warning] use_web_agent is enabled but serpapi_key is not provided. Disabling WebAgent.")
                    self.use_web_agent = False
                else:
                    web_agent_config = {
                        'serpapi_key': self.serpapi_key,
                        'model_name': self.web_agent_model_name,
                        'api_base': self.web_agent_api_base
                    }
                    web_config = {
                        'use_web_agent': True,
                        'web_location': self.web_location,
                        'web_num_results': self.web_num_results
                    }
            else:
                web_config = {'use_web_agent': False}
            # Initialize the process pool with the specified number of processes
            pool = multiprocessing.Pool(processes=self.num_processes)
            args_list = [(item, self.model_name, self.output_file, lock, web_agent_config, web_config) for item in items_to_process]
            # Use tqdm to show progress
            for _ in tqdm(pool.imap_unordered(self.process_item, args_list), total=len(args_list), desc="Processing items"):
                pass
            pool.close()
            pool.join()
        
        print("Processing completed.")
        self.cleanup_output(len(data))

    def cleanup_output(self, data_length):
        valid_items = []
        
        with open(self.output_file, "r", encoding='utf-8') as f:
            for line in f:
                try:
                    item = json.loads(line)
                    if self.model_name in item and item[self.model_name] != -1 and item[self.model_name] != None:
                        valid_items.append(item)
                except json.JSONDecodeError:
                    continue

        with open(self.output_file, "w", encoding='utf-8') as f:
            for item in valid_items:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

        print(f"Total successful items: {len(valid_items)}. \n Remaining items to process: {data_length - len(valid_items)}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate responses using LLMs model.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input JSON file.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to the output JSONL file.")
    parser.add_argument("--model_name", type=str, default="gpt-4o", help="Model name to use.")
    parser.add_argument("--openai_api_base", type=str, default="", help="Base URL for OpenAI API.")
    parser.add_argument("--num_processes", type=int, default=os.cpu_count(), help="Number of processes to use.")
    
    # WebAgent arguments
    parser.add_argument("--use_web_agent", action="store_true", help="Enable WebAgent to enhance prompts with web context.")
    parser.add_argument("--serpapi_key", type=str, default=None, help="SerpAPI key for web search (required if use_web_agent is enabled).")
    parser.add_argument("--web_location", type=str, default="United States", help="Location for web search (default: United States).")
    parser.add_argument("--web_num_results", type=int, default=10, help="Number of web search results to retrieve (default: 10).")
    parser.add_argument("--web_agent_model_name", type=str, default=None, help="Model name for WebAgent (default: same as --model_name).")
    parser.add_argument("--web_agent_api_base", type=str, default=None, help="API base for WebAgent (default: same as --openai_api_base).")
    
    args = parser.parse_args()

    reformatter = Generate(
        raw_data_file=args.input_file,
        output_file=args.output_file,
        model_name=args.model_name,
        num_processes=args.num_processes,
        openai_api_base=args.openai_api_base,
        use_web_agent=args.use_web_agent,
        serpapi_key=args.serpapi_key,
        web_location=args.web_location,
        web_num_results=args.web_num_results,
        web_agent_model_name=args.web_agent_model_name,
        web_agent_api_base=args.web_agent_api_base
    )
    reformatter.generate()
