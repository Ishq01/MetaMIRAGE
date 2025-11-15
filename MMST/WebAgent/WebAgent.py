# ==========================================
# web_context_agent.py
# ==========================================
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import requests
from chat_models.Client import Client
from Utils.keyword_extractor import KeywordExtractor
from Utils.web_scraper import SimpleWebScraper

class WebAgent:
    """Performs a web search, extracts text, and summarizes using local models."""

    def __init__(self, api_key, model_name="Qwen/Qwen2.5-14B-Instruct", openai_api_base="http://127.0.0.1:8000/v1"):
        self.api_key = api_key
        self.results_store = {}
        self.combined_text = {}
        self.keyword_extractor = KeywordExtractor(model_name, openai_api_base)
        self.web_scraper = SimpleWebScraper()
        self.client = Client(model_name=model_name, openai_api_base=openai_api_base)

        # OLD SerpAPI endpoint (not used anymore)
        # self.base_url = "https://serpapi.com/search.json"

        # NEW You.com API endpoint
        self.base_url = "https://api.you.com/search"

    def search(self, query, location="United States", num_results=10):
        """Extract keywords and perform a You.com Search API call."""
        keywords = self.keyword_extractor.extract_keywords(query)
        search_query = " ".join(keywords)

        # Build request body for You.com API
        request_body = {
            "query": search_query,
            "num_web_results": num_results,
            "page": 1,
            "recency": 365,
            "include_domains": []
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        try:
            response = requests.post(self.base_url, headers=headers, json=request_body)
            
            # Check for rate limit (429 Too Many Requests)
            if response.status_code == 429:
                print(f"⚠️  Rate limit reached (429). API call limit exceeded.")
                raise RateLimitError("You.com API rate limit exceeded (429)")
            
            response.raise_for_status()

            data = response.json()
            web_results = data.get("results", {}).get("web", [])

            # Format to match old SerpAPI structure
            results = [
                {
                    "title": r.get("title"),
                    "link": r.get("url"),
                    "snippet": r.get("description")
                        or (r.get("snippets")[0] if r.get("snippets") and len(r.get("snippets")) > 0 else "")
                }
                for r in web_results if r.get("url")
            ]

            self.results_store[query] = results
            print(f"🔍 Found {len(results)} results for '{search_query}'")
            return results

        except requests.RequestException as e:
            # Check if it's a rate limit error from response
            if hasattr(e, 'response') and e.response is not None:
                if e.response.status_code == 429:
                    print(f"⚠️  Rate limit reached (429). API call limit exceeded.")
                    raise RateLimitError("You.com API rate limit exceeded (429)")
            print(f"[Error] Web search failed: {e}")
            return []

    def get_text(self, query):
        """Scrape all result URLs and combine text."""
        self.combined_text[query] = ""
        for r in self.results_store.get(query, []):
            try:
                text_data = self.web_scraper.scrape([r["link"]])
                if text_data and r["link"] in text_data:
                    self.combined_text[query] += text_data[r["link"]]["text"] + "\n"
            except Exception as e:
                print(f"[Error] Scrape failed for {r['link']}: {e}")
        return self.combined_text[query]

    def summarize_results(self, query):
        """Summarize the scraped text into one factual paragraph."""
        text_to_summarize = self.combined_text.get(query, "")
        if not text_to_summarize:
            print("No scraped text to summarize.")
            return ""

        prompt = f"""
        Summarize the following web content into a single, factual paragraph.
        Focus on key facts, statistics, or findings related to the user's query.
        Write concisely, neutrally, and in a style suitable for use as external context.

        Text to summarize:
        {text_to_summarize[:5000]}  # limit for token safety
        """

        try:
            summary = self.client.chat(prompt=prompt, images=[])
            print("✅ Summary generated successfully.")
            return summary.strip()
        except Exception as e:
            print(f"[Error] Summarization failed: {e}")
            return ""

    def get_web_context(self, query, location="United States", num_results=10):
        """
        Complete pipeline: search -> scrape -> summarize.
        Returns a summary string that can be appended to prompts.
        """
        try:
            results = self.search(query, location=location, num_results=num_results)
            if not results:
                return ""
            
            text = self.get_text(query)
            if not text:
                return ""
            
            summary = self.summarize_results(query)
            return summary
        except Exception as e:
            print(f"[Error] Web context retrieval failed: {e}")
            return ""


class RateLimitError(Exception):
    """Raised when API rate limit is exceeded."""
    pass
