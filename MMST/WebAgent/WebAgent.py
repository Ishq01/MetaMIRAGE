# ==========================================
# web_context_agent.py
# ==========================================
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import requests
import json
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
        self.base_url = "https://serpapi.com/search.json"

    def search(self, query, location="United States", num_results=10):
        """Extract keywords and perform a SerpAPI search."""
        keywords = self.keyword_extractor.extract_keywords(query)
        search_query = " ".join(keywords)

        params = {
            "engine": "google",
            "q": search_query,
            "location": location,
            "api_key": self.api_key,
            "num": num_results
        }

        try:
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            data = response.json()
            organic_results = data.get("organic_results", [])
            results = [
                {"title": r.get("title"), "link": r.get("link"), "snippet": r.get("snippet")}
                for r in organic_results if r.get("link")
            ]
            self.results_store[query] = results
            print(f"🔍 Found {len(results)} results for '{query}'")
            return results
        except requests.RequestException as e:
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
            # Use the Client's chat method instead of direct API call
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
            # Step 1: Search
            results = self.search(query, location=location, num_results=num_results)
            if not results:
                return ""
            
            # Step 2: Scrape
            text = self.get_text(query)
            if not text:
                return ""
            
            # Step 3: Summarize
            summary = self.summarize_results(query)
            return summary
        except Exception as e:
            print(f"[Error] Web context retrieval failed: {e}")
            return ""
