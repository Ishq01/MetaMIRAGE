# ==========================================
# keyword_extractor.py
# ==========================================
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import re
from chat_models.Client import Client  # <-- your local model client wrapper

class KeywordExtractor:
    """Extracts and organizes keywords from a user query using local LLM via Client."""

    def __init__(self, model_name="Qwen/Qwen2.5-14B-Instruct", openai_api_base="http://127.0.0.1:8000/v1"):
        self.client = Client(model_name=model_name, openai_api_base=openai_api_base)

    def extract_keywords(self, query: str):
        prompt = f"""
        You are an intelligent keyword extraction assistant.
        Your goal is to extract the most important and relevant keywords or short phrases
        from the following user query, and organize them in a logical order that makes
        the resulting string ideal for performing a Google web search.

        Combine related words when appropriate (for example: "soybean pests", "wheat diseases", "Maryland agriculture"),
        and include key topics, entities (like crops, pests, locations, and years), and domain-specific terms.

        If a keyword or phrase contains multiple words that represent a fixed concept or named entity
        (for example: "soybean pests", "climate change", "drought impact"),
        enclose it in double quotes (" ") to make it search-optimized for SerpAPI and Google.
        Do NOT quote single words.

        The output must be a JSON list of strings, ordered from general to specific,
        so it can be joined and passed directly as a search query to the SerpAPI.

        Example:
        Input: "impact of drought on corn and soybean pests in Maryland 2022"
        Output: ["\\"drought impact\\"", "corn", "\\"soybean pests\\"", "Maryland", "2022"]

        Query: {query}
        """

        try:
            # ---- Model Call ----
            text = self.client.chat(prompt=prompt, images=[])
            print("🧠 Raw model response:\n", text, "\n")

            # ---- Clean Formatting ----
            text = re.sub(r"```(?:json)?", "", text).replace("```", "").strip()
            text = text.replace('\\"', '"')
            text = re.sub(r'""', '"', text)

            # ---- Parse JSON ----
            try:
                keywords = json.loads(text)
            except json.JSONDecodeError:
                match = re.search(r"\[.*\]", text, re.DOTALL)
                if match:
                    cleaned = match.group(0)
                    cleaned = cleaned.replace('\\"', '"')
                    keywords = json.loads(cleaned)
                else:
                    keywords = [w.strip().strip('"') for w in re.split(r"[,;\n]", text) if w.strip()]

            keywords = [kw.strip() for kw in keywords if isinstance(kw, str) and kw.strip()]
            print("✅ Extracted keywords:", keywords)
            return keywords

        except Exception as e:
            print(f"[Error] Keyword extraction failed: {e}")
            return []
