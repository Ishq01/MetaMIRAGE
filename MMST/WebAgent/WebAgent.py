# =============================
# WebAgent with vLLM + HuggingFaceEmbeddings
# =============================

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import WebBaseLoader
from langchain.tools import BaseTool
from serpapi import GoogleSearch
import datetime

# -----------------------------
# 1. User Input
# -----------------------------
user_query = "crops, common pest, maryland, montgomery, 2022-09-01, 07:17:00"

def parse_user_data(text: str):
    """Parse a free-form query into keywords, state, county, and timestamp.

    Expected format (comma-separated):
      "<keyword 1>, <keyword 2>, <state>, <county>, <YYYY-MM-DD>, <HH:MM:SS>"

    Returns a dict with keys: keywords (list[str]), state (str), county (str), timestamp (datetime).
    """
    if not isinstance(text, str):
        raise ValueError("Input must be a string")

    parts = [p.strip() for p in text.strip().split(',') if p is not None and p.strip() != ""]
    if len(parts) < 5:
        raise ValueError("Input must include keywords, state, county, date, and time")

    date_token = parts[-2].strip()
    time_token = parts[-1].strip()
    try:
        timestamp = datetime.datetime.strptime(f"{date_token} {time_token}", "%Y-%m-%d %H:%M:%S")
    except ValueError as exc:
        raise ValueError("Timestamp must be in 'YYYY-MM-DD HH:MM:SS' format") from exc

    county = parts[-3].strip()
    state = parts[-4].strip()
    keyword_tokens = [k.strip() for k in parts[:-4] if k.strip() != ""]
    keywords = keyword_tokens

    return {
        "keywords": keywords,
        "state": state.lower(),
        "county": county.lower(),
        "timestamp": timestamp,
    }

# Parse metadata from user query
parsed = parse_user_data(user_query)
print(parsed)
meta_data = {
    "state": parsed["state"],
    "county": parsed["county"],
    "timestamp": parsed["timestamp"],
}

keywords = parsed["keywords"]

# -----------------------------
# 2. Vector Store + Embeddings
# -----------------------------
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma(persist_directory="./vectorstore", embedding_function=embedding_model)

# -----------------------------
# 3. Web Search Tool (placeholder)
# -----------------------------
class WebSearchTool(BaseTool):
    name = "web_search"
    description = "Use this to search the web for the given keywords and meta-data"

    def _run(self):
        params = {
            "engine": "google",
            "q": getattr(self, "key_terms", ""),
            "location": getattr(self, "location", ""),
            "api_key": "d19541a8bf7c2e56f4ef32e6ea227808c29166d868dcbe0827b88d09effc805b"
        }

        try:
            search = GoogleSearch(params)
            results = search.get_dict()
            organic_results = results.get("organic_results", [])

            resultList = []
            for value in organic_results:
                myDict = {}
                myDict['title'] = value.get('title', '')
                myDict['url'] = value.get('link') or value.get('url', '')
                myDict['snippet'] = value.get('snippet', '')
                resultList.append(myDict)

            return resultList
        except Exception as exc:
            print(f"WebSearchTool error: {exc}")
            return []

param_key_terms = " ".join(keywords)
param_location = ", ".join([parsed["county"], parsed["state"], "United States"])
web_search_tool = WebSearchTool()
web_search_tool.location = param_location
web_search_tool.key_terms = param_key_terms

# -----------------------------
# 4. Fetch & Parse Web Pages
# -----------------------------
def fetch_and_parse(meta_data, myDictList):
    docs = []
    resultList = []
    for index, myDict in enumerate(myDictList):
        url = myDict["url"]
        loader = WebBaseLoader(url)
        doc = loader.load()  # returns list of Documents
        doc[0].metadata["county"] = meta_data["county"]
        doc[0].metadata["state"] = meta_data["state"]
        doc[0].metadata["month"] = meta_data["timestamp"].month
        docs.extend(doc)
        myDict['content'] = doc[0].page_content
        resultList.append(myDict)
    return (docs, resultList)

# -----------------------------
# 5. Add Embeddings & Store
# -----------------------------
# documents is a list of Document objects
def add_to_vectorstore(documents):
    vectorstore.add_documents(documents)

# -----------------------------
# 6. Retrieval + Metadata Filtering
# -----------------------------
def retrieve_relevant(meta_data, query, k=5):
    results = vectorstore.similarity_search(query, k=k)
    filtered_results = [
        doc for doc in results
        if doc.metadata.get("state") == meta_data["state"] and doc.metadata.get("county") == meta_data["county"]
        and doc.metadata.get("month") == meta_data["timestamp"].month
    ]
    return filtered_results

# -----------------------------
# 7. LLM Answer Generation (via vLLM) XXXXXXXXXXXXXXXXX
# -----------------------------
prompt_template = """
You are an agriculture assistant. 
Given the following retrieved documents with metadata and user query, provide a precise answer and cite sources.

User Query: {query}
Retrieved Docs: {docs}
"""

prompt = PromptTemplate(template=prompt_template, input_variables=["query", "docs"])

# Connect to vLLM server (OpenAI-compatible API)
llm = ChatOpenAI(
    model="meta-llama/Meta-Llama-3-8B-Instruct",  # or the model you served
    openai_api_base="http://<server-ip>:8000/v1",  # replace <server-ip>
    openai_api_key="EMPTY",  # vLLM ignores this, required by LangChain
    temperature=0,
)

qa_chain = LLMChain(llm=llm, prompt=prompt)

def generate_answer(query, docs):
    context = "\n".join([f"{d.page_content} (Source: {d.metadata.get('url')})" for d in docs])
    return qa_chain.run(query=query, docs=context)

# -----------------------------
# 8. Full Workflow
# -----------------------------
# 1. Web Search
search_results = web_search_tool.run()

# 2. Fetch & Parse pages
documents, search_results = fetch_and_parse(meta_data, search_results)

# 3. Add metadata for filtering
for index, doc in enumerate(documents):
    r = search_results[index]
    doc.metadata["url"] = r["url"]
    doc.metadata["state"] = meta_data["state"]
    doc.metadata["month"] = meta_data["timestamp"].month

# 4. Store in vector DB
add_to_vectorstore(documents)

# 5. Retrieve relevant docs
relevant_docs = retrieve_relevant(meta_data, " ".join(keywords))

# 6. Generate answer XXXXXXXXXXXXXXXXX
answer = generate_answer(user_query, relevant_docs)
print(answer)