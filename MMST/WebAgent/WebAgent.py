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
import datetime

# -----------------------------
# 1. User Input
# -----------------------------
user_query = "common pest maryland montgomery 2022-09-01 07:17:00"

# Parse metadata (simple example)
meta_data = {
    "state": "maryland",
    "county": "montgomery",
    "timestamp": datetime.datetime.strptime("2022-09-01 07:17:00", "%Y-%m-%d %H:%M:%S")
}

keywords = ["common pest"]

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
    description = "Use this to search the web for info related to agriculture pests"

    def _run(self, query: str):
        # Implement actual API call (SerpAPI, Bing, Tavily, etc.)
        return [
            {
                "title": "Soybean aphid infestations expected to peak in Illinois this September",
                "snippet": "Soybean aphids are expected to increase in September...",
                "url": "https://example.com/article1"
            },
            {
                "title": "University of Illinois crop watch: late-season pests",
                "snippet": "Late-season pests include soybean aphids and spider mites...",
                "url": "https://example.com/article2"
            }
        ]

web_search_tool = WebSearchTool()

# -----------------------------
# 4. Fetch & Parse Web Pages
# -----------------------------
def fetch_and_parse(urls):
    docs = []
    for url in urls:
        loader = WebBaseLoader(url)
        doc = loader.load()  # returns list of Documents
        docs.extend(doc)
    return docs

# -----------------------------
# 5. Add Embeddings & Store
# -----------------------------
def add_to_vectorstore(documents):
    vectorstore.add_documents(documents)

# -----------------------------
# 6. Retrieval + Metadata Filtering
# -----------------------------
def retrieve_relevant(meta_data, query, k=5):
    results = vectorstore.similarity_search(query, k=k)
    filtered_results = [
        doc for doc in results
        if doc.metadata.get("state") == meta_data["state"]
        and doc.metadata.get("month") == meta_data["timestamp"].month
    ]
    return filtered_results

# -----------------------------
# 7. LLM Answer Generation (via vLLM)
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
search_results = web_search_tool.run(" ".join(keywords + [meta_data["state"], meta_data["county"]]))

# 2. Fetch & Parse pages
urls = [r["url"] for r in search_results]
documents = fetch_and_parse(urls)

# 3. Add metadata for filtering
for doc, r in zip(documents, search_results):
    doc.metadata["url"] = r["url"]
    doc.metadata["state"] = meta_data["state"]
    doc.metadata["month"] = meta_data["timestamp"].month

# 4. Store in vector DB
add_to_vectorstore(documents)

# 5. Retrieve relevant docs
relevant_docs = retrieve_relevant(meta_data, "soybean aphid")

# 6. Generate answer
answer = generate_answer("What are common pests in Montgomery, Maryland for this time?", relevant_docs)
print(answer)
