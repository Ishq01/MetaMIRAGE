# =============================
# WebAgent with vLLM + HuggingFaceEmbeddings
# =============================


# Modification suggested by ChatGPT:Y
#1. [DONE] You’re always adding docs, but not reloading or deduplicating. Over time, the store will bloat with duplicate pages.
# ✅ Fix: either clear/recreate vectorstore per run, or use doc IDs for deduplication.
#2. [DONE] Synchronous loading of web pages
# WebBaseLoader(url).load() will block and be slow on multiple URLs.
# ✅ Fix: use async loaders (LangChain has AsyncWebBaseLoader), or parallelize with asyncio.
#3. [TO BE DONE] Memory: add ConversationBufferMemory to the agent so it can handle follow-up queries without re-parsing everything.
#4. [TO BE DONE] Use RetrievalQA inside your agent: wrap your vectorstore.as_retriever() into a RetrievalQA chain, then expose it as a tool. That way the LLM doesn’t have to interpret raw docs.
#5. [TO BE DONE] Vectorstore persistence



from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import WebBaseLoader
from langchain.tools import BaseTool
from langchain.agents import initialize_agent, AgentType
from serpapi import GoogleSearch
import datetime
import argparse
import asyncio
import aiohttp
import hashlib
from typing import List, Dict, Any

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

# -----------------------------
# 2. Vector Store + Embeddings
# -----------------------------
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma(persist_directory="./vectorstore", embedding_function=embedding_model)

# -----------------------------
# 3. Web Search Tool
# -----------------------------
class WebSearchTool(BaseTool):
    name = "web_search"
    description = "Use this to search the web for the given keywords and meta-data when confidence is low"

    def _run(self, query: str = None):
        # Check if we have confidence evaluation results
        if hasattr(self, 'confidence_evaluation_tool') and hasattr(self.confidence_evaluation_tool, 'confidence'):
            if self.confidence_evaluation_tool.confidence >= self.confidence_value:
                return f"Web search skipped - sufficient data available (confidence: {self.confidence_evaluation_tool.confidence:.2f})"
        
        # Use provided query or fall back to keywords
        search_query = query if query else " ".join(self.keywords)
        
        params = {
            "engine": "google",
            "q": search_query,
            "location": f"{self.meta_data['county']}, {self.meta_data['state']}, United States",
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

            # Use async loading for better performance (unless sync is requested)
            use_sync = getattr(self, 'use_sync', False)
            if use_sync:
                documents, resultList = fetch_and_parse(self.meta_data, resultList)
            else:
                try:
                    documents, resultList = asyncio.run(fetch_and_parse_async(self.meta_data, resultList))
                except Exception as e:
                    print(f"Async loading failed, falling back to sync: {e}")
                    documents, resultList = fetch_and_parse(self.meta_data, resultList)
            
            add_to_vectorstore(documents)
            return resultList
        except Exception as exc:
            print(f"WebSearchTool error: {exc}")
            return []


# -----------------------------
# 4. Document ID Generation for Deduplication
# -----------------------------
def generate_doc_id(url: str, content: str) -> str:
    """Generate a unique document ID based on URL and content hash for deduplication"""
    content_hash = hashlib.md5(content.encode()).hexdigest()[:8]
    url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
    return f"{url_hash}_{content_hash}"

# -----------------------------
# 5. Async Web Page Loading
# -----------------------------
async def load_web_page_async(session: aiohttp.ClientSession, url: str, meta_data: Dict[str, Any]) -> Dict[str, Any]:
    """Asynchronously load a single web page"""
    try:
        async with session.get(url, timeout=10) as response:
            if response.status == 200:
                content = await response.text()
                return {
                    'url': url,
                    'content': content,
                    'status': 'success'
                }
            else:
                return {
                    'url': url,
                    'content': '',
                    'status': f'error_{response.status}'
                }
    except Exception as e:
        return {
            'url': url,
            'content': '',
            'status': f'error_{str(e)}'
        }

async def fetch_and_parse_async(meta_data: Dict[str, Any], myDictList: List[Dict[str, Any]]) -> tuple:
    """Asynchronously fetch and parse multiple web pages"""
    docs = []
    resultList = []
    
    # Create aiohttp session
    async with aiohttp.ClientSession() as session:
        # Create tasks for all URLs
        tasks = [load_web_page_async(session, myDict["url"], meta_data) for myDict in myDictList]
        
        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Process results
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            print(f"Error loading {myDictList[i]['url']}: {result}")
            continue
            
        if result['status'] == 'success':
            # Create document
            from langchain.schema import Document
            doc = Document(
                page_content=result['content'],
                metadata={
                    'url': result['url'],
                    'county': meta_data["county"],
                    'state': meta_data["state"],
                    'month': meta_data["timestamp"].month,
                    'doc_id': generate_doc_id(result['url'], result['content'])
                }
            )
            docs.append(doc)
            
            # Update result dict
            myDictList[i]['content'] = result['content']
            myDictList[i]['doc_id'] = doc.metadata['doc_id']
            resultList.append(myDictList[i])
        else:
            print(f"Failed to load {result['url']}: {result['status']}")
    
    return docs, resultList

# -----------------------------
# 6. Synchronous Fallback (for compatibility)
# -----------------------------
def fetch_and_parse(meta_data, myDictList):
    """Synchronous version for backward compatibility"""
    docs = []
    resultList = []
    for index, myDict in enumerate(myDictList):
        url = myDict["url"]
        loader = WebBaseLoader(url)
        doc = loader.load()  # returns list of Documents
        doc[0].metadata["county"] = meta_data["county"]
        doc[0].metadata["state"] = meta_data["state"]
        doc[0].metadata["month"] = meta_data["timestamp"].month
        doc[0].metadata["doc_id"] = generate_doc_id(url, doc[0].page_content)
        docs.extend(doc)
        myDict['content'] = doc[0].page_content
        myDict['doc_id'] = doc[0].metadata["doc_id"]
        resultList.append(myDict)
    return (docs, resultList)

# -----------------------------
# 7. Add Embeddings & Store with Deduplication
# -----------------------------
def add_to_vectorstore(documents):
    """Add documents to vectorstore with deduplication"""
    if not documents:
        return
    
    # Get existing document IDs to avoid duplicates
    existing_ids = set()
    try:
        # Query the vectorstore to get existing doc_ids
        existing_docs = vectorstore.similarity_search("", k=10000)  # Get all docs
        existing_ids = {doc.metadata.get('doc_id') for doc in existing_docs if doc.metadata.get('doc_id')}
    except Exception as e:
        print(f"Warning: Could not check existing documents: {e}")
    
    # Filter out documents that already exist
    new_documents = []
    for doc in documents:
        doc_id = doc.metadata.get('doc_id')
        if doc_id and doc_id not in existing_ids:
            new_documents.append(doc)
            existing_ids.add(doc_id)  # Add to set to avoid duplicates within this batch
        elif not doc_id:
            # If no doc_id, generate one and add
            doc.metadata['doc_id'] = generate_doc_id(
                doc.metadata.get('url', ''), 
                doc.page_content
            )
            new_documents.append(doc)
    
    if new_documents:
        print(f"Adding {len(new_documents)} new documents to vectorstore (filtered from {len(documents)} total)")
        vectorstore.add_documents(new_documents)
        # Invalidate caches when new documents are added
        invalidate_tool_caches()
    else:
        print("No new documents to add - all documents already exist in vectorstore")

def invalidate_tool_caches():
    """Invalidate cached results in tools when vectorstore is updated"""
    # This function can be called when the vectorstore is updated
    # to ensure tools don't use stale cached results
    pass

def clear_vectorstore():
    """Clear the vectorstore (use with caution)"""
    try:
        # Delete the entire collection
        vectorstore.delete_collection()
        print("Vectorstore cleared successfully")
    except Exception as e:
        print(f"Error clearing vectorstore: {e}")

def get_vectorstore_stats():
    """Get statistics about the vectorstore"""
    try:
        all_docs = vectorstore.similarity_search("", k=10000)
        return {
            'total_documents': len(all_docs),
            'unique_urls': len(set(doc.metadata.get('url', '') for doc in all_docs)),
            'counties': len(set(doc.metadata.get('county', '') for doc in all_docs)),
            'states': len(set(doc.metadata.get('state', '') for doc in all_docs))
        }
    except Exception as e:
        return {'error': str(e)}

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
    
    # Calculate confidence score based on alignment with meta_data and query
    confidence = calculate_confidence(filtered_results, meta_data, query)
    
    return filtered_results, confidence

def calculate_confidence(filtered_results, meta_data, query, config=None):
    """Calculate confidence score based on how well results align with meta_data and query
    
    Args:
        filtered_results: List of filtered documents
        meta_data: Metadata dict with state, county, timestamp
        query: Search query string
        config: Optional dict with confidence calculation parameters
    """
    if not filtered_results:
        return 0.0
    
    # Default configuration - can be overridden
    default_config = {
        'max_docs_for_full_confidence': 5,  # Number of docs needed for max base confidence
        'base_confidence_weight': 0.3,      # Weight for base confidence in final score
        'metadata_max_score': 0.7,          # Maximum score for metadata alignment
        'content_max_score': 0.3,           # Maximum score for content relevance
        'state_match_bonus': 0.2,           # Bonus for state match
        'county_match_bonus': 0.3,          # Bonus for county match (higher - more specific)
        'month_match_bonus': 0.2,           # Bonus for month match
        'web_search_threshold': 0.30        # Threshold below which web search is triggered
    }
    
    # Merge with provided config
    if config:
        default_config.update(config)
    cfg = default_config
    
    # Base confidence from number of results
    base_confidence = min(len(filtered_results) / cfg['max_docs_for_full_confidence'], 1.0)
    
    # Boost confidence for exact metadata matches
    metadata_score = 0.0
    for doc in filtered_results:
        if doc.metadata.get("state") == meta_data["state"]:
            metadata_score += cfg['state_match_bonus']
        if doc.metadata.get("county") == meta_data["county"]:
            metadata_score += cfg['county_match_bonus']
        if doc.metadata.get("month") == meta_data["timestamp"].month:
            metadata_score += cfg['month_match_bonus']
    
    # Normalize metadata score
    metadata_score = min(metadata_score / len(filtered_results), cfg['metadata_max_score'])
    
    # Query relevance score (simplified - you could use more sophisticated similarity)
    query_terms = query.lower().split()
    content_relevance = 0.0
    for doc in filtered_results:
        content = doc.page_content.lower()
        matches = sum(1 for term in query_terms if term in content)
        content_relevance += matches / len(query_terms)
    
    content_relevance = min(content_relevance / len(filtered_results), cfg['content_max_score'])
    
    # Calculate total confidence
    total_confidence = (base_confidence * cfg['base_confidence_weight'] + 
                       metadata_score + 
                       content_relevance)
    
    return min(total_confidence, 1.0)

# -----------------------------
# 7. Generate Answer Tool
# -----------------------------
class GenerateAnswerTool(BaseTool):
    name = "generate_answer"
    description = "Use this to generate a structured supplement information, using retrieved documents, which can be added to the user's query to help the subject model for better reasoning"

    def _run(self, query: str = None, docs=None):
        prompt_template = """
        You are an agriculture assistant. 
        Given the following retrieved documents with metadata and provided keywords & meta-data, provide a structured summary of the information which can be added to the user's query and a additional set of metadata
        which can be provided to the subject LLM model for further processing.

        Provided Keywords & Meta-Data: {keywords} {meta_data}
        Retrieved Docs: {docs}

        You should output JSON with two keys: "summary" and "metadata". The example is shown below:
        {{ "summary": ..., "metadata": {{ "key1": "value1", "key2": "value2" }} }}
        """

        prompt = PromptTemplate(template=prompt_template, input_variables=["keywords", "meta_data", "docs"])
        
        # Create context from docs
        context = "\n".join([f"{d.page_content} (Source: {d.metadata.get('url')})" for d in docs]) if docs else ""
        
        # Use the agent's LLM
        qa_chain = LLMChain(llm=self.llm, prompt=prompt)
        
        # Generate and return answer
        return qa_chain.run(keywords=self.keywords, meta_data=self.meta_data, docs=context)


# -----------------------------
# 8. Confidence Evaluation Tool
# -----------------------------
class ConfidenceEvaluationTool(BaseTool):
    name = "confidence_evaluation"
    description = "Use this to check if existing data is sufficient or if web search is needed"

    def _run(self, query: str = None):
        # Use configurable threshold
        config = getattr(self, 'confidence_config', {})
        threshold = config.get('web_search_threshold', 0.30)
        
        # Determine the search query
        search_query = query if query else " ".join(self.keywords)
        
        # Check if we have cached results for this exact query
        if (hasattr(self, 'cached_query') and 
            hasattr(self, 'retrieved_docs') and 
            hasattr(self, 'confidence') and 
            self.cached_query == search_query):
            # Use cached results for same query
            filtered_results = self.retrieved_docs
            confidence = self.confidence
        else:
            # Perform fresh retrieval for new/different query
            filtered_results, confidence = retrieve_relevant(self.meta_data, search_query)
            # Cache results with query identifier
            self.retrieved_docs = filtered_results
            self.confidence = confidence
            self.cached_query = search_query
        
        result = {
            "confidence": confidence,
            "needs_web_search": confidence < threshold,
            "available_docs": len(filtered_results),
            "threshold": threshold,
            "message": f"Confidence: {confidence:.2f} (threshold: {threshold:.2f}). {'Web search needed' if confidence < threshold else 'Sufficient data available'}"
        }
        
        return result

# -----------------------------
# 9. Retrieval Tool
# -----------------------------
class RetrievalTool(BaseTool):
    name = "retrieval"
    description = "Use this to retrieve the relevant documents from the vector store"

    def _run(self, query: str = None):
        # Determine the search query
        search_query = query if query else " ".join(self.keywords)
        
        # Check if we can use cached results from confidence evaluation
        if (hasattr(self, 'confidence_evaluation_tool') and 
            hasattr(self.confidence_evaluation_tool, 'cached_query') and
            hasattr(self.confidence_evaluation_tool, 'retrieved_docs') and
            self.confidence_evaluation_tool.cached_query == search_query):
            # Use cached results for exact query match
            return self.confidence_evaluation_tool.retrieved_docs
        
        # Fallback to fresh retrieval if no cached results or different query
        filtered_results, confidence = retrieve_relevant(self.meta_data, search_query)
        return filtered_results

# -----------------------------
# 9. Agent 
# -----------------------------

def create_agent(keywords, meta_data, confidence_config=None, use_sync=False):
    # Initialize tools with metadata and keywords
    confidence_evaluation_tool = ConfidenceEvaluationTool()
    confidence_evaluation_tool.meta_data = meta_data
    confidence_evaluation_tool.keywords = keywords
    confidence_evaluation_tool.confidence_config = confidence_config or {}

    retrieval_tool = RetrievalTool()
    retrieval_tool.meta_data = meta_data
    retrieval_tool.keywords = keywords
    retrieval_tool.confidence_evaluation_tool = confidence_evaluation_tool  # Link tools for result sharing

    web_search_tool = WebSearchTool()
    web_search_tool.meta_data = meta_data
    web_search_tool.keywords = keywords
    web_search_tool.confidence_evaluation_tool = confidence_evaluation_tool  # Link tools
    web_search_tool.use_sync = use_sync  # Set sync flag
    web_search_tool.confidence_value = confidence_config['web_search_threshold'] if confidence_config else 0.30

    # Initialize LLM for the agent
    llm = ChatOpenAI(
        model="meta-llama/Meta-Llama-3-8B-Instruct",  # or the model you served
        openai_api_base="http://<server-ip>:8000/v1",  # replace <server-ip>
        openai_api_key="EMPTY",  # vLLM ignores this, required by LangChain
        temperature=0,
    )

    generate_answer_tool = GenerateAnswerTool()
    generate_answer_tool.meta_data = meta_data
    generate_answer_tool.keywords = keywords
    generate_answer_tool.llm = llm  # Set the LLM for the tool

    # Create agent with tools - confidence evaluation should be first
    tools = [confidence_evaluation_tool, retrieval_tool, web_search_tool, generate_answer_tool]
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
        verbose=True
    )
    return agent
#
#
    


# -----------------------------
# X. Full Workflow
# -----------------------------

def main():

    # Set up argument parser
    parser = argparse.ArgumentParser(description='Web Agent for searching and analyzing content')
    parser.add_argument('input', type=str, help='Input string to parse')
    parser.add_argument('--clear-vectorstore', action='store_true', help='Clear the vectorstore before running')
    parser.add_argument('--stats', action='store_true', help='Show vectorstore statistics and exit')
    parser.add_argument('--use-sync', action='store_true', help='Use synchronous loading instead of async')

    # Parse arguments
    args = parser.parse_args()
    
    # Handle stats command
    if args.stats:
        stats = get_vectorstore_stats()
        print("Vectorstore Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        return
    
    # Handle clear vectorstore
    if args.clear_vectorstore:
        clear_vectorstore()
        print("Vectorstore cleared. Continuing with search...")

    # Parse user input string
    try:
        parsed = parse_user_data(args.input)
        keywords = parsed["keywords"]
        meta_data = {
            'county': parsed["county"],
            'state': parsed["state"], 
            'timestamp': parsed["timestamp"]
        }
    except ValueError as e:
        print(f"Error parsing input: {e}")
        return
    # Optional: Custom confidence configuration
    # You can adjust these values based on your domain and requirements
    custom_config = {
        'web_search_threshold': 0.25,        # Lower threshold = more web searches
        'max_docs_for_full_confidence': 5,   # Fewer docs needed for full confidence
        'county_match_bonus': 0.3,           # Higher bonus for county matches
        'metadata_max_score': 0.7,           # Higher max score for metadata
    }
    
    # Initialize and run the agent
    agent = create_agent(keywords, meta_data, custom_config, use_sync=args.use_sync)
    
    # Provide clear instructions to the agent about the workflow
    instructions = f"""
    Please help me find information about {' '.join(keywords)} in {meta_data['county']}, {meta_data['state']} around {meta_data['timestamp']}.
    
    Workflow:
    1. First, use confidence_evaluation to check if we have sufficient data
    2. If confidence < 0.30, use web_search to find more information
    3. Use retrieval to get relevant documents from our database
    4. Finally, use generate_answer to create a structured response
    
    The confidence threshold is {custom_config['web_search_threshold']} - only search the web if confidence is below this threshold.
    """
    
    answer = agent.run(instructions)
    print(answer)

if __name__ == "__main__":
    main()