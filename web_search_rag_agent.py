from __future__ import annotations

from dotenv import load_dotenv
load_dotenv()

"""
web_search_rag_agent.py
=======================

This module provides a high‑level implementation of a web‑search‑based
agentic RAG system.  Unlike the SQL‑focused example in
`rag_orchestrator.py`, this agent uses a search engine to find
information on the open web.  It applies the agentic RAG principles
described in the Milvus DeepSearcher article: multiple queries are
generated from the original question, intermediate results are
evaluated, and the search is refined iteratively until enough
evidence is gathered【756007594819921†L52-L60】【756007594819921†L124-L135】.

Key components:

* **Query expansion**: A language model generates up to three focussed
  search queries from the user’s question.
* **Search API integration**: The code uses the Google Custom Search
  JSON API (also known as Programmable Search Engine) as an example.
  You must supply your own `API_KEY` and `SEARCH_ENGINE_ID`.  You can
  substitute other search APIs (e.g. SerpAPI) by modifying
  `search_google`.
* **Web scraping**: Each search result is fetched via HTTP and the
  visible text is extracted with BeautifulSoup.
* **Quality assessment and iterative refinement**: A simple heuristic
  evaluates whether enough relevant information has been retrieved.
  If not, additional queries are generated and executed.
* **Summarisation and citation**: The retrieved passages are
  summarised by a Vertex AI chat model.  The answer includes
  numbered citations pointing back to the original URLs.

IMPORTANT: This code is a template.  Running it will incur Google
Custom Search API costs.  Make sure you understand pricing and
configure rate limiting.  Also note that web scraping must respect
robots.txt and site terms of service.  You may need to implement
throttling and user‑agent headers to avoid overloading websites.
"""



import json
import os
import time
from typing import List, Dict, Any, Tuple

import requests
from bs4 import BeautifulSoup

from langchain_google_vertexai import ChatVertexAI
from langchain_google_vertexai import VertexAIEmbeddings

# Vertex AI configuration
VERTEX_PROJECT = os.getenv("VERTEX_PROJECT") #, "your‑project‑id")
VERTEX_REGION = os.getenv("VERTEX_REGION")

# Search API credentials (set these as environment variables)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") #, "YOUR_GOOGLE_API_KEY")
SEARCH_ENGINE_ID = os.getenv("SEARCH_ENGINE_ID") #, "YOUR_SEARCH_ENGINE_ID")

# ---------------------------------------------------------------------------
# Prompts

SEARCH_QUERY_PROMPT = """
You are a helpful assistant that translates a user's question into up
to three focussed web search queries.  For example, given a broad
question you may break it into queries that target specific aspects.
Return the list of queries as a JSON array of strings without any
additional text.

Question: {question}
"""

SEARCH_QUERY_REFINEMENT_PROMPT = """
You are a helpful assistant that generates new search queries to find additional information for a user's question.

The user's original question is: {question}

Previous queries that have already been tried:
{previous_queries}

Generate up to three NEW search queries that:
1. Are different from the previous queries listed above
2. Target different aspects or angles of the question
3. Use different keywords or search strategies
4. Focus on finding information that might have been missed by the previous searches

Return the list of new queries as a JSON array of strings without any additional text.
"""

SUMMARY_PROMPT = """
You are a research assistant.  The user asked the following question:

{question}

You have gathered information from multiple web pages.  Each entry in
`sources` includes a `url` and `content` (extracted text).  Carefully
read these sources and write a concise, accurate answer to the
question.  Then provide a markdown list of citations with the format
`[1]`, `[2]`, ... corresponding to the URLs in order.  Include the
citations inline in the answer where appropriate.

Sources:
{sources_json}

Answer:
"""

ANNOTATION_PROMPT = """
You are an expert annotator evaluating the quality of web search results for a given user query.

The user submitted the following query:

{query}

You have been provided with a list of search results. Each item includes a `url` and its extracted `content`. Your task is to:

1. **Relevance Judgment**:
   - For each result, determine whether it is *highly relevant*, *partially relevant*, or *not relevant* to the user’s query.
   - Explain briefly *why* it is or isn’t relevant.

2. **Coverage Assessment**:
   - Assess whether the results *collectively* cover the user’s intent.
   - If there are gaps (e.g., key aspects of the query not covered), specify what is missing.

Format your answer as follows:

### Per-Result Relevance

1. **URL**: <url>
   - **Relevance**: [Highly Relevant / Partially Relevant / Not Relevant]
   - **Reason**: <short justification>

...

### Overall Coverage

- **Coverage Judgment**: [Complete / Partial / Inadequate]
- **Reason**: <explanation of whether the user’s intent is fully addressed or what’s missing>

Sources:
{sources_json}
"""


# ---------------------------------------------------------------------------
# Helper functions

def get_chat_model(max_tokens: int = 1024, temperature: float = 0.2, model_name: str = "gemini-2.5-flash-lite") -> ChatVertexAI:
    """Instantiate a Vertex AI chat model.

    :param max_tokens: Max number of tokens.
    :param temperature: Sampling temperature.
    :param model_name: Model name.
    :return: ChatVertexAI instance.
    """
    import vertexai
    vertexai.init(project=VERTEX_PROJECT, location=VERTEX_REGION)
    
    model = ChatVertexAI(
        model_name=model_name,
        max_output_tokens=max_tokens,
        temperature=temperature
    )
    
    return model


def generate_search_queries(question: str, previous_queries: List[str] = None) -> List[str]:
    """Use a language model to generate up to three search queries.

    :param question: Natural language question.
    :param previous_queries: List of previous queries to avoid repetition.
    :return: List of search query strings.
    """
    model = get_chat_model(max_tokens=256)  # Increased max_tokens
    
    if previous_queries and len(previous_queries) > 0:
        # Use refinement prompt to avoid repeating previous queries
        previous_queries_text = "\n".join([f"- {q}" for q in previous_queries])
        prompt = SEARCH_QUERY_REFINEMENT_PROMPT.format(
            question=question, 
            previous_queries=previous_queries_text
        )
        print(f"Using refinement prompt with {len(previous_queries)} previous queries")
    else:
        # Use original prompt for first-time query generation
        prompt = SEARCH_QUERY_PROMPT.format(question=question)
        print("Using original prompt for first-time generation")
    
    result = model.invoke(prompt)
    print(f"Raw response: {result.content[:200]}...")

    try:
        # Try to extract JSON from the response
        content = result.content.strip()
        if content.startswith("```json"):
            content = content[7:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()
        
        queries = json.loads(content)
        if isinstance(queries, list) and queries:
            queries = [q.strip() for q in queries if isinstance(q, str) and q.strip()]
            print(f"Successfully parsed queries: {queries}")
            return queries
    except Exception as e:
        print(f"Error parsing JSON response: {e}")
        print(f"Response content: {result.content}")
    
    # Fallback: return the original question
    print(f"Falling back to original question: {question}")
    return [question]


def search_google(query: str, num_results: int = 5) -> List[Dict[str, Any]]:
    """Call the Google Custom Search JSON API and return search results.

    :param query: Search query string.
    :param num_results: Number of results to return.
    :return: List of dicts with keys 'title', 'snippet', 'link'.
    """
    if not GOOGLE_API_KEY or not SEARCH_ENGINE_ID:
        raise RuntimeError("Google search API key and engine ID must be set in environment variables.")
    endpoint = "https://www.googleapis.com/customsearch/v1"
    params = {
        "key": GOOGLE_API_KEY,
        "cx": SEARCH_ENGINE_ID,
        "q": query,
        "num": num_results,
        "safe": "active",
        "hl": "en",
    }
    response = requests.get(endpoint, params=params, timeout=10)
    response.raise_for_status()
    data = response.json()
    items = data.get("items", [])
    results = []
    for item in items:
        results.append({
            "title": item.get("title"),
            "snippet": item.get("snippet"),
            "link": item.get("link"),
        })
    
    print(f"DEBUG: search_google - query='{query}', num_results={num_results}, found_results={len(results)}")
    return results


def fetch_page_text(url: str, max_chars: int = 8000) -> str:
    """Fetch a web page and extract visible text.

    :param url: URL to fetch.
    :param max_chars: Maximum number of characters to return (to limit cost).
    :return: Extracted text (truncated).
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; AgenticRAG/1.0; +https://example.com/bot)"
    }
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        # Remove script/style elements
        for script in soup(["script", "style"]):
            script.decompose()
        text = " ".join(soup.stripped_strings)
        result = text[:max_chars]
        return result
    except Exception as e:
        return ""



def evaluate_relevance(question:str, sources: List[Tuple[str, str]]) -> bool:
    """Use LLM to determine if retrieved content sufficiently covers the user query.
    :param sources: List of tuples (url, content).
    :param question: The original query.
    :return: True if LLM says coverage is 'Complete', else False.
    """
    if not sources:
        print("No sources to evaluate")
        return False
        
    try:
        model = get_chat_model(max_tokens=1024)
        # Create a list of dicts for the prompt
        sources_data = [
            {"url": url, "content": content[:2000]} for url, content in sources
        ]
        
        prompt = ANNOTATION_PROMPT.format(query=question, sources_json=json.dumps(sources_data, indent=2))   
        result = model.invoke(prompt)

        # Simple heuristic to look for 'Coverage Judgment: Complete'
        result_lower = result.content.lower()
        print(f"Evaluation result: {result_lower[:200]}...")  # Print first 200 chars
        
        if "coverage judgment" in result_lower and "complete" in result_lower:
            print(f"Coverage is COMPLETE for question: '{question}'")
            return True
        else:
            print(f"Coverage is INCOMPLETE for question: '{question}'")
            return False
    except Exception as e:
        print(f"Error in evaluate_relevance: {e}")
        # If evaluation fails, assume incomplete coverage
        return False
    

def summarise_sources(question: str, sources: List[Tuple[str, str]]) -> str:
    """Summarise multiple sources using a language model.

    :param question: User question.
    :param sources: List of tuples (url, content).
    :return: Answer with citations.
    """
    
    model = get_chat_model(max_tokens=1024)
    # Create a list of dicts for the prompt
    sources_data = [
        {"url": url, "content": content[:2000]} for url, content in sources
    ]
    
    prompt = SUMMARY_PROMPT.format(question=question, sources_json=json.dumps(sources_data, indent=2))
    result = model.invoke(prompt)
    answer = result.content.strip()
    return answer


def answer_via_web(question: str, max_iterations: int = 3) -> str:
    """Main entry point for answering a question using web search.

    It implements an agentic loop: generate search queries, fetch
    results, extract text, evaluate whether enough information has been
    collected, and, if not, refine the search using additional
    generated queries.

    :param question: User's question.
    :param max_iterations: Maximum number of refinement rounds.
    :return: Final answer with citations.
    """
    # Keep track of visited URLs to avoid duplicates
    visited: set[str] = set()
    collected: List[Tuple[str, str]] = []
    all_queries: List[str] = []  # Track all queries used
    
    # Initial set of queries
    queries = generate_search_queries(question)
    all_queries.extend(queries)
    print(f"Initial queries: {all_queries}")
    
    for iteration in range(max_iterations):
        print(f"\n=== Starting iteration {iteration + 1} ===")
        
        # Process all queries in this iteration
        for query in queries:
            print(f"Processing query: {query}")
            try:
                results = search_google(query, num_results=5)
            except Exception as e:
                print(f"Search error: {e}")
                continue
                
            for item in results:
                url = item["link"]
                if url in visited:
                    continue
                visited.add(url)
                text = fetch_page_text(url)
                if text:
                    collected.append((url, text))
                    print(f"Added source: {url}")
        
        # After processing all queries in this iteration, check if we have enough information
        print(f"Collected {len(collected)} sources so far")
        
        if len(collected) > 0:
            if evaluate_relevance(question, collected):
                print("Sufficient information found!")
                break
        
        # If we don't have enough information and haven't reached max iterations, generate new queries
        if iteration < max_iterations - 1:
            print("Generating new queries for next iteration...")
            new_queries = generate_search_queries(question, all_queries)
            all_queries.extend(new_queries)
            queries = new_queries
            print(f"New queries for next iteration: {new_queries}")
        else:
            print("Reached maximum iterations")
        
        time.sleep(1)  # polite delay to avoid hitting API limits
    
    # Final evaluation before summarizing
    if len(collected) == 0:
        return (
            "I'm sorry, I wasn't able to find any relevant information to answer your question. "
            "Please try rephrasing it or checking back later as new content may become available."
        )
    
    if not evaluate_relevance(question, collected):
        return (
            "I'm sorry, I wasn't able to find sufficient relevant information to fully answer your question. "
            "Please try rephrasing it or checking back later as new content may become available."
        )
    else:
        answer = summarise_sources(question, collected)    
        return answer


if __name__ == "__main__":
    # Example usage.  Set GOOGLE_API_KEY and SEARCH_ENGINE_ID in your
    # environment before running.  The following call will perform
    # searches, fetch pages and summarise them.
    question = "how old is milad shokouhi's mother?"
    #question = "how old is donald trump's mother?"
    answer = answer_via_web(question)
    print("Final answer:\n")
    print(answer)