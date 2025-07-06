# utils/web_search.py

import os
import requests
from dotenv import load_dotenv

load_dotenv()

SERPER_API_KEY = os.getenv("SERPER_API_KEY")
SERPER_API_URL = "https://google.serper.dev/search"


def search_web(query: str) -> str:
    """
    Performs a web search using Serper.dev API and returns top results.

    Args:
        query (str): User question/query.

    Returns:
        str: Concatenated search result snippets.
    """
    headers = {
        "X-API-KEY": SERPER_API_KEY,
        "Content-Type": "application/json",
    }

    payload = {
        "q": query
    }

    try:
        response = requests.post(SERPER_API_URL, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()

        results = data.get("organic", [])[:3]  # Top 3 results
        summaries = [f"- {item['title']}: {item.get('snippet', '')}" for item in results]

        return "\n".join(summaries) if summaries else "No relevant search results found."

    except Exception as e:
        return f"Search failed: {str(e)}"
