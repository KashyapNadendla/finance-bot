import os
import requests
from dotenv import load_dotenv

load_dotenv()

def run_web_search(query):
    """
    Run a web search using the Google Custom Search JSON API.
    """
    api_key = os.getenv("WEBSEARCH_API_KEY", "") 
    cx = os.getenv("CUSTOM_SEARCH_ENGINE_ID", "")  

    if not api_key or not cx:
        return {"error": "API key or Custom Search Engine ID not found in .env file."}

    base_url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "q": query,
        "key": api_key,
        "cx": cx
    }

    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
        results = response.json()

        # Extract relevant search results
        search_items = results.get("items", [])
        extracted_results = [{"title": item.get("title"), "link": item.get("link"), "snippet": item.get("snippet")} for item in search_items]

        return extracted_results

    except requests.exceptions.RequestException as e:
        return {"error": str(e)}
    except (KeyError, ValueError) as e:
        return {"error": f"Error parsing response: {e}"}