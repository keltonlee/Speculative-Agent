"""
Tools Registry - Centralized tool loading and management

This module provides a unified interface for loading all available search and utility tools.
It handles missing packages and API keys gracefully, allowing tests to run with any combination
of available tools.
"""

import os
from typing import List, Dict, Any
from langchain_core.tools import tool

# ==================== Tool Availability Flags ====================

# Existing tools
EXA_AVAILABLE = False
TAVILY_AVAILABLE = False

# New search tools
ARXIV_AVAILABLE = False
ASKNEWS_AVAILABLE = False
BRAVE_AVAILABLE = False
DUCKDUCKGO_AVAILABLE = False
GOOGLE_SERPER_AVAILABLE = False
SEARXNG_AVAILABLE = False
YOU_AVAILABLE = False

# ==================== Import Tools with Error Handling ====================

# Existing tools
try:
    from langchain_exa import ExaSearchResults
    EXA_AVAILABLE = True
except ImportError:
    pass

try:
    from langchain_tavily import TavilySearch
    TAVILY_AVAILABLE = True
except ImportError:
    pass

# Arxiv (academic papers)
try:
    from langchain_community.retrievers import ArxivRetriever
    ARXIV_AVAILABLE = True
except ImportError:
    pass

# AskNews (news search)
try:
    from langchain_community.tools.asknews import AskNewsSearch
    ASKNEWS_AVAILABLE = True
except ImportError:
    pass

# Brave Search
try:
    from langchain_community.tools import BraveSearch
    BRAVE_AVAILABLE = True
except ImportError:
    pass

# DuckDuckGo (no API key needed!)
try:
    from langchain_community.tools import DuckDuckGoSearchRun
    DUCKDUCKGO_AVAILABLE = True
except ImportError:
    pass

# Google Serper
try:
    from langchain_community.utilities import GoogleSerperAPIWrapper
    GOOGLE_SERPER_AVAILABLE = True
except ImportError:
    pass

# SearXNG
try:
    from langchain_community.utilities import SearxSearchWrapper
    SEARXNG_AVAILABLE = True
except ImportError:
    pass

# You.com
try:
    from langchain_community.tools.you import YouSearchTool
    YOU_AVAILABLE = True
except ImportError:
    pass


# ==================== Tool Loading Functions ====================

def get_exa_tool():
    """Get Exa search tool if available."""
    if not EXA_AVAILABLE:
        return None
    if not os.environ.get("EXA_API_KEY"):
        return None

    try:
        return ExaSearchResults(
            exa_api_key=os.environ["EXA_API_KEY"],
            max_results=5,
        )
    except Exception as e:
        print(f"  ⚠️  Error loading Exa: {e}")
        return None


def get_tavily_tool():
    """Get Tavily search tool if available."""
    if not TAVILY_AVAILABLE:
        return None
    if not os.environ.get("TAVILY_API_KEY"):
        return None

    try:
        return TavilySearch(max_results=5)
    except Exception as e:
        print(f"  ⚠️  Error loading Tavily: {e}")
        return None


def get_arxiv_tool():
    """Get Arxiv academic paper search tool if available (no API key needed!)."""
    if not ARXIV_AVAILABLE:
        return None

    try:
        # Wrap ArxivRetriever as a tool
        retriever = ArxivRetriever(load_max_docs=3)

        @tool
        def arxiv_search(query: str) -> str:
            """Search for academic papers on arXiv.

            Args:
                query: Search query for academic papers
            """
            try:
                docs = retriever.get_relevant_documents(query)
                if not docs:
                    return "No papers found."

                results = []
                for i, doc in enumerate(docs[:3], 1):
                    title = doc.metadata.get('Title', 'Unknown')
                    authors = doc.metadata.get('Authors', 'Unknown')
                    summary = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                    results.append(f"{i}. {title}\n   Authors: {authors}\n   Summary: {summary}")

                return "\n\n".join(results)
            except Exception as e:
                return f"Error searching arXiv: {e}"

        return arxiv_search
    except Exception as e:
        print(f"  ⚠️  Error loading Arxiv: {e}")
        return None


def get_asknews_tool():
    """Get AskNews tool if available."""
    if not ASKNEWS_AVAILABLE:
        return None
    if not os.environ.get("ASKNEWS_CLIENT_ID") or not os.environ.get("ASKNEWS_CLIENT_SECRET"):
        return None

    try:
        return AskNewsSearch()
    except Exception as e:
        print(f"  ⚠️  Error loading AskNews: {e}")
        return None


def get_brave_tool():
    """Get Brave search tool if available."""
    if not BRAVE_AVAILABLE:
        return None
    if not os.environ.get("BRAVE_SEARCH_API_KEY"):
        return None

    try:
        return BraveSearch.from_api_key(
            api_key=os.environ["BRAVE_SEARCH_API_KEY"],
            search_kwargs={"count": 5}
        )
    except Exception as e:
        print(f"  ⚠️  Error loading Brave: {e}")
        return None


def get_duckduckgo_tool():
    """Get DuckDuckGo search tool if available (no API key needed!)."""
    if not DUCKDUCKGO_AVAILABLE:
        return None

    try:
        return DuckDuckGoSearchRun()
    except Exception as e:
        print(f"  ⚠️  Error loading DuckDuckGo: {e}")
        return None


def get_google_serper_tool():
    """Get Google Serper search tool if available."""
    if not GOOGLE_SERPER_AVAILABLE:
        return None
    if not os.environ.get("SERPER_API_KEY"):
        return None

    try:
        search = GoogleSerperAPIWrapper()
        @tool
        def google_serper_search(query: str) -> str:
            """Search Google using Serper API. Use for general web searches."""
            return search.run(query)    
        return google_serper_search
    except Exception as e:
        print(f"  ⚠️  Error loading Google Serper: {e}")
        return None


def get_searxng_tool():
    """Get SearXNG metasearch tool if available."""
    if not SEARXNG_AVAILABLE:
        return None

    # SearXNG requires a URL to a SearXNG instance
    searxng_url = os.environ.get("SEARXNG_URL", "https://searx.be")

    try:
        search = SearxSearchWrapper(searx_host=searxng_url)

        @tool
        def searxng_search(query: str) -> str:
            """Search using SearXNG metasearch engine.

            Args:
                query: Search query
            """
            try:
                return search.run(query)
            except Exception as e:
                return f"Error searching SearXNG: {e}"

        return searxng_search
    except Exception as e:
        print(f"  ⚠️  Error loading SearXNG: {e}")
        return None


def get_you_tool():
    """Get You.com search tool if available."""
    if not YOU_AVAILABLE:
        return None
    if not os.environ.get("YDC_API_KEY"):
        return None

    try:
        return YouSearchTool(api_key=os.environ["YDC_API_KEY"])
    except Exception as e:
        print(f"  ⚠️  Error loading You.com: {e}")
        return None


# ==================== Main Function ====================

def get_all_available_tools(verbose: bool = True) -> List[Any]:
    """
    Get all available tools based on installed packages and API keys.

    Args:
        verbose: If True, print information about tool availability

    Returns:
        List of available tools
    """
    tools = []

    if verbose:
        print("\n📦 Loading Tools...")
        print("="*60)

    # Track tool categories
    search_tools = []
    missing_tools = []

    # Try to load each search tool
    tool_loaders = [
        ("Exa Search", get_exa_tool, "EXA_API_KEY"),
        ("Tavily Search", get_tavily_tool, "TAVILY_API_KEY"),
        ("Arxiv (Academic)", get_arxiv_tool, None),  # No key needed
        ("AskNews", get_asknews_tool, "ASKNEWS_CLIENT_ID + SECRET"),
        ("Brave Search", get_brave_tool, "BRAVE_SEARCH_API_KEY"),
        ("DuckDuckGo", get_duckduckgo_tool, None),  # No key needed
        ("Google Serper", get_google_serper_tool, "SERPER_API_KEY"),
        ("SearXNG", get_searxng_tool, "SEARXNG_URL (optional)"),
        ("You.com", get_you_tool, "YDC_API_KEY"),
    ]

    for name, loader, key_name in tool_loaders:
        tool = loader()
        if tool:
            tools.append(tool)
            search_tools.append(name)
            if verbose:
                print(f"✅ {name}")
        else:
            missing_tools.append((name, key_name))
            if verbose:
                if key_name:
                    print(f"⚠️  {name} (missing {key_name})")
                else:
                    print(f"⚠️  {name} (package not installed)")

    if verbose:
        print("="*60)
        print(f"✅ Loaded {len(search_tools)} search tools")
        print(f"⚠️  {len(missing_tools)} tools unavailable")
        print(f"\nTotal available: {len(tools)} tools")

    return tools


def get_tool_info() -> Dict[str, Any]:
    """Get information about tool availability."""
    return {
        "search_tools": {
            "exa": EXA_AVAILABLE and bool(os.environ.get("EXA_API_KEY")),
            "tavily": TAVILY_AVAILABLE and bool(os.environ.get("TAVILY_API_KEY")),
            "arxiv": ARXIV_AVAILABLE,
            "asknews": ASKNEWS_AVAILABLE and bool(os.environ.get("ASKNEWS_CLIENT_ID")),
            "brave": BRAVE_AVAILABLE and bool(os.environ.get("BRAVE_SEARCH_API_KEY")),
            "duckduckgo": DUCKDUCKGO_AVAILABLE,
            "google_serper": GOOGLE_SERPER_AVAILABLE and bool(os.environ.get("SERPER_API_KEY")),
            "searxng": SEARXNG_AVAILABLE,
            "you": YOU_AVAILABLE and bool(os.environ.get("YDC_API_KEY")),
        },
        "packages_available": {
            "exa": EXA_AVAILABLE,
            "tavily": TAVILY_AVAILABLE,
            "arxiv": ARXIV_AVAILABLE,
            "asknews": ASKNEWS_AVAILABLE,
            "brave": BRAVE_AVAILABLE,
            "duckduckgo": DUCKDUCKGO_AVAILABLE,
            "google_serper": GOOGLE_SERPER_AVAILABLE,
            "searxng": SEARXNG_AVAILABLE,
            "you": YOU_AVAILABLE,
        }
    }


# ==================== Example Usage ====================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("TOOLS REGISTRY TEST")
    print("="*60)

    # Get all available tools
    tools = get_all_available_tools(verbose=True)

    # Show tool info
    print("\n" + "="*60)
    print("DETAILED TOOL INFO")
    print("="*60)

    info = get_tool_info()

    print("\n📊 Search Tools:")
    for name, available in info["search_tools"].items():
        status = "✅ Available" if available else "⚠️  Not available"
        print(f"  {name}: {status}")

    print("\n" + "="*60)
    print(f"✅ TEST COMPLETE - {len(tools)} tools ready to use")
    print("="*60)
