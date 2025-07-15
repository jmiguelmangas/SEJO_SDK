"""Web search tool implementation using DuckDuckGo search."""

from typing import List, Dict, Any, Optional
import json
from duckduckgo_search import DDGS
from SEJO_SDK.tools import Tool


class WebSearchTool(Tool):
    """A tool for performing web searches using DuckDuckGo.
    
    This tool allows performing web searches and retrieving results in a structured format.
    It includes features like limiting the number of results, filtering by region and time.
    """
    
    def __init__(self, max_results: int = 5, region: str = 'wt-wt', timelimit: Optional[str] = None):
        """Initialize the WebSearchTool.
        
        Args:
            max_results: Maximum number of search results to return (default: 5)
            region: Region code for search results (default: 'wt-wt' for worldwide)
            timelimit: Time period to search (e.g., 'd' for day, 'w' for week, 'm' for month)
        """
        super().__init__(
            name="web_search",
            description=(
                "Useful for searching the web for current information. "
                "Input should be a search query string. "
                f"Returns up to {max_results} search results."
            ),
            func=self.run
        )
        self.max_results = max_results
        self.region = region
        self.timelimit = timelimit
    
    def run(self, query: str) -> str:
        """Execute a web search and return formatted results.
        
        Args:
            query: The search query string
            
        Returns:
            str: Formatted search results or error message
        """
        try:
            results = self._search_web(query)
            return self._format_results(results)
        except Exception as e:
            return f"Error performing web search: {str(e)}"
    
    def _search_web(self, query: str) -> List[Dict[str, Any]]:
        """Perform the actual web search.
        
        Args:
            query: The search query string
            
        Returns:
            List of search result dictionaries
        """
        with DDGS() as ddgs:
            return [
                {
                    'title': r.get('title', ''),
                    'url': r.get('href', ''),
                    'snippet': r.get('body', '')
                }
                for r in ddgs.text(
                    query,
                    max_results=self.max_results,
                    region=self.region,
                    timelimit=self.timelimit
                )
            ]
    
    @staticmethod
    def _format_results(results: List[Dict[str, str]]) -> str:
        """Format search results into a readable string.
        
        Args:
            results: List of search result dictionaries
            
        Returns:
            Formatted string with search results
        """
        if not results:
            return "No results found."
            
        formatted = []
        for i, result in enumerate(results, 1):
            formatted.append(
                f"{i}. {result.get('title', 'No title')}\n"
                f"   URL: {result.get('url', 'No URL')}\n"
                f"   {result.get('snippet', 'No description')}\n"
            )
        return "\n".join(formatted)


def create_web_search_tool(**kwargs) -> WebSearchTool:
    """Create and return a configured WebSearchTool instance.
    
    Args:
        **kwargs: Arguments to pass to WebSearchTool constructor
        
    Returns:
        Configured WebSearchTool instance
    """
    return WebSearchTool(**kwargs)
