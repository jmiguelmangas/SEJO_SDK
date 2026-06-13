"""Web search tool implementation using DuckDuckGo search."""

from typing import Any, Optional

from SEJO_SDK.errors import ProviderDependencyError
from SEJO_SDK.tools.base import Tool


class WebSearchTool(Tool):
    """Tool for performing web searches using DuckDuckGo."""

    def __init__(
        self,
        max_results: int = 5,
        region: str = "wt-wt",
        timelimit: Optional[str] = None,
    ):
        super().__init__(
            name="web_search",
            description=(
                "Useful for searching the web for current information. "
                "Input should be a search query string. "
                f"Returns up to {max_results} search results."
            ),
            func=self.run,
        )
        self.max_results = max_results
        self.region = region
        self.timelimit = timelimit

    def run(self, query: str) -> str:
        """Execute a web search and return formatted results."""
        try:
            results = self._search_web(query)
            return self._format_results(results)
        except Exception as exc:
            return f"Error performing web search: {str(exc)}"

    def _search_web(self, query: str) -> list[dict[str, Any]]:
        try:
            from duckduckgo_search import DDGS
        except ImportError as exc:
            raise ProviderDependencyError(
                "Install web search support with `pip install sejo-sdk[websearch]`."
            ) from exc

        with DDGS() as ddgs:
            return [
                {
                    "title": result.get("title", ""),
                    "url": result.get("href", ""),
                    "snippet": result.get("body", ""),
                }
                for result in ddgs.text(
                    query,
                    max_results=self.max_results,
                    region=self.region,
                    timelimit=self.timelimit,
                )
            ]

    @staticmethod
    def _format_results(results: list[dict[str, str]]) -> str:
        if not results:
            return "No results found."

        formatted = []
        for index, result in enumerate(results, 1):
            formatted.append(
                f"{index}. {result.get('title', 'No title')}\n"
                f"   URL: {result.get('url', 'No URL')}\n"
                f"   {result.get('snippet', 'No description')}\n"
            )
        return "\n".join(formatted)


def create_web_search_tool(**kwargs) -> WebSearchTool:
    """Create and return a configured WebSearchTool instance."""

    return WebSearchTool(**kwargs)
