"""PostgreSQL database tool implementation."""

from SEJO_SDK.tools.base import Tool


class DatabasePostgreSQLTool(Tool):
    def __init__(self, name: str, description: str, func):
        super().__init__(name, description, func)

    def run(self, query: str) -> str:
        return self.func(query) or "No results found."
