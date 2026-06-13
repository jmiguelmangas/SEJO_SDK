import asyncio

from SEJO_SDK.tools import Tool
from SEJO_SDK.tools.database_postgresql import DatabasePostgreSQLTool
from SEJO_SDK.tools.websearch_tool import WebSearchTool, create_web_search_tool


def test_tool_run_calls_wrapped_function():
    tool = Tool(
        name="double",
        description="Double a number",
        func=lambda value: value * 2,
    )

    assert tool.run(21) == 42


def test_tool_arun_awaits_async_function():
    async def double(value):
        return value * 2

    async def exercise():
        tool = Tool(
            name="double",
            description="Double a number",
            func=double,
        )

        assert await tool.arun(21) == 42

    asyncio.run(exercise())


def test_tool_to_schema_uses_function_shape():
    parameters = {
        "type": "object",
        "properties": {"value": {"type": "number"}},
        "required": ["value"],
    }
    tool = Tool(
        name="double",
        description="Double a number",
        func=lambda value: value * 2,
        parameters=parameters,
    )

    assert tool.to_schema() == {
        "type": "function",
        "function": {
            "name": "double",
            "description": "Double a number",
            "parameters": parameters,
        },
    }


def test_web_search_tool_formats_results():
    results = [
        {
            "title": "SEJO SDK",
            "url": "https://example.com",
            "snippet": "Provider agnostic agents.",
        }
    ]

    assert "SEJO SDK" in WebSearchTool._format_results(results)
    assert WebSearchTool._format_results([]) == "No results found."


def test_web_search_tool_returns_errors_from_search():
    class FailingWebSearchTool(WebSearchTool):
        def _search_web(self, query):
            raise RuntimeError("offline")

    tool = FailingWebSearchTool()

    assert tool.run("anything") == "Error performing web search: offline"


def test_create_web_search_tool_passes_configuration():
    tool = create_web_search_tool(max_results=2, region="us-en")

    assert tool.max_results == 2
    assert tool.region == "us-en"


def test_database_postgresql_tool_formats_empty_results():
    tool = DatabasePostgreSQLTool(
        name="db",
        description="Run SQL",
        func=lambda query: "",
    )

    assert tool.run("select 1") == "No results found."
