from SEJO_SDK.tools import Tool


def test_tool_run_calls_wrapped_function():
    tool = Tool(
        name="double",
        description="Double a number",
        func=lambda value: value * 2,
    )

    assert tool.run(21) == 42
