from SEJO_SDK.messages import (
    Message,
    ModelResponse,
    ToolCall,
    assistant_message,
    messages_to_dicts,
    messages_to_prompt,
    system_message,
    tool_message,
    user_message,
)


def test_message_helpers_create_typed_messages():
    assert system_message("Rules").to_dict() == {
        "role": "system",
        "content": "Rules",
    }
    assert user_message("Hi").to_dict() == {"role": "user", "content": "Hi"}
    assert assistant_message("Hello").to_dict() == {
        "role": "assistant",
        "content": "Hello",
    }
    assert tool_message("42", name="answer", tool_call_id="call_1").to_dict() == {
        "role": "tool",
        "content": "42",
        "name": "answer",
        "tool_call_id": "call_1",
    }


def test_message_and_tool_call_round_trip_from_dict():
    message = Message.from_dict(
        {
            "role": "tool",
            "content": "ok",
            "name": "lookup",
            "tool_call_id": "call_1",
        }
    )
    tool_call = ToolCall.from_dict(
        {
            "id": "call_1",
            "name": "lookup",
            "arguments": {"query": "SEJO"},
        }
    )

    assert message.to_dict()["name"] == "lookup"
    assert tool_call.to_dict() == {
        "id": "call_1",
        "name": "lookup",
        "arguments": {"query": "SEJO"},
    }


def test_model_response_serializes_tool_calls():
    response = ModelResponse(
        content="Need a tool",
        tool_calls=[ToolCall(id="call_1", name="add", arguments={"left": 2})],
    )

    assert response.to_dict() == {
        "content": "Need a tool",
        "tool_calls": [
            {
                "id": "call_1",
                "name": "add",
                "arguments": {"left": 2},
            }
        ],
    }


def test_message_collection_helpers_accept_dicts_and_messages():
    messages = [
        system_message("Rules"),
        {"role": "user", "content": "Hello"},
        tool_message("42", name="lookup"),
    ]

    assert messages_to_prompt(messages) == "system: Rules\nuser: Hello\ntool lookup: 42"
    assert messages_to_dicts(messages) == [
        {"role": "system", "content": "Rules"},
        {"role": "user", "content": "Hello"},
        {"role": "tool", "content": "42", "name": "lookup"},
    ]
