import asyncio
import json
import sys

from anthropic.types import TextBlock, ToolUseBlock

from src.retrieval_arena.config import Config
from src.retrieval_arena.mcp_server.client import (
    SERVER_MODULE,
    RetrievalChoiceResult,
    ask_with_mcp_tools,
    run_conversation,
)


def run(coro):
    return asyncio.run(coro)


class _FakeAsyncCM:
    def __init__(self, value):
        self._value = value

    async def __aenter__(self):
        return self._value

    async def __aexit__(self, *exc_info):
        return False


def make_mcp_tool(mocker, name: str, description: str = "desc", input_schema: dict | None = None):
    # MagicMock(name=...) is reserved for the mock's own repr, not an attribute -
    # must be set after construction, not passed as a constructor kwarg
    tool = mocker.MagicMock()
    tool.name = name
    tool.description = description
    tool.inputSchema = input_schema or {"type": "object", "properties": {}}
    return tool


def make_session(mocker, tools: list, call_tool_result_text: str | None = None):
    session = mocker.MagicMock()
    session.list_tools = mocker.AsyncMock(return_value=mocker.MagicMock(tools=tools))
    if call_tool_result_text is not None:
        fake_result = mocker.MagicMock()
        fake_result.content = [mocker.MagicMock(text=call_tool_result_text)]
        session.call_tool = mocker.AsyncMock(return_value=fake_result)
    return session


# run_conversation - happy path

def test_run_conversation_executes_the_chosen_tool_and_returns_a_final_answer(mocker):
    tool = make_mcp_tool(mocker, "vector_search")
    retrieved_payload = [{"idx": "c1", "text": "t", "score": 0.9, "source": "s"}]
    session = make_session(mocker, [tool], call_tool_result_text=json.dumps(retrieved_payload))

    tool_use_block = ToolUseBlock(id="toolu_1", name="vector_search", input={"query": "q"}, type="tool_use")
    first_response = mocker.MagicMock(content=[tool_use_block])
    final_text_block = TextBlock(text="the answer", type="text", citations=None)
    second_response = mocker.MagicMock(content=[final_text_block])
    anthropic_client = mocker.MagicMock()
    anthropic_client.messages.create = mocker.MagicMock(side_effect=[first_response, second_response])

    result = run(run_conversation(session, anthropic_client, "what is RAG?", "claude-haiku-4-5-20251001", top_k=3))

    assert result == RetrievalChoiceResult(
        question="what is RAG?",
        chosen_method="vector_search",
        tool_input={"query": "q"},
        retrieved_results=retrieved_payload,
        claude_final_text="the answer",
    )
    session.call_tool.assert_awaited_once_with("vector_search", {"query": "q"})


def test_run_conversation_forces_exactly_one_non_parallel_tool_call(mocker):
    tool = make_mcp_tool(mocker, "vector_search")
    session = make_session(mocker, [tool], call_tool_result_text="[]")
    tool_use_block = ToolUseBlock(id="toolu_1", name="vector_search", input={"query": "q"}, type="tool_use")
    anthropic_client = mocker.MagicMock()
    anthropic_client.messages.create = mocker.MagicMock(side_effect=[
        mocker.MagicMock(content=[tool_use_block]),
        mocker.MagicMock(content=[TextBlock(text="", type="text", citations=None)]),
    ])

    run(run_conversation(session, anthropic_client, "q", "model"))

    first_call_kwargs = anthropic_client.messages.create.call_args_list[0].kwargs
    assert first_call_kwargs["tool_choice"] == {"type": "any", "disable_parallel_tool_use": True}


def test_run_conversation_includes_top_k_in_the_prompt(mocker):
    tool = make_mcp_tool(mocker, "vector_search")
    session = make_session(mocker, [tool], call_tool_result_text="[]")
    tool_use_block = ToolUseBlock(id="toolu_1", name="vector_search", input={"query": "q"}, type="tool_use")
    anthropic_client = mocker.MagicMock()
    anthropic_client.messages.create = mocker.MagicMock(side_effect=[
        mocker.MagicMock(content=[tool_use_block]),
        mocker.MagicMock(content=[TextBlock(text="", type="text", citations=None)]),
    ])

    run(run_conversation(session, anthropic_client, "what is RAG?", "model", top_k=7))

    prompt = anthropic_client.messages.create.call_args_list[0].kwargs["messages"][0]["content"]
    assert "top_k=7" in prompt
    assert "what is RAG?" in prompt


def test_run_conversation_converts_mcp_tools_to_anthropic_tool_format(mocker):
    tool = make_mcp_tool(mocker, "hybrid_search", description="hybrid desc", input_schema={"type": "object", "required": ["query"]})
    session = make_session(mocker, [tool], call_tool_result_text="[]")
    tool_use_block = ToolUseBlock(id="toolu_1", name="hybrid_search", input={"query": "q"}, type="tool_use")
    anthropic_client = mocker.MagicMock()
    anthropic_client.messages.create = mocker.MagicMock(side_effect=[
        mocker.MagicMock(content=[tool_use_block]),
        mocker.MagicMock(content=[TextBlock(text="", type="text", citations=None)]),
    ])

    run(run_conversation(session, anthropic_client, "q", "model"))

    tools_sent = anthropic_client.messages.create.call_args_list[0].kwargs["tools"]
    assert tools_sent == [{
        "name": "hybrid_search",
        "description": "hybrid desc",
        "input_schema": {"type": "object", "required": ["query"]},
    }]


def test_run_conversation_sends_the_correct_followup_message_structure(mocker):
    tool = make_mcp_tool(mocker, "vector_search")
    session = make_session(mocker, [tool], call_tool_result_text=json.dumps([{"idx": "c1"}]))
    tool_use_block = ToolUseBlock(id="toolu_42", name="vector_search", input={"query": "q"}, type="tool_use")
    first_response = mocker.MagicMock(content=[tool_use_block])
    anthropic_client = mocker.MagicMock()
    anthropic_client.messages.create = mocker.MagicMock(side_effect=[
        first_response,
        mocker.MagicMock(content=[TextBlock(text="answer", type="text", citations=None)]),
    ])

    run(run_conversation(session, anthropic_client, "the question", "model"))

    followup_messages = anthropic_client.messages.create.call_args_list[1].kwargs["messages"]
    assert followup_messages == [
        {"role": "user", "content": "the question"},
        {"role": "assistant", "content": first_response.content},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "toolu_42", "content": json.dumps([{"idx": "c1"}])}
        ]},
    ]


# run_conversation - no tool used

def test_run_conversation_returns_early_when_claude_uses_no_tool(mocker):
    tool = make_mcp_tool(mocker, "vector_search")
    session = make_session(mocker, [tool])
    text_block = TextBlock(text="no tool needed", type="text", citations=None)
    anthropic_client = mocker.MagicMock()
    anthropic_client.messages.create = mocker.MagicMock(return_value=mocker.MagicMock(content=[text_block]))

    result = run(run_conversation(session, anthropic_client, "q", "model"))

    assert result == RetrievalChoiceResult(
        question="q", chosen_method=None, tool_input=None, retrieved_results=[], claude_final_text="no tool needed"
    )
    session.call_tool.assert_not_called()
    assert anthropic_client.messages.create.call_count == 1


# ask_with_mcp_tools

def test_ask_with_mcp_tools_builds_stdio_params_and_delegates_to_run_conversation(mocker):
    fake_session = mocker.MagicMock()
    fake_session.initialize = mocker.AsyncMock()
    mock_stdio_client = mocker.patch(
        "src.retrieval_arena.mcp_server.client.stdio_client",
        return_value=_FakeAsyncCM(("read", "write")),
    )
    mock_client_session_cls = mocker.patch(
        "src.retrieval_arena.mcp_server.client.ClientSession",
        return_value=_FakeAsyncCM(fake_session),
    )
    mock_run_conversation = mocker.patch(
        "src.retrieval_arena.mcp_server.client.run_conversation",
        new=mocker.AsyncMock(return_value=mocker.sentinel.result),
    )
    mocker.patch(
        "src.retrieval_arena.mcp_server.client.get_default_environment",
        return_value={"PATH": "/usr/bin"},
    )
    mock_params_cls = mocker.patch("src.retrieval_arena.mcp_server.client.StdioServerParameters")

    fake_anthropic_client = mocker.sentinel.anthropic_client
    result = run(ask_with_mcp_tools(
        "hello", model="a-model", top_k=7, server_env={"X": "1"}, anthropic_client=fake_anthropic_client
    ))

    assert result is mocker.sentinel.result
    mock_params_cls.assert_called_once_with(
        command=sys.executable,
        args=["-m", SERVER_MODULE],
        cwd=str(Config.project_root),
        env={"PATH": "/usr/bin", "X": "1"},
    )
    mock_stdio_client.assert_called_once_with(mock_params_cls.return_value)
    mock_client_session_cls.assert_called_once_with("read", "write")
    fake_session.initialize.assert_awaited_once()
    mock_run_conversation.assert_awaited_once_with(fake_session, fake_anthropic_client, "hello", "a-model", 7)


def test_ask_with_mcp_tools_constructs_an_anthropic_client_when_none_given(mocker):
    fake_session = mocker.MagicMock()
    fake_session.initialize = mocker.AsyncMock()
    mocker.patch(
        "src.retrieval_arena.mcp_server.client.stdio_client",
        return_value=_FakeAsyncCM(("read", "write")),
    )
    mocker.patch(
        "src.retrieval_arena.mcp_server.client.ClientSession",
        return_value=_FakeAsyncCM(fake_session),
    )
    mocker.patch(
        "src.retrieval_arena.mcp_server.client.run_conversation",
        new=mocker.AsyncMock(return_value=mocker.sentinel.result),
    )
    mocker.patch("src.retrieval_arena.mcp_server.client.get_default_environment", return_value={})
    mocker.patch("src.retrieval_arena.mcp_server.client.StdioServerParameters")
    mock_anthropic_cls = mocker.patch("src.retrieval_arena.mcp_server.client.anthropic.Anthropic")

    run(ask_with_mcp_tools("hello"))

    mock_anthropic_cls.assert_called_once_with(api_key=Config.ANTHROPIC_API_KEY)


def test_ask_with_mcp_tools_does_not_construct_a_client_when_one_is_given(mocker):
    fake_session = mocker.MagicMock()
    fake_session.initialize = mocker.AsyncMock()
    mocker.patch(
        "src.retrieval_arena.mcp_server.client.stdio_client",
        return_value=_FakeAsyncCM(("read", "write")),
    )
    mocker.patch(
        "src.retrieval_arena.mcp_server.client.ClientSession",
        return_value=_FakeAsyncCM(fake_session),
    )
    mocker.patch(
        "src.retrieval_arena.mcp_server.client.run_conversation",
        new=mocker.AsyncMock(return_value=mocker.sentinel.result),
    )
    mocker.patch("src.retrieval_arena.mcp_server.client.get_default_environment", return_value={})
    mocker.patch("src.retrieval_arena.mcp_server.client.StdioServerParameters")
    mock_anthropic_cls = mocker.patch("src.retrieval_arena.mcp_server.client.anthropic.Anthropic")

    run(ask_with_mcp_tools("hello", anthropic_client=mocker.sentinel.given_client))

    mock_anthropic_cls.assert_not_called()
