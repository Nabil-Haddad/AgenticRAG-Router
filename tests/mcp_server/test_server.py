import asyncio
import json

import pytest

from src.retrieval_arena.mcp_server.server import (
    TOOL_DESCRIPTIONS,
    _resolve_chunks_path,
    _resolve_collection,
    call_tool,
    list_tools,
)


@pytest.fixture(autouse=True)
def clear_collection_cache():
    # _resolve_collection is lru_cache'd at module scope, so a stale cached
    # collection from one test would otherwise leak into the next
    _resolve_collection.cache_clear()
    yield
    _resolve_collection.cache_clear()


def run(coro):
    return asyncio.run(coro)


# list_tools

def test_list_tools_returns_all_three_methods():
    tools = run(list_tools())

    assert {t.name for t in tools} == {"bm25_search", "vector_search", "hybrid_search"}


def test_list_tools_descriptions_match_and_share_the_input_schema():
    tools = run(list_tools())

    by_name = {t.name: t for t in tools}
    for name, description in TOOL_DESCRIPTIONS.items():
        assert by_name[name].description == description
        assert by_name[name].inputSchema["required"] == ["query"]


# call_tool - dispatch

def test_call_tool_dispatches_to_bm25_search(mocker):
    mock_bm25 = mocker.patch(
        "src.retrieval_arena.mcp_server.server.bm25_search",
        return_value=[{"idx": "c1", "text": "t", "score": 1.0, "source": "s"}],
    )

    result = run(call_tool("bm25_search", {"query": "q", "top_k": 3}))

    mock_bm25.assert_called_once_with("q", path=None, top_k=3)
    assert len(result) == 1
    assert json.loads(result[0].text) == [{"idx": "c1", "text": "t", "score": 1.0, "source": "s"}]


def test_call_tool_dispatches_to_vector_search(mocker):
    fake_collection = mocker.sentinel.collection
    mocker.patch("src.retrieval_arena.mcp_server.server._resolve_collection", return_value=fake_collection)
    mock_vector = mocker.patch(
        "src.retrieval_arena.mcp_server.server.cosign_simularity",
        return_value=[{"idx": "c2", "text": "t", "score": 0.9, "source": "s"}],
    )

    result = run(call_tool("vector_search", {"query": "q", "top_k": 4}))

    mock_vector.assert_called_once_with("q", top_k=4, collection=fake_collection)
    assert json.loads(result[0].text) == [{"idx": "c2", "text": "t", "score": 0.9, "source": "s"}]


def test_call_tool_dispatches_to_hybrid_search(mocker):
    fake_collection = mocker.sentinel.collection
    mocker.patch("src.retrieval_arena.mcp_server.server._resolve_collection", return_value=fake_collection)
    mock_hybrid = mocker.patch(
        "src.retrieval_arena.mcp_server.server.search_hybrid_rrf",
        return_value=[{"idx": "c3", "text": "t", "score": 0.8, "source": "s"}],
    )

    result = run(call_tool("hybrid_search", {"query": "q", "top_k": 2}))

    mock_hybrid.assert_called_once_with("q", top_k=2, bm25_path=None, collection=fake_collection)
    assert json.loads(result[0].text) == [{"idx": "c3", "text": "t", "score": 0.8, "source": "s"}]


def test_call_tool_defaults_top_k_to_5_when_not_provided(mocker):
    mock_bm25 = mocker.patch("src.retrieval_arena.mcp_server.server.bm25_search", return_value=[])

    run(call_tool("bm25_search", {"query": "q"}))

    mock_bm25.assert_called_once_with("q", path=None, top_k=5)


def test_call_tool_raises_on_unknown_tool_name():
    with pytest.raises(ValueError, match="Unknown tool"):
        run(call_tool("made_up_tool", {"query": "q"}))


def test_call_tool_raises_on_missing_query():
    with pytest.raises(KeyError):
        run(call_tool("bm25_search", {}))


# call_tool - env-var scoping

def test_call_tool_resolves_collection_name_from_env_var(monkeypatch, mocker):
    monkeypatch.setenv("RETRIEVAL_ARENA_MCP_COLLECTION", "custom_collection")
    mock_get_collection = mocker.patch(
        "src.retrieval_arena.mcp_server.server.get_collection", return_value=mocker.sentinel.collection
    )
    mocker.patch("src.retrieval_arena.mcp_server.server.cosign_simularity", return_value=[])

    run(call_tool("vector_search", {"query": "q"}))

    mock_get_collection.assert_called_once_with("custom_collection")


def test_call_tool_resolves_collection_as_none_when_env_var_unset(monkeypatch, mocker):
    monkeypatch.delenv("RETRIEVAL_ARENA_MCP_COLLECTION", raising=False)
    mock_get_collection = mocker.patch(
        "src.retrieval_arena.mcp_server.server.get_collection", return_value=mocker.sentinel.collection
    )
    mocker.patch("src.retrieval_arena.mcp_server.server.cosign_simularity", return_value=[])

    run(call_tool("vector_search", {"query": "q"}))

    mock_get_collection.assert_called_once_with(None)


def test_call_tool_resolves_chunks_path_from_env_var(monkeypatch, mocker, tmp_path):
    chunks_file = tmp_path / "chunks.json"
    monkeypatch.setenv("RETRIEVAL_ARENA_MCP_CHUNKS_PATH", str(chunks_file))
    mock_bm25 = mocker.patch("src.retrieval_arena.mcp_server.server.bm25_search", return_value=[])

    run(call_tool("bm25_search", {"query": "q"}))

    mock_bm25.assert_called_once_with("q", path=chunks_file, top_k=5)


def test_call_tool_resolves_chunks_path_as_none_when_env_var_unset(monkeypatch, mocker):
    monkeypatch.delenv("RETRIEVAL_ARENA_MCP_CHUNKS_PATH", raising=False)
    mock_bm25 = mocker.patch("src.retrieval_arena.mcp_server.server.bm25_search", return_value=[])

    run(call_tool("bm25_search", {"query": "q"}))

    mock_bm25.assert_called_once_with("q", path=None, top_k=5)


# _resolve_chunks_path

def test_resolve_chunks_path_returns_none_when_unset(monkeypatch):
    monkeypatch.delenv("RETRIEVAL_ARENA_MCP_CHUNKS_PATH", raising=False)

    assert _resolve_chunks_path() is None


def test_resolve_chunks_path_returns_path_when_set(monkeypatch, tmp_path):
    target = tmp_path / "chunks.json"
    monkeypatch.setenv("RETRIEVAL_ARENA_MCP_CHUNKS_PATH", str(target))

    assert _resolve_chunks_path() == target


# _resolve_collection caching

def test_resolve_collection_caches_by_name(mocker):
    mock_get_collection = mocker.patch(
        "src.retrieval_arena.mcp_server.server.get_collection", return_value=mocker.sentinel.collection
    )

    first = _resolve_collection("same_name")
    second = _resolve_collection("same_name")

    assert first is second
    mock_get_collection.assert_called_once_with("same_name")


def test_resolve_collection_does_not_share_cache_across_names(mocker):
    mocker.patch(
        "src.retrieval_arena.mcp_server.server.get_collection",
        side_effect=lambda name: f"collection-{name}",
    )

    assert _resolve_collection("a") == "collection-a"
    assert _resolve_collection("b") == "collection-b"
