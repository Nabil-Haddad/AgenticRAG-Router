import json
import os

import pytest

import src.retrieve as retrieve
from src.Config import Config
from src.retrieve import bm25_search, cosign_simularity, peek_first_5_elements, search_hybrid_rrf


@pytest.fixture(autouse=True)
def clear_bm25_cache():
    retrieve._bm25_cache.clear()
    yield
    retrieve._bm25_cache.clear()


def make_chunk_dict(id_: str, text: str, source: str = "paper.pdf") -> dict:
    return {
        "id": id_,
        "text": text,
        "source": source,
        "page_start": 1,
        "page_end": 1,
        "token_count": len(text.split()),
        "metadata": {},
    }


def write_chunks(path, chunks: list[dict]) -> None:
    path.write_text(json.dumps(chunks), encoding="utf-8")


# peek_first_5_elements

def test_peek_prints_samples_on_success(mocker, capsys):
    fake_collection = mocker.MagicMock()
    fake_collection.peek.return_value = {
        "ids": ["c1", "c2"],
        "documents": ["first document text", "second document text"],
    }
    mocker.patch("src.retrieve.get_collection", return_value=fake_collection)

    result = peek_first_5_elements()

    fake_collection.peek.assert_called_once_with(limit=5)
    assert result is None
    out = capsys.readouterr().out
    assert "VectorDB Connection Successful" in out
    assert "Retrieved 2 items" in out
    assert "c1" in out
    assert "c2" in out


def test_peek_handles_failure_without_crashing(mocker, capsys):
    fake_collection = mocker.MagicMock()
    fake_collection.peek.side_effect = Exception("simulated connection failure")
    mocker.patch("src.retrieve.get_collection", return_value=fake_collection)

    result = peek_first_5_elements()

    assert result is None
    out = capsys.readouterr().out
    assert "connection failed" in out
    assert "simulated connection failure" in out


# cosign_simularity

def test_cosign_simularity_embeds_query_and_returns_results(mocker):
    fake_collection = mocker.MagicMock()
    fake_collection.query.return_value = {
        "ids": [["c1", "c2"]],
        "documents": [["text one", "text two"]],
        "distances": [[0.1, 0.2]],
    }
    mocker.patch("src.retrieve.get_collection", return_value=fake_collection)
    mock_embed = mocker.patch("src.retrieve.embed_texts", return_value=[[0.5, 0.6]])

    results = cosign_simularity("what is RAG?", top_k=2)

    mock_embed.assert_called_once_with(["what is RAG?"])
    fake_collection.query.assert_called_once_with(
        query_embeddings=[[0.5, 0.6]],
        n_results=2,
    )
    assert results == [
        {"idx": "c1", "text": "text one", "score": 0.1},
        {"idx": "c2", "text": "text two", "score": 0.2},
    ]


def test_cosign_simularity_defaults_top_k_to_5(mocker):
    fake_collection = mocker.MagicMock()
    fake_collection.query.return_value = {"ids": [[]], "documents": [[]], "distances": [[]]}
    mocker.patch("src.retrieve.get_collection", return_value=fake_collection)
    mocker.patch("src.retrieve.embed_texts", return_value=[[0.1]])

    cosign_simularity("a query")

    fake_collection.query.assert_called_once_with(
        query_embeddings=[[0.1]],
        n_results=5,
    )


def test_cosign_simularity_returns_empty_list_on_failure(mocker):
    fake_collection = mocker.MagicMock()
    fake_collection.query.side_effect = Exception("simulated connection failure")
    mocker.patch("src.retrieve.get_collection", return_value=fake_collection)
    mocker.patch("src.retrieve.embed_texts", return_value=[[0.1]])

    assert cosign_simularity("a query") == []


# bm25_search

def test_bm25_search_returns_empty_list_when_file_missing(tmp_path):
    assert bm25_search("anything", path=tmp_path / "missing.json") == []


def test_bm25_search_returns_empty_list_when_corpus_empty(tmp_path):
    path = tmp_path / "chunks.json"
    write_chunks(path, [])

    assert bm25_search("anything", path=path) == []


def test_bm25_search_ranks_most_relevant_chunk_first(tmp_path):
    path = tmp_path / "chunks.json"
    write_chunks(path, [
        make_chunk_dict("c1", "cats are small furry animals"),
        make_chunk_dict("c2", "dogs are loyal furry animals"),
        make_chunk_dict("c3", "quantum computing uses qubits"),
    ])

    results = bm25_search("furry animals", path=path, top_k=2)

    assert [r["idx"] for r in results] == ["c1", "c2"] or [r["idx"] for r in results] == ["c2", "c1"]
    assert all(r["score"] > 0 for r in results)


def test_bm25_search_respects_top_k(tmp_path):
    path = tmp_path / "chunks.json"
    write_chunks(path, [make_chunk_dict(f"c{i}", "shared keyword text") for i in range(5)])

    results = bm25_search("shared keyword", path=path, top_k=2)

    assert len(results) == 2


def test_bm25_search_result_shape_matches_cosign_simularity(tmp_path):
    path = tmp_path / "chunks.json"
    write_chunks(path, [make_chunk_dict("c1", "hello world")])

    results = bm25_search("hello", path=path, top_k=1)

    assert len(results) == 1
    assert set(results[0].keys()) == {"idx", "text", "score"}
    assert results[0]["idx"] == "c1"
    assert results[0]["text"] == "hello world"
    assert isinstance(results[0]["score"], float)


def test_bm25_search_defaults_path_to_config_chunks_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(Config, "chunks_dir", tmp_path)
    write_chunks(tmp_path / "chunks.json", [make_chunk_dict("c1", "hello world")])

    results = bm25_search("hello")

    assert results[0]["idx"] == "c1"


def test_bm25_search_reuses_cached_index_when_file_unchanged(tmp_path, mocker):
    path = tmp_path / "chunks.json"
    write_chunks(path, [make_chunk_dict("c1", "hello world")])
    spy = mocker.spy(retrieve, "load_chunks")

    bm25_search("hello", path=path)
    bm25_search("world", path=path)

    spy.assert_called_once()


def test_bm25_search_rebuilds_index_when_file_changes(tmp_path, mocker):
    path = tmp_path / "chunks.json"
    write_chunks(path, [make_chunk_dict("c1", "hello world")])
    spy = mocker.spy(retrieve, "load_chunks")

    bm25_search("hello", path=path)

    write_chunks(path, [make_chunk_dict("c1", "hello world"), make_chunk_dict("c2", "goodbye moon")])
    # force a distinct mtime regardless of the filesystem's timestamp resolution
    new_mtime = path.stat().st_mtime + 5
    os.utime(path, (new_mtime, new_mtime))

    results = bm25_search("goodbye", path=path)

    assert spy.call_count == 2
    assert results[0]["idx"] == "c2"


def test_bm25_search_returns_empty_list_on_unexpected_error(tmp_path, mocker, capsys):
    path = tmp_path / "chunks.json"
    write_chunks(path, [make_chunk_dict("c1", "hello world")])
    mocker.patch("src.retrieve._get_bm25_index", side_effect=Exception("simulated failure"))

    results = bm25_search("hello", path=path)

    assert results == []
    out = capsys.readouterr().out
    assert "bm25 search failed" in out
    assert "simulated failure" in out


# search_hybrid_rrf

def test_search_hybrid_rrf_ranks_chunks_in_both_rankings_highest(mocker):
    mocker.patch("src.retrieve.bm25_search", return_value=[
        {"idx": "c1", "text": "chunk one", "score": 5.0},
        {"idx": "c2", "text": "chunk two", "score": 3.0},
    ])
    mocker.patch("src.retrieve.cosign_simularity", return_value=[
        {"idx": "c2", "text": "chunk two", "score": 0.1},
        {"idx": "c3", "text": "chunk three", "score": 0.2},
    ])

    results = search_hybrid_rrf("query", top_k=3)

    # c2 appears in both rankings (bm25 rank 2, vector rank 1), so its fused
    # score should beat c1 and c3, which each appear in only one ranking
    assert [r["idx"] for r in results] == ["c2", "c1", "c3"]


def test_search_hybrid_rrf_respects_top_k(mocker):
    mocker.patch("src.retrieve.bm25_search", return_value=[
        {"idx": f"c{i}", "text": f"text {i}", "score": 1.0} for i in range(5)
    ])
    mocker.patch("src.retrieve.cosign_simularity", return_value=[])

    results = search_hybrid_rrf("query", top_k=2)

    assert len(results) == 2


def test_search_hybrid_rrf_passes_candidate_k_to_both_backends(mocker):
    bm25_mock = mocker.patch("src.retrieve.bm25_search", return_value=[])
    vector_mock = mocker.patch("src.retrieve.cosign_simularity", return_value=[])

    search_hybrid_rrf("query", top_k=5, candidate_k=15)

    bm25_mock.assert_called_once_with("query", top_k=15)
    vector_mock.assert_called_once_with("query", top_k=15)


def test_search_hybrid_rrf_returns_empty_list_when_both_backends_empty(mocker):
    mocker.patch("src.retrieve.bm25_search", return_value=[])
    mocker.patch("src.retrieve.cosign_simularity", return_value=[])

    assert search_hybrid_rrf("query") == []


def test_search_hybrid_rrf_result_shape_and_score_formula(mocker):
    mocker.patch("src.retrieve.bm25_search", return_value=[{"idx": "c1", "text": "hello", "score": 1.0}])
    mocker.patch("src.retrieve.cosign_simularity", return_value=[])

    results = search_hybrid_rrf("query", top_k=1)

    assert results == [{"idx": "c1", "text": "hello", "score": pytest.approx(1 / 61)}]


def test_search_hybrid_rrf_k_parameter_changes_score_weighting(mocker):
    mocker.patch("src.retrieve.bm25_search", return_value=[{"idx": "c1", "text": "hello", "score": 1.0}])
    mocker.patch("src.retrieve.cosign_simularity", return_value=[])

    results = search_hybrid_rrf("query", top_k=1, k=10)

    assert results[0]["score"] == pytest.approx(1 / 11)
