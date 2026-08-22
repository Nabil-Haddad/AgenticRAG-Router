import json

import matplotlib.pyplot as plt
import pytest

from benchmarks.pubmedqa_arena import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EVAL_TOP_K,
    K_VALUES,
    METHODS,
    _run_method,
    build_pubmedqa_index,
    evaluate_method,
    make_figure,
    run_arena,
    save_results,
)


# build_pubmedqa_index

def test_build_pubmedqa_index_reuses_existing_chunks_without_rebuilding(tmp_path, mocker):
    chunks_dir = tmp_path / "chunks_dir"
    chunks_dir.mkdir()
    (chunks_dir / "chunks.json").write_text("[]", encoding="utf-8")
    chunk_all = mocker.patch("benchmarks.pubmedqa_arena.chunk_all")
    store_index = mocker.patch("benchmarks.pubmedqa_arena.store_index")

    result = build_pubmedqa_index({"1": "text"}, chunks_dir, "some_collection")

    assert result == chunks_dir / "chunks.json"
    chunk_all.assert_not_called()
    store_index.assert_not_called()


def test_build_pubmedqa_index_builds_and_stores_when_missing(tmp_path, mocker):
    chunks_dir = tmp_path / "chunks_dir"
    chunk_all = mocker.patch("benchmarks.pubmedqa_arena.chunk_all", return_value=[["chunk_a"], ["chunk_b"]])
    save_chunks_json = mocker.patch("benchmarks.pubmedqa_arena.save_chunks_json")
    store_index = mocker.patch("benchmarks.pubmedqa_arena.store_index")

    result = build_pubmedqa_index({"1": "text one", "2": "text two"}, chunks_dir, "some_collection")

    assert result == chunks_dir / "chunks.json"
    documents = chunk_all.call_args.args[0]
    assert [doc.source for group in documents for doc in group] == ["1", "2"]
    assert chunk_all.call_args.kwargs == {"chunk_size": CHUNK_SIZE, "overlap": CHUNK_OVERLAP}
    save_chunks_json.assert_called_once_with(["chunk_a", "chunk_b"], chunks_dir)
    store_index.assert_called_once_with(["chunk_a", "chunk_b"], collection_name="some_collection")


def test_build_pubmedqa_index_force_rebuilds_even_if_chunks_exist(tmp_path, mocker):
    chunks_dir = tmp_path / "chunks_dir"
    chunks_dir.mkdir()
    (chunks_dir / "chunks.json").write_text("[]", encoding="utf-8")
    mocker.patch("benchmarks.pubmedqa_arena.chunk_all", return_value=[])
    mocker.patch("benchmarks.pubmedqa_arena.save_chunks_json")
    store_index = mocker.patch("benchmarks.pubmedqa_arena.store_index")

    build_pubmedqa_index({"1": "text"}, chunks_dir, "some_collection", force=True)

    store_index.assert_called_once()


# _run_method

def test_run_method_dispatches_to_bm25(mocker):
    bm25_search = mocker.patch("benchmarks.pubmedqa_arena.bm25_search", return_value=["bm25_result"])
    path = mocker.sentinel.path
    collection = mocker.sentinel.collection

    result = _run_method("bm25", "a question", top_k=10, chunks_path=path, collection=collection)

    assert result == ["bm25_result"]
    bm25_search.assert_called_once_with("a question", path=path, top_k=10)


def test_run_method_dispatches_to_vector(mocker):
    cosign_simularity = mocker.patch("benchmarks.pubmedqa_arena.cosign_simularity", return_value=["vector_result"])
    path = mocker.sentinel.path
    collection = mocker.sentinel.collection

    result = _run_method("vector", "a question", top_k=10, chunks_path=path, collection=collection)

    assert result == ["vector_result"]
    cosign_simularity.assert_called_once_with("a question", top_k=10, collection=collection)


def test_run_method_dispatches_to_hybrid(mocker):
    search_hybrid_rrf = mocker.patch("benchmarks.pubmedqa_arena.search_hybrid_rrf", return_value=["hybrid_result"])
    path = mocker.sentinel.path
    collection = mocker.sentinel.collection

    result = _run_method("hybrid", "a question", top_k=10, chunks_path=path, collection=collection)

    assert result == ["hybrid_result"]
    search_hybrid_rrf.assert_called_once_with("a question", top_k=10, bm25_path=path, collection=collection)


def test_run_method_raises_on_unknown_method(mocker):
    with pytest.raises(ValueError, match="Unknown method"):
        _run_method("made_up_method", "a question", top_k=10, chunks_path=mocker.sentinel.path, collection=None)


# evaluate_method

def test_evaluate_method_computes_recall_and_mrr_across_queries(mocker):
    # q1's relevant source is at rank 1, q2's relevant source is at rank 3 (outside K=1)
    results_by_query = {
        "q1": [{"source": "doc_a"}, {"source": "doc_b"}],
        "q2": [{"source": "doc_x"}, {"source": "doc_y"}, {"source": "doc_z"}],
    }
    mocker.patch("benchmarks.pubmedqa_arena._run_method", side_effect=lambda method, question, **kw: results_by_query[question])

    queries = {"q1": "q1", "q2": "q2"}
    qrels = {"q1": ["doc_a"], "q2": ["doc_z"]}

    result = evaluate_method("vector", queries, qrels, chunks_path=mocker.sentinel.path, collection=mocker.sentinel.collection)

    assert set(result["recall"]) == set(K_VALUES)
    # q1 hits at rank 1 (recall=1 for every k); q2 only hits once k >= 3
    assert result["recall"][1]["mean"] == 0.5
    assert result["recall"][3]["mean"] == 1.0
    assert result["recall"][10]["mean"] == 1.0
    # reciprocal rank: q1 -> 1/1, q2 -> 1/3
    assert result["mrr"]["mean"] == pytest.approx((1.0 + 1 / 3) / 2)
    assert result["mrr"]["stdev"] > 0.0


def test_evaluate_method_stdev_is_zero_for_a_single_query(mocker):
    mocker.patch("benchmarks.pubmedqa_arena._run_method", return_value=[{"source": "doc_a"}])

    result = evaluate_method("bm25", {"q1": "q1"}, {"q1": ["doc_a"]}, chunks_path=mocker.sentinel.path, collection=None)

    assert result["mrr"]["stdev"] == 0.0
    assert all(result["recall"][k]["stdev"] == 0.0 for k in K_VALUES)


def test_evaluate_method_passes_eval_top_k_through_to_run_method(mocker):
    run_method = mocker.patch("benchmarks.pubmedqa_arena._run_method", return_value=[])

    evaluate_method("hybrid", {"q1": "question text"}, {"q1": ["doc_a"]}, chunks_path=mocker.sentinel.path, collection=None)

    run_method.assert_called_once_with(
        "hybrid", "question text", top_k=EVAL_TOP_K, chunks_path=mocker.sentinel.path, collection=None
    )


# save_results

def test_save_results_writes_expected_schema(tmp_path):
    # dict keys are ints going in, since K_VALUES is a tuple of ints -
    # json.dump stringifies them, so the round-trip is expected to differ
    results = {"bm25": {"recall": {1: {"mean": 0.5, "stdev": 0.0}}, "mrr": {"mean": 0.5, "stdev": 0.0}}}

    output_path = save_results(results, n_queries=42, output_dir=tmp_path)

    assert output_path == tmp_path / "results.json"
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["n_queries"] == 42
    assert payload["chunk_size"] == CHUNK_SIZE
    assert payload["chunk_overlap"] == CHUNK_OVERLAP
    assert payload["eval_top_k"] == EVAL_TOP_K
    assert payload["results"] == {"bm25": {"recall": {"1": {"mean": 0.5, "stdev": 0.0}}, "mrr": {"mean": 0.5, "stdev": 0.0}}}


def test_save_results_creates_output_dir_if_missing(tmp_path):
    output_dir = tmp_path / "does_not_exist_yet"

    save_results({}, n_queries=0, output_dir=output_dir)

    assert output_dir.exists()


# make_figure

def _fake_results() -> dict:
    return {
        method: {
            "recall": {k: {"mean": 0.5 + 0.01 * k, "stdev": 0.1} for k in K_VALUES},
            "mrr": {"mean": 0.7, "stdev": 0.05},
        }
        for method in METHODS
    }


def test_make_figure_writes_a_png_and_closes_the_figure(tmp_path):
    output_path = tmp_path / "figure.png"

    make_figure(_fake_results(), n_queries=100, output_path=output_path)

    assert output_path.exists()
    assert output_path.stat().st_size > 0
    assert plt.get_fignums() == []


def test_make_figure_creates_parent_directory(tmp_path):
    output_path = tmp_path / "nested" / "dir" / "figure.png"

    make_figure(_fake_results(), n_queries=100, output_path=output_path)

    assert output_path.exists()


# run_arena

def test_run_arena_full_run_uses_the_real_collection_and_does_not_force_rebuild(mocker):
    mocker.patch(
        "benchmarks.pubmedqa_arena.get_pubmedqa_data",
        return_value=({"1": "doc text"}, {"1": "question"}, {"1": ["1"]}),
    )
    build_index = mocker.patch("benchmarks.pubmedqa_arena.build_pubmedqa_index", return_value=mocker.sentinel.chunks_path)
    mocker.patch("benchmarks.pubmedqa_arena.get_collection", return_value=mocker.sentinel.collection)
    mocker.patch(
        "benchmarks.pubmedqa_arena.evaluate_method",
        # a fresh dict per call - run_arena pops "per_query" off the result,
        # which would corrupt a single shared return_value= across the 3 calls
        side_effect=lambda *a, **kw: {"recall": {k: {"mean": 1.0} for k in K_VALUES}, "mrr": {"mean": 1.0}, "per_query": []},
    )
    mocker.patch("benchmarks.pubmedqa_arena.save_per_query_csvs", return_value={})
    save_results_mock = mocker.patch("benchmarks.pubmedqa_arena.save_results")
    make_figure_mock = mocker.patch("benchmarks.pubmedqa_arena.make_figure")

    run_arena(sample_size=None)

    from benchmarks.pubmedqa_arena import COLLECTION_NAME, PUBMEDQA_DIR

    build_index.assert_called_once_with({"1": "doc text"}, PUBMEDQA_DIR, COLLECTION_NAME, force=False)
    save_results_mock.assert_called_once()
    make_figure_mock.assert_called_once()


def test_run_arena_sample_run_subsets_data_and_uses_an_isolated_collection(mocker):
    corpus = {"1": "doc one", "2": "doc two", "3": "doc three"}
    queries = {"1": "q1", "2": "q2", "3": "q3"}
    qrels = {"1": ["1"], "2": ["2"], "3": ["3"]}
    mocker.patch("benchmarks.pubmedqa_arena.get_pubmedqa_data", return_value=(corpus, queries, qrels))
    build_index = mocker.patch("benchmarks.pubmedqa_arena.build_pubmedqa_index", return_value=mocker.sentinel.chunks_path)
    mocker.patch("benchmarks.pubmedqa_arena.get_collection", return_value=mocker.sentinel.collection)
    mocker.patch(
        "benchmarks.pubmedqa_arena.evaluate_method",
        # a fresh dict per call - run_arena pops "per_query" off the result,
        # which would corrupt a single shared return_value= across the 3 calls
        side_effect=lambda *a, **kw: {"recall": {k: {"mean": 1.0} for k in K_VALUES}, "mrr": {"mean": 1.0}, "per_query": []},
    )
    mocker.patch("benchmarks.pubmedqa_arena.save_per_query_csvs", return_value={})
    mocker.patch("benchmarks.pubmedqa_arena.save_results")
    mocker.patch("benchmarks.pubmedqa_arena.make_figure")

    run_arena(sample_size=2)

    from benchmarks.pubmedqa_arena import COLLECTION_NAME, PUBMEDQA_DIR

    called_corpus, chunks_dir, collection_name = build_index.call_args.args
    assert called_corpus == {"1": "doc one", "2": "doc two"}
    assert chunks_dir == PUBMEDQA_DIR / "sample"
    assert collection_name == f"{COLLECTION_NAME}_sample"
    assert build_index.call_args.kwargs == {"force": True}


def test_run_arena_evaluates_every_method_and_returns_their_results(mocker):
    mocker.patch(
        "benchmarks.pubmedqa_arena.get_pubmedqa_data",
        return_value=({"1": "doc text"}, {"1": "question"}, {"1": ["1"]}),
    )
    mocker.patch("benchmarks.pubmedqa_arena.build_pubmedqa_index", return_value=mocker.sentinel.chunks_path)
    mocker.patch("benchmarks.pubmedqa_arena.get_collection", return_value=mocker.sentinel.collection)
    evaluate_method_mock = mocker.patch(
        "benchmarks.pubmedqa_arena.evaluate_method",
        side_effect=lambda method, *a, **kw: {
            "recall": {k: {"mean": 0.5} for k in K_VALUES}, "mrr": {"mean": 0.5}, "per_query": [], "method": method,
        },
    )
    mocker.patch("benchmarks.pubmedqa_arena.save_results")
    mocker.patch("benchmarks.pubmedqa_arena.make_figure")
    mocker.patch("benchmarks.pubmedqa_arena.save_per_query_csvs", return_value={})

    results = run_arena(sample_size=None)

    assert set(results) == set(METHODS)
    assert {r["method"] for r in results.values()} == set(METHODS)
    assert evaluate_method_mock.call_count == len(METHODS)
