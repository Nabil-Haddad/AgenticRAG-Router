import json

from benchmarks.pubmedqa_dataset import (
    _download_pubmedqa,
    _load_cached,
    _save,
    get_pubmedqa_data,
)


def make_row(pubid: int, question: str, contexts: list[str]) -> dict:
    return {
        "pubid": pubid,
        "question": question,
        "context": {"contexts": contexts, "labels": [], "meshes": []},
        "long_answer": "some answer",
        "final_decision": "yes",
    }


# _download_pubmedqa

def test_download_pubmedqa_builds_corpus_queries_and_qrels(mocker):
    fake_rows = [
        make_row(111, "Is X true?", ["Background text.", "Results text."]),
        make_row(222, "Is Y true?", ["Only one section."]),
    ]
    mocker.patch("benchmarks.pubmedqa_dataset.load_dataset", return_value=fake_rows)

    corpus, queries, qrels = _download_pubmedqa()

    assert corpus == {
        "111": "Background text. Results text.",
        "222": "Only one section.",
    }
    assert queries == {"111": "Is X true?", "222": "Is Y true?"}
    assert qrels == {"111": ["111"], "222": ["222"]}


def test_download_pubmedqa_converts_pubid_to_string(mocker):
    fake_rows = [make_row(999, "Q?", ["ctx"])]
    mocker.patch("benchmarks.pubmedqa_dataset.load_dataset", return_value=fake_rows)

    corpus, _, _ = _download_pubmedqa()

    assert "999" in corpus
    assert 999 not in corpus


# _save / _load_cached

def test_save_writes_three_separate_files(tmp_path):
    corpus = {"1": "doc text"}
    queries = {"1": "question text"}
    qrels = {"1": ["1"]}

    _save(corpus, queries, qrels, tmp_path)

    assert json.loads((tmp_path / "corpus.json").read_text()) == corpus
    assert json.loads((tmp_path / "queries.json").read_text()) == queries
    assert json.loads((tmp_path / "qrels.json").read_text()) == qrels


def test_load_cached_returns_none_when_files_missing(tmp_path):
    assert _load_cached(tmp_path) is None


def test_load_cached_returns_none_when_only_some_files_exist(tmp_path):
    (tmp_path / "corpus.json").write_text("{}", encoding="utf-8")
    # queries.json and qrels.json intentionally missing

    assert _load_cached(tmp_path) is None


def test_load_cached_round_trips_with_save(tmp_path):
    corpus = {"1": "doc text"}
    queries = {"1": "question text"}
    qrels = {"1": ["1"]}
    _save(corpus, queries, qrels, tmp_path)

    result = _load_cached(tmp_path)

    assert result == (corpus, queries, qrels)


# get_pubmedqa_data

def test_get_pubmedqa_data_uses_cache_without_downloading(tmp_path, mocker):
    corpus = {"1": "doc text"}
    queries = {"1": "question text"}
    qrels = {"1": ["1"]}
    _save(corpus, queries, qrels, tmp_path)
    mock_download = mocker.patch("benchmarks.pubmedqa_dataset._download_pubmedqa")

    result = get_pubmedqa_data(tmp_path)

    assert result == (corpus, queries, qrels)
    mock_download.assert_not_called()


def test_get_pubmedqa_data_downloads_and_caches_when_missing(tmp_path, mocker):
    fake_rows = [make_row(111, "Is X true?", ["Background text."])]
    mocker.patch("benchmarks.pubmedqa_dataset.load_dataset", return_value=fake_rows)

    corpus, queries, qrels = get_pubmedqa_data(tmp_path)

    assert corpus == {"111": "Background text."}
    assert (tmp_path / "corpus.json").exists()
    assert (tmp_path / "queries.json").exists()
    assert (tmp_path / "qrels.json").exists()
