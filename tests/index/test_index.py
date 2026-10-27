import json
from pathlib import Path

import pytest

from src.retrieval_arena.config import Config
from src.retrieval_arena.ingestion.chunk import Chunk
from src.retrieval_arena.index.index import (
    CollectionMismatchError,
    build_index,
    get_collection,
    load_data,
    store_index,
)


def write_chunks_json(path: Path, chunks: list[dict]) -> None:
    path.write_text(json.dumps(chunks), encoding="utf-8")


def make_chunk_dict( chunk_id: str,text: str,source: str = "paper.pdf",page_start: int = 1,page_end: int = 1,token_count: int = 10,metadata: dict | None = None,) -> dict:
    return {
        "id": chunk_id,
        "text": text,
        "source": source,
        "page_start": page_start,
        "page_end": page_end,
        "token_count": token_count,
        "metadata": metadata or {},
    }


def make_chunk(
    chunk_id: str,
    text: str,
    source: str = "paper.pdf",
    page_start: int = 1,
    page_end: int = 1,
    token_count: int = 10,
    metadata: dict | None = None,
) -> Chunk:
    return Chunk(
        id=chunk_id,
        text=text,
        source=source,
        page_start=page_start,
        page_end=page_end,
        token_count=token_count,
        metadata=metadata or {},
    )


# load_data

def test_load_data_reads_chunks_from_json(tmp_path):
    path = tmp_path / "chunks.json"
    write_chunks_json(path, [make_chunk_dict("c1", "text one"), make_chunk_dict("c2", "text two")])

    chunks = load_data(path)

    assert len(chunks) == 2
    assert chunks[0].id == "c1"
    assert chunks[0].text == "text one"
    assert chunks[1].id == "c2"


def mock_chroma(mocker, space: str = "cosine", embed_model: str | None = None):
    # a Chroma client whose collection reports the given space and embedding model
    mock_client_cls = mocker.patch("src.retrieval_arena.index.index.chromadb.PersistentClient")
    mock_client = mock_client_cls.return_value
    collection = mock_client.get_or_create_collection.return_value
    collection.configuration = {"hnsw": {"space": space}}
    collection.metadata = {"embed_model": embed_model or Config.EMBED_MODEL_NAME}
    return mock_client_cls, mock_client, collection


# get_collection

def test_get_collection_uses_configured_path_and_name(mocker):
    mock_client_cls, mock_client, _ = mock_chroma(mocker)

    collection = get_collection()

    mock_client_cls.assert_called_once_with(path=str(Config.VECTORDB_DIR))
    mock_client.get_or_create_collection.assert_called_once_with(
        name=Config.COLLECTION_NAME,
        configuration={"hnsw": {"space": "cosine"}},
        metadata={"embed_model": Config.EMBED_MODEL_NAME},
    )
    assert collection is mock_client.get_or_create_collection.return_value


def test_get_collection_uses_a_given_name_instead_of_the_default(mocker):
    _, mock_client, _ = mock_chroma(mocker)

    get_collection("custom_collection")

    assert mock_client.get_or_create_collection.call_args.kwargs["name"] == "custom_collection"


def test_get_collection_rejects_a_collection_with_the_wrong_distance_space(mocker):
    mock_chroma(mocker, space="l2")

    with pytest.raises(CollectionMismatchError, match="distance space is 'l2'"):
        get_collection()


def test_get_collection_rejects_a_collection_built_with_another_model(mocker):
    mock_chroma(mocker, embed_model="some-other-model")

    with pytest.raises(CollectionMismatchError, match="some-other-model"):
        get_collection()


def test_get_collection_rejects_a_legacy_collection_without_model_metadata(mocker):
    _, _, collection = mock_chroma(mocker)
    collection.metadata = None

    with pytest.raises(CollectionMismatchError, match="'None'"):
        get_collection()


# store_index

def test_store_index_raises_on_empty_chunks():
    with pytest.raises(ValueError):
        store_index([])


def test_store_index_embeds_and_stores_chunks(mocker):
    chunks = [
        make_chunk("c1", "first chunk", source="a.pdf", page_start=1, page_end=1, token_count=5),
        make_chunk("c2", "second chunk", source="a.pdf", page_start=2, page_end=2, token_count=7, metadata={"hard_split": True}),
    ]

    mock_embed = mocker.patch("src.retrieval_arena.index.index.embed_texts", return_value=[[0.1, 0.2], [0.3, 0.4]])
    _, mock_client, mock_collection = mock_chroma(mocker)

    result = store_index(chunks)

    assert result == 2
    mock_embed.assert_called_once_with(["first chunk", "second chunk"])
    mock_client.get_or_create_collection.assert_called_once()
    mock_collection.delete.assert_called_once_with(where={"source": {"$in": ["a.pdf"]}})
    mock_collection.add.assert_called_once_with(
        ids=["c1", "c2"],
        documents=["first chunk", "second chunk"],
        embeddings=[[0.1, 0.2], [0.3, 0.4]],
        metadatas=[
            {"source": "a.pdf", "page_start": 1, "page_end": 1, "token_count": 5},
            {"source": "a.pdf", "page_start": 2, "page_end": 2, "token_count": 7, "hard_split": True},
        ],
    )


def test_store_index_uses_a_given_collection_name_instead_of_the_default(mocker):
    chunks = [make_chunk("c1", "first chunk")]

    mocker.patch("src.retrieval_arena.index.index.embed_texts", return_value=[[0.1]])
    _, mock_client, _ = mock_chroma(mocker)

    store_index(chunks, collection_name="custom_collection")

    assert mock_client.get_or_create_collection.call_args.kwargs["name"] == "custom_collection"


def test_store_index_deletes_only_the_changed_sources(mocker):
    chunks = [
        make_chunk("c1", "chunk from a", source="a.pdf"),
        make_chunk("c2", "chunk from b", source="b.pdf"),
    ]

    mocker.patch("src.retrieval_arena.index.index.embed_texts", return_value=[[0.1], [0.2]])
    _, _, mock_collection = mock_chroma(mocker)

    store_index(chunks)

    mock_collection.delete.assert_called_once_with(where={"source": {"$in": ["a.pdf", "b.pdf"]}})


# build_index (orchestrates Process_data -> chunk_data -> store_index)

def test_build_index_orchestrates_the_pipeline_in_order(tmp_path, mocker):
    fake_changed_docs = [["sentinel-doc-group"]]
    fake_chunks = [make_chunk("c1", "text one")]

    mocker.patch("src.retrieval_arena.index.index.get_collection")
    mock_process = mocker.patch("src.retrieval_arena.index.index.Process_data", return_value=fake_changed_docs)
    mock_chunk_data = mocker.patch("src.retrieval_arena.index.index.chunk_data", return_value=fake_chunks)
    mock_store = mocker.patch("src.retrieval_arena.index.index.store_index", return_value=1)

    result = build_index(tmp_path)

    mock_process.assert_called_once_with(tmp_path)
    mock_chunk_data.assert_called_once_with(fake_changed_docs)
    mock_store.assert_called_once_with(chunks=fake_chunks)
    assert result == 1


def test_build_index_skips_when_nothing_changed(tmp_path, mocker):
    mocker.patch("src.retrieval_arena.index.index.get_collection")
    mock_process = mocker.patch("src.retrieval_arena.index.index.Process_data", return_value=None)
    mock_chunk_data = mocker.patch("src.retrieval_arena.index.index.chunk_data")
    mock_store = mocker.patch("src.retrieval_arena.index.index.store_index")

    result = build_index(tmp_path)

    mock_process.assert_called_once_with(tmp_path)
    mock_chunk_data.assert_not_called()
    mock_store.assert_not_called()
    assert result is None


def test_build_index_skips_storing_when_no_chunks_produced(tmp_path, mocker):
    mocker.patch("src.retrieval_arena.index.index.get_collection")
    mocker.patch("src.retrieval_arena.index.index.Process_data", return_value=[["sentinel-doc-group"]])
    mocker.patch("src.retrieval_arena.index.index.chunk_data", return_value=[])
    mock_store = mocker.patch("src.retrieval_arena.index.index.store_index")

    result = build_index(tmp_path)

    mock_store.assert_not_called()
    assert result is None


def test_build_index_fails_before_touching_the_manifest_when_the_collection_is_incompatible(tmp_path, mocker):
    mocker.patch("src.retrieval_arena.index.index.get_collection", side_effect=CollectionMismatchError("stale"))
    mock_process = mocker.patch("src.retrieval_arena.index.index.Process_data")

    with pytest.raises(CollectionMismatchError):
        build_index(tmp_path)

    # Process_data is what writes the manifest; running it first would make the retry skip the rebuild
    mock_process.assert_not_called()
