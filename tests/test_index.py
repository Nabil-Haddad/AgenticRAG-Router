import json
from pathlib import Path

import pytest

from src.Index import Config, build_index, get_collection, load_data


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


# load_data 

def test_load_data_reads_chunks_from_json(tmp_path):
    path = tmp_path / "chunks.json"
    write_chunks_json(path, [make_chunk_dict("c1", "text one"), make_chunk_dict("c2", "text two")])

    chunks = load_data(path)

    assert len(chunks) == 2
    assert chunks[0].id == "c1"
    assert chunks[0].text == "text one"
    assert chunks[1].id == "c2"


# get_collection 

def test_get_collection_uses_configured_path_and_name(mocker):
    mock_client_cls = mocker.patch("src.Index.chromadb.PersistentClient")
    mock_client = mock_client_cls.return_value

    collection = get_collection()

    mock_client_cls.assert_called_once_with(path=str(Config.VECTORDB_DIR))
    mock_client.get_or_create_collection.assert_called_once_with(name=Config.COLLECTION_NAME)
    assert collection is mock_client.get_or_create_collection.return_value


# build_index 

def test_build_index_raises_on_empty_chunks(tmp_path):
    path = tmp_path / "chunks.json"
    write_chunks_json(path, [])

    with pytest.raises(ValueError):
        build_index(path)


def test_build_index_embeds_and_stores_chunks(tmp_path, mocker):
    path = tmp_path / "chunks.json"
    write_chunks_json(path, [
        make_chunk_dict("c1", "first chunk", source="a.pdf", page_start=1, page_end=1, token_count=5),
        make_chunk_dict("c2", "second chunk", source="a.pdf", page_start=2, page_end=2, token_count=7, metadata={"hard_split": True}),
    ])

    mock_embed = mocker.patch("src.Index.embed_texts", return_value=[[0.1, 0.2], [0.3, 0.4]])
    mock_client_cls = mocker.patch("src.Index.chromadb.PersistentClient")
    mock_client = mock_client_cls.return_value
    mock_collection = mock_client.get_or_create_collection.return_value

    result = build_index(path)

    assert result == 2
    mock_embed.assert_called_once_with(["first chunk", "second chunk"])
    mock_client.delete_collection.assert_called_once_with(Config.COLLECTION_NAME)
    mock_client.get_or_create_collection.assert_called_once_with(name=Config.COLLECTION_NAME)
    mock_collection.add.assert_called_once_with(
        ids=["c1", "c2"],
        documents=["first chunk", "second chunk"],
        embeddings=[[0.1, 0.2], [0.3, 0.4]],
        metadatas=[
            {"source": "a.pdf", "page_start": 1, "page_end": 1, "token_count": 5},
            {"source": "a.pdf", "page_start": 2, "page_end": 2, "token_count": 7, "hard_split": True},
        ],
    )


def test_build_index_swallows_delete_collection_errors(tmp_path, mocker):
    path = tmp_path / "chunks.json"
    write_chunks_json(path, [make_chunk_dict("c1", "only chunk")])

    mocker.patch("src.Index.embed_texts", return_value=[[0.1]])
    mock_client_cls = mocker.patch("src.Index.chromadb.PersistentClient")
    mock_client = mock_client_cls.return_value
    mock_client.delete_collection.side_effect = Exception("collection does not exist")

    result = build_index(path)

    assert result == 1
    mock_client.get_or_create_collection.assert_called_once_with(name=Config.COLLECTION_NAME)
