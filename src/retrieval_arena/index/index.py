from ..config import Config
from ..ingestion.embed import embed_texts
from ..ingestion.process import Process_data
from ..ingestion.chunk import Chunk, chunk_data
from pathlib import Path
import chromadb
import logging
import json

logger = logging.getLogger(__name__)



DISTANCE_SPACE = "cosine"


class CollectionMismatchError(RuntimeError):
    pass


# Create a collection
def get_collection(name: str | None = None):
    if name is None:
        name = Config.COLLECTION_NAME
    client = chromadb.PersistentClient(path=str(Config.VECTORDB_DIR))
    collection = client.get_or_create_collection(
        name=name,
        configuration={"hnsw": {"space": DISTANCE_SPACE}},
        metadata={"embed_model": Config.EMBED_MODEL_NAME},
    )
    _check_collection(collection, name)
    return collection


def _check_collection(collection, name: str) -> None:
    # get_or_create_collection returns an existing collection as-is, ignoring the settings above,
    # so verify it was built with this model and metric instead of silently mixing vectors
    space = collection.configuration["hnsw"]["space"]
    embed_model = (collection.metadata or {}).get("embed_model")
    problems = []
    if space != DISTANCE_SPACE:
        problems.append(f"distance space is '{space}', expected '{DISTANCE_SPACE}'")
    if embed_model != Config.EMBED_MODEL_NAME:
        problems.append(f"built with embedding model '{embed_model}', expected '{Config.EMBED_MODEL_NAME}'")
    if problems:
        raise CollectionMismatchError(
            f"Collection '{name}' is incompatible with the current config ({'; '.join(problems)}). "
            f"Delete it (e.g. remove {Config.VECTORDB_DIR} and {Config.manifest_path}) and re-ingest."
        )

# load the data 
def load_data(path:Path)->list[Chunk]:
    with open(path, 'r', encoding="utf-8")as fh:
        data = json.load(fh)

    docs = [Chunk(**d)for d in data]
    return docs


def store_index(chunks : list[Chunk], collection_name: str | None = None)-> int :
    if not chunks:
        raise ValueError("No chunks founded")

    texts = [c.text for c in chunks]

    embeddings = embed_texts(texts)

    collection = get_collection(collection_name)

    # only these chunks are being (re)written; drop their stale entries
    # first instead of wiping and rebuilding the whole collection
    changed_sources = sorted({c.source for c in chunks})
    collection.delete(where={"source": {"$in": changed_sources}})

    collection.add(
        ids= [c.id for c in chunks],
        documents=texts,
        embeddings=embeddings,
        metadatas=[
            {
                "source": c.source,
                "page_start": c.page_start,
                "page_end": c.page_end,
                "token_count": c.token_count,
                **c.metadata,
            }
            for c in chunks
        ],
    )

    return len(chunks)


def build_index(path: Path)->int | None:
    # fail fast: Process_data writes the manifest, so an incompatible collection
    # must be caught before that or the next run would skip the rebuild
    get_collection()
    changed_documents = Process_data(path)
    if changed_documents is None:
        return None
    chunks = chunk_data(changed_documents)
    if not chunks:
        return None
    return store_index(chunks=chunks)


if __name__ == "__main__":
    Config.configure_logging()
    chunks_number = build_index(path=Config.data_dir)
    if chunks_number is None:
        logger.info("Nothing to index; data unchanged since the last run.")
    else:
        logger.info(f"{chunks_number} have been stored in the database")



