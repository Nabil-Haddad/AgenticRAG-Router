try:
    from .Config import Config
    from .Embed import embed_texts
    from .Process import Process_data
    from .Chunk import Chunk, chunk_data
except ImportError:
    from Config import Config
    from Embed import embed_texts
    from Process import Process_data
    from Chunk import Chunk, chunk_data
from pathlib import Path
import chromadb
import logging
import json

logger = logging.getLogger(__name__)



# Create a collection 
def get_collection():
    client = chromadb.PersistentClient(path=str(Config.VECTORDB_DIR))
    return client.get_or_create_collection(name=Config.COLLECTION_NAME)

# load the data 
def load_data(path:Path)->list[Chunk]:
    with open(path, 'r', encoding="utf-8")as fh:
        data = json.load(fh)

    docs = [Chunk(**d)for d in data]
    return docs


def store_index(chunks : list[Chunk])-> int :
    if not chunks:
        raise ValueError("No chunks founded")

    texts = [c.text for c in chunks]

    embeddings = embed_texts(texts)

    client = chromadb.PersistentClient(path=str(Config.VECTORDB_DIR))
    collection = client.get_or_create_collection(name=Config.COLLECTION_NAME)

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



