from Config import Config
from Embed import embed_texts
from Chunk import Chunk
from pathlib import Path
import chromadb
import logging
import json

logger = logging.getLogger(__name__)


# In this file we need to get the chunked data , embed them and store inside the vectordb

# we need two pimar functions
# 1 => loads the data and returns an array of chunks 
# 2 => take the array , extract the text and store it in a seperate array , call the embedding function 
# store the data + the embeddings in the data base 


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


def build_index(path:Path)-> int :
    chunks : list[Chunk] = load_data(path)

    if not chunks:
        raise ValueError("No chunks founded")

    texts = [c.text for c in chunks]

    embeddings = embed_texts(texts)

    client = chromadb.PersistentClient(path=str(Config.VECTORDB_DIR))
    try:
        client.delete_collection(Config.COLLECTION_NAME)
    except Exception:
        pass
    collection = client.get_or_create_collection(name=Config.COLLECTION_NAME)


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


if __name__ == "__main__":
    Config.configure_logging()
    path = Config.chunks_dir / "chunks.json"
    chunks_number = build_index(path=path)
    logger.info(f"{chunks_number} have been stored in the database")


