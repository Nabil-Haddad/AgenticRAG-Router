try:
    from .Index import get_collection
    from .Chunk import Chunk, load_chunks
    from .Embed import embed_texts
    from .Config import Config
except ImportError:
    from Index import get_collection
    from Chunk import Chunk, load_chunks
    from Embed import embed_texts
    from Config import Config
import logging
from pathlib import Path
import numpy as np
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)

# create 3 different function cosign_simularity , mb25_search , search_hypbrid_rrf

def peek_first_5_elements()->None:
    try :
        client = get_collection()
        samples = client.peek(limit=5)
    except Exception as e:
        print("connection failed")
        print(e)
        return

    print("VectorDB Connection Successful\n")
    print(f"Retrieved {len(samples['ids'])} items from the database:\n")
    for i in range(len(samples['ids'])):
        print(f"The sample {i} , with the id {samples['ids'][i]}")
        print(f"With Documents : {samples['documents'][i][:40]}")


def cosign_simularity(query : str, top_k : int = 5 )->list[dict]:
    try :
        collection = get_collection()
        # embed the query with the same pipeline used to embed the corpus,
        # instead of letting Chroma use its own (different) default embedder
        query_embedding = embed_texts([query])
        results = collection.query(
            query_embeddings=query_embedding,
            n_results=top_k
        )
    except Exception as e:
        print("connection failed")
        print(e)
        return []

    return [
        {"idx" : idx, "text" : text, "score" : score  }
        for idx, text, score in zip(results['ids'][0], results['documents'][0], results['distances'][0])
    ]

    
# cache is keyed by path and invalidated by mtime, so a re-run of the
# ingestion pipeline (new/changed chunks.json) is picked up automatically
# instead of silently searching a stale in-memory index
_bm25_cache: dict[Path, tuple[float, BM25Okapi, list[Chunk]]] = {}


def _get_bm25_index(path: Path) -> tuple[BM25Okapi, list[Chunk]] | None:
    if not path.exists():
        return None

    mtime = path.stat().st_mtime
    cached = _bm25_cache.get(path)
    if cached is not None and cached[0] == mtime:
        return cached[1], cached[2]

    chunks = load_chunks(path)
    if not chunks:
        return None

    tokenized_corpus = [c.text.lower().split() for c in chunks]
    bm25 = BM25Okapi(tokenized_corpus)
    _bm25_cache[path] = (mtime, bm25, chunks)
    return bm25, chunks


def bm25_search(query: str, path: Path | None = None, top_k: int = 5) -> list[dict]:
    if path is None:
        path = Config.chunks_dir / "chunks.json"

    try:
        index_data = _get_bm25_index(path)
        if index_data is None:
            return []
        bm25, chunks = index_data

        tokenized_query = query.lower().split()
        scores: np.ndarray = bm25.get_scores(tokenized_query)
        top_indices: np.ndarray = np.argsort(scores)[::-1][:top_k]
    except Exception as e:
        print("bm25 search failed")
        print(e)
        return []

    # same {idx, text, score} shape as cosign_simularity, and idx is the same
    # chunk id used in Chroma - required so RRF can merge rankings by id later
    return [
        {"idx": chunks[i].id, "text": chunks[i].text, "score": float(scores[i])}
        for i in top_indices
    ]

def search_hybrid_rrf()->list:
    pass


def main()->None:
    pass


if __name__ == "__main__":
    main()