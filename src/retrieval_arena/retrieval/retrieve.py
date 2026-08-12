from ..index.index import get_collection
from ..ingestion.chunk import Chunk, load_chunks
from ..ingestion.embed import embed_texts
from ..config import Config
import logging
import sys
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

    # same {idx, text, score} shape as cosign_simularity
    return [
        {"idx": chunks[i].id, "text": chunks[i].text, "score": float(scores[i])}
        for i in top_indices
    ]

def search_hybrid_rrf(query: str, top_k: int = 5, candidate_k: int = 20, k: int = 60) -> list[dict]:
    # pull a wider candidate pool from each method than top_k, so a chunk
    # that's strong in one ranking but just outside the other's top_k still
    # gets a fair shot at surviving the fusion
    bm25_results = bm25_search(query, top_k=candidate_k)
    vector_results = cosign_simularity(query, top_k=candidate_k)

    rrf_scores: dict[str, float] = {}
    texts: dict[str, str] = {}

    for ranking in (bm25_results, vector_results):
        for rank, result in enumerate(ranking, start=1):
            rrf_scores[result["idx"]] = rrf_scores.get(result["idx"], 0.0) + 1 / (k + rank)
            texts[result["idx"]] = result["text"]

    ranked = sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True)[:top_k]

    return [{"idx": idx, "text": texts[idx], "score": score} for idx, score in ranked]


# single entry point for callers outside this module 
_RETRIEVAL_METHODS = {
    "hybrid": search_hybrid_rrf,
    "bm25": bm25_search,
    "vector": cosign_simularity,
}


def retrieve(query: str, method: str | None = None, top_k: int = 5, **kwargs) -> list[dict]:
    if method is None:
        method = Config.RETRIEVAL_METHOD

    search_fn = _RETRIEVAL_METHODS.get(method)
    if search_fn is None:
        raise ValueError(f"Unknown retrieval method {method!r}, choose from {sorted(_RETRIEVAL_METHODS)}")

    return search_fn(query, top_k=top_k, **kwargs)


if __name__ == "__main__":
    Config.configure_logging()
    demo_query = " ".join(sys.argv[1:]) or "what is retrieval augmented generation"
    for result in retrieve(demo_query):
        print(f"{result['idx']}  {result['score']:.4f}  {result['text'][:80]}")