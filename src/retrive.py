try:
    from .Index import get_collection
    from .Chunk import Chunk
    from .Embed import embed_texts
except ImportError:
    from Index import get_collection
    from Chunk import Chunk
    from Embed import embed_texts
import logging

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

    

    
def bm25_search()->list[Chunk]:
    pass

def search_hybrid_rrf()->list[Chunk]:
    pass


def main()->None:
    pass


if __name__ == "__main__":
    main()