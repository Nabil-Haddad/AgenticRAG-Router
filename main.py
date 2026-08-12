import logging
import sys

from src.retrieval_arena.config import Config
from src.retrieval_arena.retrieval.retrieve import retrieve

logger = logging.getLogger(__name__)


def retrieve_chunks(query: str, method: str | None = None, top_k: int = 5) -> list[dict]:
    return retrieve(query, method=method, top_k=top_k)


def main() -> None:
    query = " ".join(sys.argv[1:])
    if not query:
        logger.error("Usage: python main.py <query>")
        return

    for result in retrieve_chunks(query):
        print(f"{result['idx']}  {result['score']:.4f}  {result['text'][:80]}")


if __name__ == "__main__":
    Config.configure_logging()
    main()
