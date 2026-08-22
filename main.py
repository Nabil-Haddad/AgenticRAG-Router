import argparse
import asyncio
import logging

import anthropic

from src.retrieval_arena.config import Config
from src.retrieval_arena.mcp_server.client import ask_with_mcp_tools
from src.retrieval_arena.retrieval.retrieve import retrieve

logger = logging.getLogger(__name__)


def retrieve_chunks(query: str, method: str | None = None, top_k: int = 5) -> list[dict]:
    return retrieve(query, method=method, top_k=top_k)


def run_direct(query: str, method: str | None, top_k: int) -> None:
    for result in retrieve_chunks(query, method=method, top_k=top_k):
        print(f"{result['idx']}  {result['score']:.4f}  {result['text'][:80]}")


def _iter_leaf_exceptions(exc: BaseException):
    if isinstance(exc, BaseExceptionGroup):
        for sub_exc in exc.exceptions:
            yield from _iter_leaf_exceptions(sub_exc)
    else:
        yield exc


def run_agent(query: str, top_k: int) -> None:
    if not Config.ANTHROPIC_API_KEY:
        logger.error("ANTHROPIC_API_KEY is not set in .env — --agent calls the real Anthropic API.")
        return

    result = None
    try:
        result = asyncio.run(ask_with_mcp_tools(query, top_k=top_k))
    except* anthropic.APIError as eg:
        for exc in _iter_leaf_exceptions(eg):
            logger.error(f"Anthropic API call failed: {exc}")

    if result is None:
        return

    print(f"Chosen method: {result.chosen_method}")
    print(f"Retrieved {len(result.retrieved_results)} chunks")
    print()
    print(result.claude_final_text)


def main() -> None:
    parser = argparse.ArgumentParser(description="Query the retrieval-arena index.")
    parser.add_argument("query", nargs="+", help="The question to search for.")
    parser.add_argument( "--agent", action="store_true",
        help="Let Claude choose which retrieval method to use, via the MCP server, instead of picking one yourself.",
    )
    parser.add_argument( "--method", default=None, choices=["bm25", "vector", "hybrid"],
        help="Retrieval method to use directly (ignored with --agent). Defaults to Config.RETRIEVAL_METHOD.",
    )
    parser.add_argument("--top-k", type=int, default=5, help="How many results to retrieve.")
    args = parser.parse_args()

    query = " ".join(args.query)
    if args.agent:
        run_agent(query, args.top_k)
    else:
        run_direct(query, args.method, args.top_k)


if __name__ == "__main__":
    Config.configure_logging()
    main()
