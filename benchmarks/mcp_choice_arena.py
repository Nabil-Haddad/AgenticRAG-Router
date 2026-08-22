import argparse
import asyncio
import json
import logging
import statistics
import sys
from collections import Counter
from pathlib import Path

import anthropic
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import get_default_environment, stdio_client

from benchmarks.pubmedqa_arena import COLLECTION_NAME, build_pubmedqa_index
from benchmarks.pubmedqa_dataset import PUBMEDQA_DIR, get_pubmedqa_data
from benchmarks.pubmedqa_metrics import recall_at_k, reciprocal_rank
from src.retrieval_arena.config import Config
from src.retrieval_arena.index.index import get_collection
from src.retrieval_arena.mcp_server.client import SERVER_MODULE, run_conversation
from src.retrieval_arena.retrieval.retrieve import cosign_simularity

logger = logging.getLogger(__name__)

DEFAULT_SAMPLE_SIZE = 30  # real, billed Anthropic API calls - keep small by default
EVAL_TOP_K = 5
RESULTS_DIR = PUBMEDQA_DIR / "mcp_choice"

TOOL_NAMES = ("bm25_search", "vector_search", "hybrid_search")
# same categorical palette used throughout the rest of the arena, keyed by
# MCP tool name instead of pubmedqa_arena's short method name
TOOL_COLORS = {"bm25_search": "#2a78d6", "vector_search": "#eb6834", "hybrid_search": "#1baf7a"}
TOOL_LABELS = {"bm25_search": "BM25", "vector_search": "Vector", "hybrid_search": "Hybrid RRF"}


async def run_choice_sample(
    queries: dict[str, str],
    qrels: dict[str, list[str]],
    chunks_path: Path,
    collection,
    model: str,
    top_k: int,
    output_dir: Path,
) -> list[dict]:
    anthropic_client = anthropic.Anthropic(api_key=Config.ANTHROPIC_API_KEY)
    env = {
        **get_default_environment(),
        "RETRIEVAL_ARENA_MCP_COLLECTION": COLLECTION_NAME,
        "RETRIEVAL_ARENA_MCP_CHUNKS_PATH": str(chunks_path),
    }
    params = StdioServerParameters(command=sys.executable, args=["-m", SERVER_MODULE], cwd=str(Config.project_root), env=env)

    records = []
    # one subprocess/session reused across every query
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            for i, (qid, question) in enumerate(queries.items(), start=1):
                logger.info(f"[{i}/{len(queries)}] Asking Claude: {question[:80]}...")
                claude_result = await run_conversation(session, anthropic_client, question, model, top_k)
                vector_results = cosign_simularity(question, top_k=top_k, collection=collection)

                claude_sources = [r["source"] for r in claude_result.retrieved_results]
                vector_sources = [r["source"] for r in vector_results]
                relevant = qrels[qid]

                logger.info(f"  chose {claude_result.chosen_method}")
                records.append({
                    "question_id": qid,
                    "chosen_method": claude_result.chosen_method,
                    "claude_recall": recall_at_k(claude_sources, relevant, top_k),
                    "claude_reciprocal_rank": reciprocal_rank(claude_sources, relevant),
                    "vector_recall": recall_at_k(vector_sources, relevant, top_k),
                    "vector_reciprocal_rank": reciprocal_rank(vector_sources, relevant),
                })
                # saved after every query, not just at the end - each one is a
                # real billed API call, so an interrupted or failed run should
                # never lose progress that already cost money
                save_results(records, summarize(records), output_dir)

    return records


def summarize(records: list[dict]) -> dict:
    method_counts = Counter(r["chosen_method"] for r in records)
    return {
        "n_queries": len(records),
        "method_distribution": {tool: method_counts.get(tool, 0) for tool in TOOL_NAMES},
        "claude": {
            "recall_mean": statistics.mean(r["claude_recall"] for r in records),
            "mrr_mean": statistics.mean(r["claude_reciprocal_rank"] for r in records),
        },
        "vector_baseline": {
            "recall_mean": statistics.mean(r["vector_recall"] for r in records),
            "mrr_mean": statistics.mean(r["vector_reciprocal_rank"] for r in records),
        },
    }


def save_results(records: list[dict], summary: dict, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "results.json"
    payload = {"eval_top_k": EVAL_TOP_K, "summary": summary, "per_query": records}
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    return output_path


def make_figure(summary: dict, output_path: Path) -> None:
    fig, (ax_choice, ax_compare) = plt.subplots(1, 2, figsize=(11, 4.5), width_ratios=[1.2, 1])

    choice_counts = [summary["method_distribution"][tool] for tool in TOOL_NAMES]
    bars = ax_choice.bar(
        [TOOL_LABELS[t] for t in TOOL_NAMES], choice_counts, color=[TOOL_COLORS[t] for t in TOOL_NAMES],
    )
    ax_choice.set_ylabel("Number of queries")
    ax_choice.set_title("Which method Claude chose")
    ax_choice.set_ylim(0, max(choice_counts) * 1.25 if max(choice_counts) else 1)
    ax_choice.yaxis.grid(True, linewidth=0.8, color="#d8d8d5")
    ax_choice.set_axisbelow(True)
    for side in ("top", "right"):
        ax_choice.spines[side].set_visible(False)
    for bar, value in zip(bars, choice_counts):
        ax_choice.text(
            bar.get_x() + bar.get_width() / 2, value + max(choice_counts) * 0.03, str(value),
            ha="center", fontsize=9, color="#0b0b0b",
        )

    categories = ["Recall@5", "MRR"]
    claude_color = "#7a5cff"
    claude_values = [summary["claude"]["recall_mean"], summary["claude"]["mrr_mean"]]
    baseline_values = [summary["vector_baseline"]["recall_mean"], summary["vector_baseline"]["mrr_mean"]]
    bar_width = 0.3
    x_positions = list(range(len(categories)))
    ax_compare.bar([x - bar_width / 2 for x in x_positions], claude_values, width=bar_width, color=claude_color)
    ax_compare.bar(
        [x + bar_width / 2 for x in x_positions], baseline_values, width=bar_width,
        color=TOOL_COLORS["vector_search"],
    )
    ax_compare.set_xticks(x_positions)
    ax_compare.set_xticklabels(categories)
    ax_compare.set_ylim(0, 1.15)
    ax_compare.set_title("Claude's choice vs. a vector-only baseline")
    ax_compare.yaxis.grid(True, linewidth=0.8, color="#d8d8d5")
    ax_compare.set_axisbelow(True)
    for side in ("top", "right"):
        ax_compare.spines[side].set_visible(False)

    # one legend for the whole figure, placed above both panels - a
    # per-axes legend collided with this panel's own title
    handles = [Patch(facecolor=claude_color, label="Claude's choice"), Patch(facecolor=TOOL_COLORS["vector_search"], label="Vector only")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False)

    fig.suptitle(f"MCP retrieval-choice arena — {summary['n_queries']} queries", fontsize=13, fontweight="bold", y=1.1)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def run_mcp_choice_arena(sample_size: int = DEFAULT_SAMPLE_SIZE, model: str = Config.ANTHROPIC_MODEL) -> dict:
    if not Config.ANTHROPIC_API_KEY:
        raise SystemExit("ANTHROPIC_API_KEY is not set — this harness calls the real Anthropic API.")

    corpus, queries, qrels = get_pubmedqa_data()

    # sample only the queries sent to the API - the corpus/index always stays
    # the full 1,000 documents, otherwise this becomes "needle in a haystack
    # of `sample_size`" instead of the real 1,000-document retrieval task,
    # and can't be compared against the full-corpus numbers in results.md
    sample_ids = list(queries)[:sample_size]
    sampled_queries = {qid: queries[qid] for qid in sample_ids}

    chunks_path = build_pubmedqa_index(corpus, PUBMEDQA_DIR, COLLECTION_NAME, force=False)
    collection = get_collection(COLLECTION_NAME)

    logger.info(f"Running {len(sampled_queries)} queries through Claude + MCP...")
    try:
        records = asyncio.run(
            run_choice_sample(sampled_queries, qrels, chunks_path, collection, model, EVAL_TOP_K, RESULTS_DIR)
        )
    except BaseException:
        # every query already saved its own progress via run_choice_sample,
        # so an interrupted/failed run still has real (partial) results on
        # disk - rebuild the figure from them before re-raising, rather than
        # leaving a results.json with no matching chart
        partial_path = RESULTS_DIR / "results.json"
        if partial_path.exists():
            with open(partial_path, "r", encoding="utf-8") as fh:
                partial_records = json.load(fh)["per_query"]
            if partial_records:
                logger.warning(f"Run did not finish - plotting the {len(partial_records)} queries completed so far.")
                make_figure(summarize(partial_records), RESULTS_DIR / "method_choice.png")
        raise

    summary = summarize(records)
    logger.info(f"Method distribution: {summary['method_distribution']}")
    logger.info(f"Claude: Recall@{EVAL_TOP_K}={summary['claude']['recall_mean']:.3f}  MRR={summary['claude']['mrr_mean']:.3f}")
    logger.info(f"Vector: Recall@{EVAL_TOP_K}={summary['vector_baseline']['recall_mean']:.3f}  MRR={summary['vector_baseline']['mrr_mean']:.3f}")

    results_path = save_results(records, summary, RESULTS_DIR)
    figure_path = RESULTS_DIR / "method_choice.png"
    make_figure(summary, figure_path)
    logger.info(f"Saved results to {results_path} and figure to {figure_path}")

    return summary


if __name__ == "__main__":
    Config.configure_logging()
    parser = argparse.ArgumentParser(description="Test whether Claude picks one retrieval method or varies its choice per question.")
    parser.add_argument(
        "--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE,
        help=f"Number of PubMedQA questions to send through the real API (default {DEFAULT_SAMPLE_SIZE}).",
    )
    args = parser.parse_args()
    run_mcp_choice_arena(sample_size=args.sample_size)
