import argparse
import csv
import json
import logging
import statistics
from pathlib import Path

import matplotlib.pyplot as plt

from benchmarks.pubmedqa_dataset import PUBMEDQA_DIR, get_pubmedqa_data
from benchmarks.pubmedqa_metrics import precision_at_k, recall_at_k, reciprocal_rank
from src.retrieval_arena.config import Config
from src.retrieval_arena.ingestion.chunk import chunk_all, save_chunks_json
from src.retrieval_arena.ingestion.document import Document
from src.retrieval_arena.index.index import get_collection, store_index
from src.retrieval_arena.retrieval.retrieve import bm25_search, cosign_simularity, search_hybrid_rrf

logger = logging.getLogger(__name__)

COLLECTION_NAME = "pubmedqa_chunks"
# PubMedQA abstract sections measured ~100 tokens each; a smaller chunk_size
# than the PDF pipeline's default (400) keeps chunks close to one section
# instead of blending Background/Methods/Results together - verified against
# real context text before this was picked, not tuned against eval results
CHUNK_SIZE = 120
CHUNK_OVERLAP = 20
EVAL_TOP_K = 10  # how many results to fetch per query; must cover max(K_VALUES)
K_VALUES = (1, 3, 5, 10)
PER_QUERY_K = 5  # depth used for the per-query recall/precision CSV export
METHODS = ("bm25", "vector", "hybrid")
RESULTS_DIR = PUBMEDQA_DIR / "results"

# validated categorical palette, fixed order - passes CVD checks for
# all-pairs comparison at 3 series (see dataviz skill reference palette)
METHOD_COLORS = {"bm25": "#2a78d6", "vector": "#eb6834", "hybrid": "#1baf7a"}
METHOD_LABELS = {"bm25": "BM25", "vector": "Vector", "hybrid": "Hybrid RRF"}


def build_pubmedqa_index(corpus: dict[str, str], chunks_dir: Path, collection_name: str, force: bool = False) -> Path:
    chunks_path = chunks_dir / "chunks.json"
    if not force and chunks_path.exists():
        logger.info(f"Using existing PubMedQA chunks at {chunks_path}")
        return chunks_path

    documents = [
        [Document(id=f"{pubid}_p1", content=text, source=pubid, page_num=1, metadata={})]
        for pubid, text in corpus.items()
    ]
    grouped_chunks = chunk_all(documents, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
    chunks = [c for group in grouped_chunks for c in group]

    save_chunks_json(chunks, chunks_dir)
    store_index(chunks, collection_name=collection_name)
    logger.info(f"Indexed {len(chunks)} chunks from {len(corpus)} documents into '{collection_name}'")
    return chunks_path


def _run_method(method: str, question: str, top_k: int, chunks_path: Path, collection) -> list[dict]:
    if method == "bm25":
        return bm25_search(question, path=chunks_path, top_k=top_k)
    if method == "vector":
        return cosign_simularity(question, top_k=top_k, collection=collection)
    if method == "hybrid":
        return search_hybrid_rrf(question, top_k=top_k, bm25_path=chunks_path, collection=collection)
    raise ValueError(f"Unknown method {method!r}")


def evaluate_method( method: str, queries: dict[str, str], qrels: dict[str, list[str]], chunks_path: Path,collection, ) -> dict:
    recall_samples: dict[int, list[float]] = {k: [] for k in K_VALUES}
    rr_samples: list[float] = []
    per_query: list[dict] = []

    for qid, question in queries.items():
        results = _run_method(method, question, top_k=EVAL_TOP_K, chunks_path=chunks_path, collection=collection)
        ranked_sources = [r["source"] for r in results]
        relevant = qrels[qid]

        for k in K_VALUES:
            recall_samples[k].append(recall_at_k(ranked_sources, relevant, k))
        rr = reciprocal_rank(ranked_sources, relevant)
        rr_samples.append(rr)

        per_query.append({
            "question_id": qid,
            "recall": recall_at_k(ranked_sources, relevant, PER_QUERY_K),
            "precision": precision_at_k(ranked_sources, relevant, PER_QUERY_K),
            "reciprocal_rank": rr,
        })

    return {
        "recall": {
            k: {"mean": statistics.mean(v), "stdev": statistics.stdev(v) if len(v) > 1 else 0.0}
            for k, v in recall_samples.items()
        },
        "mrr": {
            "mean": statistics.mean(rr_samples),
            "stdev": statistics.stdev(rr_samples) if len(rr_samples) > 1 else 0.0,
        },
        "per_query": per_query,
    }


def save_per_query_csvs(per_query_by_method: dict[str, list[dict]], output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    metric_to_filename = {"recall": "recall.csv", "precision": "precision.csv", "reciprocal_rank": "mrr.csv"}

    # every method evaluates the same query set, in the same order
    question_ids = [row["question_id"] for row in per_query_by_method[METHODS[0]]]
    value_by_method_and_qid = {
        method: {row["question_id"]: row for row in rows} for method, rows in per_query_by_method.items()
    }

    paths = {}
    for metric, filename in metric_to_filename.items():
        output_path = output_dir / filename
        with open(output_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["question_id", *METHODS])
            for qid in question_ids:
                writer.writerow([qid, *(value_by_method_and_qid[method][qid][metric] for method in METHODS)])
        paths[metric] = output_path

    return paths


def save_results(results: dict, n_queries: int, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "n_queries": n_queries,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "eval_top_k": EVAL_TOP_K,
        "results": results,
    }
    output_path = output_dir / "results.json"
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    return output_path


def make_figure(results: dict, n_queries: int, output_path: Path) -> None:
    fig, (ax_recall, ax_mrr) = plt.subplots(1, 2, figsize=(11, 4.5), width_ratios=[2.2, 1])

    bar_width = 0.25
    x_positions = list(range(len(K_VALUES)))
    for i, method in enumerate(METHODS):
        means = [results[method]["recall"][k]["mean"] for k in K_VALUES]
        offsets = [xi + (i - 1) * bar_width for xi in x_positions]
        ax_recall.bar(offsets, means, width=bar_width, color=METHOD_COLORS[method], label=METHOD_LABELS[method])

    ax_recall.set_xticks(x_positions)
    ax_recall.set_xticklabels([f"k={k}" for k in K_VALUES])
    ax_recall.set_ylim(0, 1.0)
    ax_recall.set_ylabel("Recall@k")
    ax_recall.set_title("Recall@k by retrieval method")
    ax_recall.yaxis.grid(True, linewidth=0.8, color="#d8d8d5")
    ax_recall.set_axisbelow(True)
    for side in ("top", "right"):
        ax_recall.spines[side].set_visible(False)
    ax_recall.legend(frameon=False, loc="upper left")

    mrr_means = [results[m]["mrr"]["mean"] for m in METHODS]
    bars = ax_mrr.bar(
        [METHOD_LABELS[m] for m in METHODS],
        mrr_means,
        color=[METHOD_COLORS[m] for m in METHODS],
    )
    # headroom above 1.0 so a perfect-score label never collides with the title
    ax_mrr.set_ylim(0, 1.15)
    ax_mrr.set_ylabel("MRR")
    ax_mrr.set_title("Mean Reciprocal Rank")
    ax_mrr.yaxis.grid(True, linewidth=0.8, color="#d8d8d5")
    ax_mrr.set_axisbelow(True)
    for side in ("top", "right"):
        ax_mrr.spines[side].set_visible(False)
    for bar, value in zip(bars, mrr_means):
        ax_mrr.text(
            bar.get_x() + bar.get_width() / 2, value + 0.02, f"{value:.3f}",
            ha="center", fontsize=9, color="#0b0b0b",
        )

    fig.suptitle(f"PubMedQA retrieval arena — {n_queries} queries", fontsize=13, fontweight="bold")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, facecolor="white")
    plt.close(fig)


def run_arena(sample_size: int | None = None) -> dict:
    corpus, queries, qrels = get_pubmedqa_data()

    if sample_size is not None:
        sample_ids = list(queries)[:sample_size]
        corpus = {pubid: corpus[pubid] for pubid in sample_ids}
        queries = {pubid: queries[pubid] for pubid in sample_ids}
        qrels = {pubid: qrels[pubid] for pubid in sample_ids}
        # sampled runs are for quick iteration - kept fully separate from the
        # real corpus/collection so a small test run can never shadow the
        # cached full-corpus artifact
        chunks_dir = PUBMEDQA_DIR / "sample"
        collection_name = f"{COLLECTION_NAME}_sample"
    else:
        chunks_dir = PUBMEDQA_DIR
        collection_name = COLLECTION_NAME

    chunks_path = build_pubmedqa_index(corpus, chunks_dir, collection_name, force=(sample_size is not None))
    collection = get_collection(collection_name)

    results = {}
    per_query_by_method = {}
    for method in METHODS:
        logger.info(f"Evaluating {method} over {len(queries)} queries...")
        method_result = evaluate_method(method, queries, qrels, chunks_path, collection)
        per_query_by_method[method] = method_result.pop("per_query")
        results[method] = method_result
        recall5 = results[method]["recall"][5]["mean"]
        mrr = results[method]["mrr"]["mean"]
        logger.info(f"{method}: Recall@5={recall5:.3f}  MRR={mrr:.3f}")

    results_path = save_results(results, len(queries), RESULTS_DIR)
    figure_path = RESULTS_DIR / "recall_mrr.png"
    make_figure(results, len(queries), figure_path)
    csv_paths = save_per_query_csvs(per_query_by_method, RESULTS_DIR)
    logger.info(f"Saved results to {results_path}, figure to {figure_path}")
    logger.info(f"Saved per-query CSVs: {', '.join(str(p) for p in csv_paths.values())}")

    return results


if __name__ == "__main__":
    Config.configure_logging()
    parser = argparse.ArgumentParser(description="Run the PubMedQA retrieval arena.")
    parser.add_argument(
        "--sample-size", type=int, default=None,
        help="Evaluate on a small subset instead of the full 1,000 queries (for quick iteration).",
    )
    args = parser.parse_args()
    run_arena(sample_size=args.sample_size)
