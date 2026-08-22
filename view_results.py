import csv
import logging
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from benchmarks.pubmedqa_arena import METHOD_COLORS, METHOD_LABELS, METHODS, RESULTS_DIR
from src.retrieval_arena.config import Config

logger = logging.getLogger(__name__)

RANK_BUCKETS = ("Rank 1", "Rank 2", "Rank 3-5", "Rank 6-10", "Not found")
# Rank 1 is the non-error case and dwarfs the rest on a shared linear axis
# (it's 90%+ of queries for every method) - excluded here so the panel stays
# readable and focused on what it's actually meant to show: where things
# went wrong, not how often they didn't
ERROR_BUCKETS = tuple(b for b in RANK_BUCKETS if b != "Rank 1")


def _load_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _rank_bucket(reciprocal_rank: float) -> str:
    if reciprocal_rank == 0.0:
        return "Not found"
    rank = round(1 / reciprocal_rank)
    if rank == 1:
        return "Rank 1"
    if rank == 2:
        return "Rank 2"
    if rank <= 5:
        return "Rank 3-5"
    return "Rank 6-10"


def plot_recall_errors(ax, recall_rows: list[dict]) -> None:
    error_counts = [sum(1 for row in recall_rows if float(row[m]) == 0.0) for m in METHODS]
    bars = ax.bar([METHOD_LABELS[m] for m in METHODS], error_counts, color=[METHOD_COLORS[m] for m in METHODS])

    ax.set_ylabel("Queries missed (Recall@5 = 0)")
    ax.set_title("Recall@5 errors")
    ax.set_ylim(0, max(error_counts) * 1.25 if max(error_counts) else 1)
    ax.yaxis.grid(True, linewidth=0.8, color="#d8d8d5")
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for bar, value in zip(bars, error_counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2, value + max(error_counts) * 0.03, str(value),
            ha="center", fontsize=9, color="#0b0b0b",
        )


def plot_precision_means(ax, precision_rows: list[dict]) -> None:
    means = [sum(float(row[m]) for row in precision_rows) / len(precision_rows) for m in METHODS]
    bars = ax.bar([METHOD_LABELS[m] for m in METHODS], means, color=[METHOD_COLORS[m] for m in METHODS])

    ax.set_ylabel("Mean Precision@5")
    ax.set_title("Precision@5\n(proportional to Recall@5 here - one relevant doc per query)", fontsize=10)
    ax.set_ylim(0, max(means) * 1.3 if max(means) else 1)
    ax.yaxis.grid(True, linewidth=0.8, color="#d8d8d5")
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for bar, value in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2, value + max(means) * 0.03, f"{value:.3f}",
            ha="center", fontsize=9, color="#0b0b0b",
        )


def plot_rank_distribution(ax, mrr_rows: list[dict]) -> None:
    counts = {m: {b: 0 for b in RANK_BUCKETS} for m in METHODS}
    for row in mrr_rows:
        for m in METHODS:
            counts[m][_rank_bucket(float(row[m]))] += 1

    bar_width = 0.25
    x_positions = list(range(len(ERROR_BUCKETS)))
    for i, m in enumerate(METHODS):
        values = [counts[m][bucket] for bucket in ERROR_BUCKETS]
        offsets = [xi + (i - 1) * bar_width for xi in x_positions]
        ax.bar(offsets, values, width=bar_width, color=METHOD_COLORS[m], label=METHOD_LABELS[m])

    ax.set_xticks(x_positions)
    ax.set_xticklabels(ERROR_BUCKETS)
    ax.set_ylabel("Number of queries")
    ax.set_title("Where retrieval fell short (excludes Rank 1 hits)")
    ax.yaxis.grid(True, linewidth=0.8, color="#d8d8d5")
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def make_error_analysis_figure(results_dir: Path, output_path: Path) -> None:
    recall_rows = _load_csv(results_dir / "recall.csv")
    precision_rows = _load_csv(results_dir / "precision.csv")
    mrr_rows = _load_csv(results_dir / "mrr.csv")

    fig, (ax_recall, ax_precision, ax_rank) = plt.subplots(1, 3, figsize=(15.5, 4.5), width_ratios=[1, 1, 1.6])

    plot_recall_errors(ax_recall, recall_rows)
    plot_precision_means(ax_precision, precision_rows)
    plot_rank_distribution(ax_rank, mrr_rows)

    # one legend for the whole figure, not one per panel - all three panels
    # already share the same color-to-method mapping, and a per-panel legend
    # placed above its axes would compete with that panel's own title for
    # the same vertical space
    handles = [Patch(facecolor=METHOD_COLORS[m], label=METHOD_LABELS[m]) for m in METHODS]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False)

    fig.suptitle(
        f"PubMedQA retrieval arena — per-query error analysis ({len(recall_rows)} queries)",
        fontsize=13, fontweight="bold", y=1.1,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    output_path = RESULTS_DIR / "error_analysis.png"
    make_error_analysis_figure(RESULTS_DIR, output_path)
    logger.info(f"Saved error analysis figure to {output_path}")
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    Config.configure_logging()
    main()
