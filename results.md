# PubMedQA Retrieval Arena — Results

This document explains what the PubMedQA benchmark found when comparing the three retrieval methods (BM25, vector similarity, and hybrid RRF) against each other, and why the results aren't what the usual "hybrid search wins" rule of thumb would predict.

## What was tested

[PubMedQA](https://huggingface.co/datasets/qiaojin/PubMedQA) (`pqa_labeled` split) provides 1,000 real biomedical research questions, each paired with the one specific abstract it was written from. That structure makes it a clean retrieval benchmark: for every question there's exactly one correct document, so "did retrieval find the right one, and how quickly?" has an unambiguous answer.

For each of the 1,000 questions, all three retrieval methods were run against the same indexed corpus, and scored with two metrics:

- **Recall@k** — did the correct document appear anywhere in the top *k* results? (yes/no, since there's only one correct document per question)
- **MRR (Mean Reciprocal Rank)** — averaged over all questions, `1 / rank` of the correct document (1.0 if it's the very first result, 0.5 if second, 0.2 if fifth, 0 if not found at all). Unlike recall, this is sensitive to *how far down* the correct answer was, not just whether it showed up.

*(A third metric, Precision@k, was also computed per the project owner's request, but for this dataset it turns out to be mathematically redundant with Recall@k — see the note in Finding 3.)*

## Headline results

| Method | Recall@1 | Recall@5 | MRR |
|---|---|---|---|
| BM25 (keyword search) | 88.6% | 94.4% | 0.913 |
| Vector search (semantic) | 97.2% | 98.9% | 0.980 |
| Hybrid (BM25 + Vector, fused) | 94.6% | 98.9% | 0.963 |

**Vector search alone wins outright, on every metric.** That's the surprising part — combining two methods is usually expected to beat either one alone, and it doesn't here.

## Finding 1: why "hybrid should win" doesn't hold on this dataset

Hybrid search uses a technique called Reciprocal Rank Fusion (RRF): it runs both BM25 and vector search, then combines their two rankings into one, giving more weight to documents that rank highly in *either* list. This normally helps because the two methods tend to have different weaknesses — BM25 is good at exact keyword/terminology matches that vector search can miss, and vector search understands meaning and phrasing that BM25 can't.

That trade-off assumes both methods are roughly comparable in quality. On PubMedQA, they aren't: vector search is already very close to the ceiling (97-99% accurate) on its own, because the task — matching a paraphrased question back to the one abstract it came from — is exactly the kind of semantic matching vector search is built for. BM25 is noticeably weaker here (88.6% at rank 1), not because it's a bad algorithm, but because these questions often share little exact vocabulary with the abstract they're based on.

When you fuse a strong method with a much weaker one, RRF has no way to know it should trust the strong one more — it just adds up votes from both. That can actively hurt the strong method: if a query's real answer is confidently ranked #1 by vector search but is completely invisible to BM25 (not in BM25's results at all), it gets zero help from BM25. Meanwhile, some *other, wrong* document that both methods rank moderately can accumulate more combined votes and win the fusion instead.

Tracing this directly: hybrid *fixed* 12 questions that vector search got wrong, but *broke* 38 questions vector search had gotten right — a net loss. Of those 38, 25 (66%) were cases where BM25 didn't just rank the real answer poorly, it never found it at all anywhere in its own results.

## Finding 2: same Recall@5, very different failure patterns

Interestingly, Vector and Hybrid *tie* on Recall@5 — both miss the correct document only 11 times out of 1,000. Looking at exactly *how* each method's 11 misses happen tells a more complete story:

| Where the answer landed (when not in 1st place) | BM25 | Vector | Hybrid |
|---|---|---|---|
| 2nd place | 40 | 11 | 20 |
| 3rd-5th place | 18 | 6 | 23 |
| 6th-10th place | 13 | 5 | 4 |
| Not found at all | 43 | 6 | 7 |

Vector search's mistakes are mostly clean losses — when it's wrong, the answer is usually nowhere in its results at all (6 of its 11 recall failures). Hybrid's mistakes are the opposite shape: very few total losses, but many more near-misses at 2nd-5th place (43 combined, vs. Vector's 17). That matches Finding 1 exactly — hybrid isn't usually *losing* the answer, it's *demoting* an answer vector search would have ranked 1st down to 2nd-5th place instead. Recall@5 doesn't penalize that (the answer's still in the top 5, so it still counts as a "hit"), but MRR does — which is exactly why Hybrid's MRR (0.963) is worse than Vector's (0.980) even though their Recall@5 numbers are identical.

BM25 is the clear outlier throughout this table too — it accounts for the large majority of every failure category, especially total losses (43 vs. 6-7 for the other two), reinforcing that its weakness here is fundamental to the method, not a side effect of fusion.

## Finding 3: a note on Precision@5

Precision@5 was computed and recorded (`precision.csv`) alongside recall and MRR, but for this specific dataset it carries no information beyond what Recall@5 already shows. Because every question has exactly one correct document, Recall@5 can only ever be 0 (missed) or 1 (found), and Precision@5 is always exactly that value divided by 5 (0 or 0.2) — a fixed rescaling, not an independent measurement. It's included in the results for completeness, but the real signal in this analysis comes from Recall@5 and MRR (and, as shown above, the *rank* MRR is built from).

## Where to look

- `Data/pubmedqa/results/recall_mrr.png` — the headline comparison chart (the table above, visualized).
- `Data/pubmedqa/results/recall.csv`, `precision.csv`, `mrr.csv` — the raw per-question, per-method scores behind every number in this document.
- `Data/pubmedqa/results/error_analysis.png` — the error-pattern chart behind Finding 2.
- `benchmarks/pubmedqa_arena.py` — the code that produced the headline results.
- `view_results.py` — the code that produced the error-pattern chart.
