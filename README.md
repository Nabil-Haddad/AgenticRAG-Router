# retrieval-arena

A from-scratch RAG (Retrieval-Augmented Generation) pipeline: take raw PDFs, extract and clean their text, split it into token-sized chunks, embed those chunks, index them in a vector database, and retrieve against them with three different strategies (BM25, vector similarity, and a hybrid RRF fusion of the two). The name points at the longer-term goal — comparing retrieval strategies against each other on a real benchmark (PubMedQA), not just building one fixed pipeline.

Everything is built without a framework (no LangChain/LlamaIndex) — each stage is a small, explicit module, so the mechanics of chunking, embedding, indexing, and retrieval are hand-written rather than hidden behind a library call.

## Pipeline overview

```
Data/raw/*.pdf
      │  ingestion/process.py  (load_pdf, Validate, Process_data)
      ▼
Data/processed/*.json      -- one file per PDF, page-level Document objects
      │  ingestion/chunk.py  (tag_sentences, chunk_document, chunk_data)
      ▼
Data/chunks/chunks.json    -- single combined file, all chunks from every PDF
      │  index/index.py  (build_index, via ingestion/embed.py)
      ▼
vectordb/                  -- persistent ChromaDB collection
      │  retrieval/retrieve.py  (bm25_search, cosign_simularity, search_hybrid_rrf)
      ▼
retrieve(query, method="hybrid")  -- ranked {idx, text, score} results
```

Every stage past the first is idempotent: a `manifest.json` tracks a content hash per source file, so re-running the pipeline only re-processes, re-chunks, and re-embeds files that actually changed.

## Project layout

```
src/retrieval_arena/
├── config.py             # central settings, sourced from .env
├── ingestion/
│   ├── document.py       # Document dataclass (one page of extracted text)
│   ├── process.py        # PDF extraction, reference stripping, idempotency
│   ├── chunk.py          # token-aware chunking with overlap
│   └── embed.py          # sentence-transformers wrapper
├── index/
│   └── index.py          # embeds chunks, stores/updates them in ChromaDB
└── retrieval/
    └── retrieve.py        # bm25_search, cosign_simularity, search_hybrid_rrf, retrieve()

benchmarks/                # evaluation harnesses, not library code
└── corpus_loading.py      # json-file vs. vector-db corpus loading, benchmarked

build_index.py             # root orchestrator: run the full ingestion pipeline
tests/                      # mirrors src/retrieval_arena/ + benchmarks/
```

## Stage by stage

### `config.py` — central configuration
Loads secrets and settings from `.env` (`OPENAI_API_KEY`, `COHERE_API_KEY`, `ENVIRONMENT`, `LOG_LEVEL`). Defines every path used by the pipeline (`data_dir`, `output_dir`, `chunks_dir`, `VECTORDB_DIR`), anchored to the project root via `Path(__file__).resolve()` so paths resolve correctly regardless of which directory a script is launched from. Also holds chunking settings (`CHUNK_SIZE=400`, `CHUNK_OVERLAP=60`), embedding/DB settings (`EMBED_MODEL_NAME`, `COLLECTION_NAME`), and the default retrieval strategy (`RETRIEVAL_METHOD`) — all overridable via env vars. `configure_logging()` wires `LOG_LEVEL` into Python's `logging` module, called once at the top of every entrypoint. At the default log level it also raises the noisiest third-party loggers (`httpx`, `huggingface_hub`, etc.) to ERROR and filters a couple of specific vendor warnings, since running a query was otherwise dominated by HTTP request logs and CUDA/HF Hub warnings rather than the actual results; `LOG_LEVEL=DEBUG` still shows everything.

### `ingestion/document.py` — the page-level data model
A small dataclass representing one page of extracted PDF text: `id`, `content`, `source` (originating filename), `page_num`, and a `metadata` dict.

### `ingestion/process.py` — PDF extraction, cleaning, and idempotency
- `load_pdf(path)`: opens a PDF with PyMuPDF (imported as `pymupdf`, not the deprecated `fitz` alias) and extracts text page by page, skipping blank pages. Extracted text passes through `_clean_extracted_text`, which rejoins hyphenated line-wraps (`"frame-\nwork"` → `"framework"`, only when a newline directly follows the hyphen, so real compound words like `"state-of-the-art"` are untouched) and collapses every other run of whitespace to a single space — PDF line-wraps are typographic, not semantic, and raw `\n` characters were otherwise showing up mid-word and mid-sentence in chunk text.
- `_strip_running_header(docs)`: some source PDFs print a running header (e.g. `"Preprint."`) on every page, which PyMuPDF extracts as if it were the start of the page's body text. Detected per-PDF rather than hardcoded to a specific string — a short, period-terminated token is stripped only if it repeats at the start of more than half of that PDF's pages — since not every source PDF has one.
- `Validate(docs)`: strips reference/bibliography sections before they reach the chunker. Not a naive "drop everything after References" cut — it truncates the page where the heading appears (keeping any real content before it), then drops only *subsequent* pages whose citation-marker density (`URL http`, `arXiv preprint`, `doi:`, `Proceedings of`, `pages`, counted anywhere in the text and normalized per word) crosses a threshold. Checking the real extracted text showed references can span several pages and be followed by a genuine Appendix, which a hard cutoff would have destroyed.
- `verify_data(documents)` / `load_manifest` / `save_manifest`: hash each PDF's pages independently and compare against `Data/manifest.json`, so only new or changed files are re-processed — not the whole corpus every run.
- `Process_data(path)`: orchestrates the above and writes `Data/processed/<name>.json`.

### `ingestion/chunk.py` — token-aware chunking
- `split_into_sentences`: splits on `. `/`! `/`? ` boundaries, then merges a split piece back onto the previous one if the previous one ends with a known abbreviation (`et al.`, `e.g.`, `Fig.`, etc.). A plain period-then-space split treats narrative citations like `"Wu et al. (2023) introduce..."` as a sentence boundary, which left chunks opening on a dangling fragment like `"2017) from human feedback..."` with no context for what "2017" refers to.
- `tag_sentences`: flattens a PDF's pages into one ordered list of `(sentence, page_number)` pairs, so chunking can cross page boundaries.
- `chunk_document`: greedily fills a chunk until adding the next sentence would exceed `CHUNK_SIZE` tokens (via `tiktoken`'s `cl100k_base` encoding), then carries roughly `CHUNK_OVERLAP` tokens of trailing sentences into the next chunk. Hard-splits the rare single sentence that alone exceeds the token budget.
- `chunk_data(changed_documents)`: merges newly chunked files into the existing `Data/chunks/chunks.json`, dropping only the stale chunks belonging to changed sources — the rest of the corpus's chunks are left untouched.

### `ingestion/embed.py` — embedding model wrapper
Wraps `sentence-transformers` (`all-MiniLM-L6-v2`, 384-dimensional). `get_model()` is cached with `lru_cache` so the model loads once per process.

### `index/index.py` — vector database indexing
`store_index(chunks)` embeds chunk text and stores it in a persistent ChromaDB collection, deleting only the stale entries for the sources being re-written first (not a full collection rebuild). `build_index(path)` threads `Process_data` → `chunk_data` → `store_index`, short-circuiting to `None` at any stage with nothing new to do.

### `retrieval/retrieve.py` — retrieval
Three independent search strategies, all returning the same `{idx, text, score}` shape so results are directly comparable and mergeable:
- `bm25_search(query)`: keyword search via `rank_bm25`, with an mtime-keyed cache so the index is built once and only rebuilt when `chunks.json` actually changes on disk.
- `cosign_simularity(query)`: cosine similarity search against the ChromaDB collection, embedding the query with the same pipeline used to embed the corpus.
- `search_hybrid_rrf(query)`: Reciprocal Rank Fusion over both of the above — each chunk's fused score is `sum(1 / (k + rank))` across every ranking it appears in, using rank position rather than raw score since BM25 scores and cosine distances aren't on the same scale.

`retrieve(query, method=None, top_k=5, **kwargs)` is the single entry point other code should call — it dispatches by name (`"bm25"`, `"vector"`, `"hybrid"`) and defaults to `Config.RETRIEVAL_METHOD` (`"hybrid"`) when no method is given.

### `benchmarks/corpus_loading.py`
Compares loading the corpus from `chunks.json` versus reading it back out of ChromaDB, to decide BM25's data source. Interleaves trials, reports mean and stdev (not just a single number), and includes a synthetic scaling benchmark (150 / 1,500 / 15,000 chunks) in an isolated temporary DB, in addition to the real corpus.

## Design decisions worth noting

- **No fixed-character chunking.** Chunk size is measured in tokens (via `tiktoken`), since token count is what actually determines whether text fits an embedding model's input limit.
- **Chunking crosses page boundaries deliberately**, since pages are longer than the target chunk size and a naive per-page split would cut ideas off at every page break.
- **Reference stripping is heuristic and vocabulary-density-based**, not a hard cutoff, because a real check of the extracted text showed useful content (an Appendix) can follow the bibliography. An earlier version counted `[n]`-style markers only at the *start* of each `\n`-split line, which assumed one citation per line — on some PDFs a single citation wraps across several extracted lines before the next `[n]` appears, diluting the ratio below threshold and letting whole bibliography pages through uncaught. Querying the real corpus surfaced this directly: a "machine learning algorithms" search returned mostly raw citation text instead of prose. The fix counts bibliography-specific vocabulary (URLs, DOIs, venue names, page ranges) anywhere in the page text instead of relying on line boundaries or a specific in-text citation style, which also removes the earlier dependency on numbered-bracket citations specifically. Verified against the real corpus: 27 of 147 chunks (~18%) were pure bibliography text before the fix; 0 after, and the same query now returns only prose.
- **Idempotency is per-file, not whole-corpus.** Adding one new PDF re-processes, re-chunks, and re-embeds only that file — verified by adding a duplicate PDF and confirming the vector count and per-source chunk counts changed by exactly the expected amount.
- **Retrieval methods share one result shape and one chunk-id space** (`idx` is the same id in Chroma, in BM25 results, and in the fused RRF results), which is what makes merging rankings by id in `search_hybrid_rrf` possible at all.
- **Per-file idempotency tracks extracted text, not chunking logic.** The manifest hashes each PDF's cleaned page text, so a change to how that text gets *split into chunks* (e.g. the sentence-splitter fix above) doesn't register as a change for files whose underlying text didn't change — those files' existing chunks go stale silently until something else changes them or the corpus is rebuilt from scratch. Worth knowing before trusting `build_index.py`'s output after any chunking-logic change specifically.
- **Generated data is never committed.** `Data/raw/`, `Data/processed/`, `Data/chunks/`, `Data/manifest.json`, and `vectordb/` are all `.gitignore`d — only source code and configuration are tracked, since everything else is regenerable from the pipeline.
- **Structured logging over `print`.** Every entrypoint configures logging once (`Config.configure_logging()`) and gets its own logger via `logging.getLogger(__name__)`, driven by the `LOG_LEVEL` env var.

## Running it

```bash
pip install -r requirements.txt

# build/refresh the index from Data/raw/*.pdf
python build_index.py

# query it
python main.py "your question here"
```

Individual modules are runnable directly too, e.g. `python -m src.retrieval_arena.ingestion.embed` for a quick manual check — always via `-m` from the project root, since the package uses relative imports internally.

Run the test suite with `pytest` from the project root.

## Current status / not yet done

- Only tested against 3 sample arXiv PDFs in `Data/raw/`.
- The PubMedQA benchmark comparing all three retrieval methods (the actual "arena") is in progress.
- Removed-file cleanup in the manifest/chunks/vectordb isn't automatic — a deleted source PDF's old chunks only disappear as a side effect of some other file changing.
- Idempotency is based on a hash of each PDF's extracted text; a chunking-*logic* change (chunk size, sentence-splitting rules, etc.) isn't tracked, so it silently doesn't apply to already-processed files unless the corpus is rebuilt from scratch.
- `Process_data` has no per-file error handling — one corrupted/unreadable PDF would currently abort processing of the entire batch.
- Reference-stripping thresholds (the citation-density cutoff, and which vocabulary counts as a marker) were calibrated against the 3 sample PDFs; may need re-tuning on a more varied corpus.
- No CI/CD yet.
