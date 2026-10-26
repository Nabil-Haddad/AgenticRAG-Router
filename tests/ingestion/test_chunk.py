import json

import pytest

from src.retrieval_arena.config import Config
from src.retrieval_arena.ingestion.chunk import (
    chunk_all,
    chunk_data,
    chunk_document,
    count_tokens,
    get_tokenizer,
    load_chunks,
    save_chunks_json,
    split_into_sentences,
    tag_sentences,
    validate_chunk_lengths,
)
from src.retrieval_arena.ingestion.document import Document


def make_doc(content: str, page_num: int, source: str = "paper.pdf") -> Document:
    return Document(id=f"{source}_p{page_num}", content=content, source=source, page_num=page_num, metadata={})


# count_tokens

def test_count_tokens_uses_the_embedders_wordpiece_tokenizer():
    text = "Unbelievably, retrieval-augmented generation works."
    expected = len(get_tokenizer().encode(text, add_special_tokens=False))
    assert count_tokens(text) == expected
    # WordPiece splits rare words into pieces, so this is more than the whitespace word count
    assert count_tokens(text) > len(text.split())


def test_count_tokens_excludes_special_tokens():
    assert count_tokens("hello") == 1


def test_embed_max_seq_length_matches_the_real_model():
    sentence_transformers = pytest.importorskip("sentence_transformers")
    try:
        model = sentence_transformers.SentenceTransformer(Config.EMBED_MODEL_NAME)
    except Exception:
        pytest.skip("embedding model not available offline")
    assert model.max_seq_length == Config.EMBED_MAX_SEQ_LENGTH


# Split_into_sentences

def test_split_into_sentences_basic():
    text = "First sentence. Second sentence! Third one?"
    assert split_into_sentences(text) == [
        "First sentence.",
        "Second sentence!",
        "Third one?",
    ]


def test_split_into_sentences_strips_blank_and_whitespace():
    text = "  One sentence.   \n\n  Another sentence.  "
    result = split_into_sentences(text)
    assert result == ["One sentence.", "Another sentence."]


def test_split_into_sentences_empty_text():
    assert split_into_sentences("") == []
    assert split_into_sentences("   ") == []


def test_split_into_sentences_does_not_split_on_narrative_citation():
    text = "Wu et al. (2023) introduce fine-grained RLHF with multiple reward models."
    assert split_into_sentences(text) == [text]


def test_split_into_sentences_merges_multiple_chained_abbreviations():
    text = "See Fig. 3 for details. Also refer to Eq. 5 for the derivation."
    assert split_into_sentences(text) == [
        "See Fig. 3 for details.",
        "Also refer to Eq. 5 for the derivation.",
    ]


def test_split_into_sentences_still_splits_real_sentence_boundaries():
    text = "Wu et al. (2023) introduce fine-grained RLHF. This is a new sentence."
    assert split_into_sentences(text) == [
        "Wu et al. (2023) introduce fine-grained RLHF.",
        "This is a new sentence.",
    ]


# Tag_sentences

def test_tag_sentences_preserves_order_and_page_numbers():
    docs = [
        make_doc("Sentence one. Sentence two.", page_num=1),
        make_doc("Sentence three.", page_num=2),
    ]
    assert tag_sentences(docs) == [
        ("Sentence one.", 1),
        ("Sentence two.", 1),
        ("Sentence three.", 2),
    ]


# Chunk_document 

def test_chunk_document_empty_input_returns_empty_list():
    assert chunk_document([], chunk_size=100, overlap=10) == []


def test_chunk_document_single_chunk_when_everything_fits():
    docs = [make_doc("Short sentence one. Short sentence two.", page_num=1)]
    chunks = chunk_document(docs, chunk_size=1000, overlap=100)

    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.text == "Short sentence one. Short sentence two."
    assert chunk.source == "paper.pdf"
    assert chunk.page_start == 1
    assert chunk.page_end == 1
    assert chunk.id == "paper_chunk_000"
    assert chunk.token_count == count_tokens(chunk.text)


def test_chunk_document_splits_when_over_budget():
    sentences = [f"This is test sentence number {i}." for i in range(10)]
    docs = [make_doc(" ".join(sentences), page_num=1)]

    # tight enough budget that everything can't fit in one chunk
    per_sentence_tokens = max(count_tokens(s) for s in sentences)
    chunk_size = per_sentence_tokens * 3
    chunks = chunk_document(docs, chunk_size=chunk_size, overlap=0)

    assert len(chunks) > 1
    # with no overlap, chunks should reconstruct the input exactly, in order
    assert " ".join(c.text for c in chunks) == " ".join(sentences)
    # ids are sequential per source
    assert [c.id for c in chunks] == [f"paper_chunk_{i:03d}" for i in range(len(chunks))]


def test_chunk_document_overlap_repeats_trailing_sentence():
    sentences = [f"Sentence number {i} is here." for i in range(6)]
    docs = [make_doc(" ".join(sentences), page_num=1)]

    token_len = count_tokens(sentences[0])
    chunk_size = token_len * 2
    overlap = token_len

    chunks = chunk_document(docs, chunk_size=chunk_size, overlap=overlap)
    assert len(chunks) >= 2

    # the sentence carried forward as overlap should end one chunk and
    # start the next
    for i in range(len(chunks) - 1):
        end_of_this = split_into_sentences(chunks[i].text)[-1]
        start_of_next = split_into_sentences(chunks[i + 1].text)[0]
        assert end_of_this == start_of_next


def test_chunk_document_overlap_is_dropped_when_it_would_push_the_next_sentence_over_budget():
    # a short sentence followed by one that nearly fills the chunk: carrying the short one
    # as overlap would make the next chunk exceed chunk_size
    short = "Short lead in sentence here."
    long_ = " ".join(["alpha"] * 90) + "."
    filler = " ".join(["beta"] * 90) + "."
    docs = [make_doc(" ".join([filler, short, long_]), page_num=1)]

    chunk_size = count_tokens(long_) + 5
    overlap = count_tokens(short) + 2  # big enough to carry `short` into the next chunk
    chunks = chunk_document(docs, chunk_size=chunk_size, overlap=overlap)

    assert len(chunks) >= 2
    assert all(count_tokens(c.text) <= chunk_size for c in chunks)


def test_chunk_document_never_exceeds_chunk_size_with_overlap():
    sentences = [" ".join(["word"] * n) + "." for n in (5, 60, 8, 55, 3, 59, 12, 58)] * 6
    docs = [make_doc(" ".join(sentences), page_num=1)]

    for chunk_size, overlap in [(70, 20), (64, 40), (100, 30)]:
        chunks = chunk_document(docs, chunk_size=chunk_size, overlap=overlap)
        assert all(count_tokens(c.text) <= chunk_size for c in chunks), (chunk_size, overlap)


def test_chunk_document_tracks_page_span_across_pages():
    docs = [
        make_doc("Sentence on page one.", page_num=1),
        make_doc("Sentence on page two.", page_num=2),
    ]
    chunks = chunk_document(docs, chunk_size=1000, overlap=0)

    assert len(chunks) == 1
    assert chunks[0].page_start == 1
    assert chunks[0].page_end == 2


def test_chunk_document_hard_splits_oversized_single_sentence():
    # one giant "sentence" (no internal punctuation) that alone exceeds chunk_size
    huge_sentence = ("word " * 500).strip() + "."
    docs = [make_doc(huge_sentence, page_num=1)]

    chunk_size = 50
    chunks = chunk_document(docs, chunk_size=chunk_size, overlap=0)

    assert len(chunks) > 1
    assert all(c.metadata.get("hard_split") for c in chunks)
    assert all(c.token_count <= chunk_size for c in chunks)


def test_chunk_document_hard_split_preserves_original_text():
    # decoding WordPiece ids would lowercase this; slicing by offsets must not
    sentence = " ".join(["Transformers", "BERT", "RAG", "Naïve"] * 40) + "."
    docs = [make_doc(sentence, page_num=1)]

    chunks = chunk_document(docs, chunk_size=20, overlap=0)

    assert len(chunks) > 1
    assert all(c.token_count <= 20 for c in chunks)
    # pieces are cut at token boundaries, so only whitespace between them can differ
    assert "".join("".join(c.text.split()) for c in chunks) == "".join(sentence.split())
    assert any("Transformers" in c.text for c in chunks)


def test_chunk_document_default_size_chunks_fit_the_embedder_window():
    sentences = [f"Retrieval-augmented models handle claim number {i} rather unevenly." for i in range(300)]
    docs = [make_doc(" ".join(sentences), page_num=1)]

    # the documented defaults (Config.CHUNK_SIZE / CHUNK_OVERLAP can be overridden by a local .env)
    chunks = chunk_document(docs, chunk_size=250, overlap=40)

    assert len(chunks) > 1
    validate_chunk_lengths(chunks)
    tokenizer = get_tokenizer()
    assert all(len(tokenizer.encode(c.text)) <= Config.EMBED_MAX_SEQ_LENGTH for c in chunks)


# validate_chunk_lengths

def test_validate_chunk_lengths_accepts_chunks_within_the_limit():
    chunks = chunk_document([make_doc("A short sentence.", page_num=1)])
    validate_chunk_lengths(chunks)


def test_validate_chunk_lengths_counts_special_tokens():
    # exactly max_seq_length content tokens + [CLS] + [SEP] overflows by 2
    text = " ".join(["word"] * Config.EMBED_MAX_SEQ_LENGTH)
    chunk = chunk_document([make_doc(text, page_num=1)], chunk_size=1000)[0]
    assert chunk.token_count == Config.EMBED_MAX_SEQ_LENGTH
    with pytest.raises(ValueError, match="exceed"):
        validate_chunk_lengths([chunk])


def test_validate_chunk_lengths_names_the_offending_chunk():
    text = " ".join(["word"] * 300)
    chunk = chunk_document([make_doc(text, page_num=1)], chunk_size=1000)[0]
    with pytest.raises(ValueError, match=chunk.id):
        validate_chunk_lengths([chunk])


# Chunk_all 

def test_chunk_all_resets_chunk_index_per_document():
    doc_a = [make_doc("Sentence a one. Sentence a two.", page_num=1, source="a.pdf")]
    doc_b = [make_doc("Sentence b one. Sentence b two.", page_num=1, source="b.pdf")]

    results = chunk_all([doc_a, doc_b], chunk_size=1000, overlap=0)

    assert len(results) == 2
    assert results[0][0].id == "a_chunk_000"
    assert results[1][0].id == "b_chunk_000"


# Save_chunks_json

def test_save_chunks_json_writes_combined_file(tmp_path):
    doc_a = [make_doc("Sentence a.", page_num=1, source="a.pdf")]
    doc_b = [make_doc("Sentence b.", page_num=1, source="b.pdf")]
    grouped = chunk_all([doc_a, doc_b], chunk_size=1000, overlap=0)
    chunks = [c for pdf_chunks in grouped for c in pdf_chunks]

    save_chunks_json(chunks, tmp_path)

    output_file = tmp_path / "chunks.json"
    assert output_file.exists()

    data = json.loads(output_file.read_text(encoding="utf-8"))
    assert len(data) == 2
    assert {d["source"] for d in data} == {"a.pdf", "b.pdf"}


# load_chunks

def test_load_chunks_returns_empty_list_when_missing(tmp_path):
    assert load_chunks(tmp_path / "chunks.json") == []


def test_load_chunks_round_trips_with_save_chunks_json(tmp_path):
    doc_a = [make_doc("Sentence a.", page_num=1, source="a.pdf")]
    grouped = chunk_all([doc_a], chunk_size=1000, overlap=0)
    chunks = [c for pdf_chunks in grouped for c in pdf_chunks]

    save_chunks_json(chunks, tmp_path)

    assert load_chunks(tmp_path / "chunks.json") == chunks


# chunk_data

def test_chunk_data_empty_input_returns_empty_list():
    assert chunk_data([]) == []


def test_chunk_data_writes_and_returns_chunks_on_first_run(tmp_path, monkeypatch):
    monkeypatch.setattr(Config, "chunks_dir", tmp_path)

    doc_a = [make_doc("Sentence a.", page_num=1, source="a.pdf")]

    new_chunks = chunk_data([doc_a])

    assert len(new_chunks) == 1
    assert new_chunks[0].source == "a.pdf"
    saved = load_chunks(tmp_path / "chunks.json")
    assert saved == new_chunks


def test_chunk_data_rejects_chunks_that_would_be_truncated(tmp_path, monkeypatch):
    monkeypatch.setattr(Config, "chunks_dir", tmp_path)
    monkeypatch.setattr(Config, "CHUNK_SIZE", 1000)
    monkeypatch.setattr(
        "src.retrieval_arena.ingestion.chunk.chunk_all",
        lambda docs: [chunk_document(d, chunk_size=1000, overlap=0) for d in docs],
    )
    doc = [make_doc(" ".join(["word"] * 400), page_num=1, source="a.pdf")]

    with pytest.raises(ValueError, match="exceed"):
        chunk_data([doc])

    # nothing bad is written to disk
    assert not (tmp_path / "chunks.json").exists()


def test_chunk_data_merges_changed_file_without_touching_others(tmp_path, monkeypatch):
    monkeypatch.setattr(Config, "chunks_dir", tmp_path)

    doc_a = [make_doc("Sentence a.", page_num=1, source="a.pdf")]
    doc_b = [make_doc("Sentence b.", page_num=1, source="b.pdf")]
    chunk_data([doc_a, doc_b])  # baseline: both files indexed

    doc_b_changed = [make_doc("Sentence b changed.", page_num=1, source="b.pdf")]
    new_chunks = chunk_data([doc_b_changed])

    # only the changed file's chunk comes back for (re-)embedding
    assert len(new_chunks) == 1
    assert new_chunks[0].source == "b.pdf"
    assert new_chunks[0].text == "Sentence b changed."

    # but the combined file on disk still has both, a untouched and b updated
    all_chunks = load_chunks(tmp_path / "chunks.json")
    sources = {c.source: c.text for c in all_chunks}
    assert sources == {"a.pdf": "Sentence a.", "b.pdf": "Sentence b changed."}
