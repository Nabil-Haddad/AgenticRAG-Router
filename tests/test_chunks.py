import json

from src.Chunk import (
    Config,
    chunk_all,
    chunk_data,
    chunk_document,
    count_tokens,
    load_chunks,
    save_chunks_json,
    split_into_sentences,
    tag_sentences,
)
from src.Document import Document


def make_doc(content: str, page_num: int, source: str = "paper.pdf") -> Document:
    return Document(id=f"{source}_p{page_num}", content=content, source=source, page_num=page_num, metadata={})


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
    assert all(c.token_count <= chunk_size + 5 for c in chunks)


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
