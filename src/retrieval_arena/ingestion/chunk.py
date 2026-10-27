import json
import logging
import re
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path

from transformers import AutoTokenizer

from ..config import Config
from .document import Document

logger = logging.getLogger(__name__)

SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+")
ABBREVIATIONS = ("et al.", "e.g.", "i.e.", "cf.", "vs.", "Fig.", "Eq.", "etc.")


@dataclass
class Chunk:
    id: str
    text: str
    source: str
    page_start: int
    page_end: int
    token_count: int
    metadata: dict = field(default_factory=dict)


@lru_cache(maxsize=1)
def get_tokenizer() -> AutoTokenizer:
    # the embedder's own WordPiece tokenizer, so chunk sizes match what the model actually sees.
    # model_max_length is lifted only to silence the "sequence too long" warning on long sentences
    return AutoTokenizer.from_pretrained(
        f"sentence-transformers/{Config.EMBED_MODEL_NAME}", model_max_length=int(1e9)
    )


def count_tokens(text: str) -> int:
    return len(get_tokenizer().encode(text, add_special_tokens=False))


def split_into_sentences(text: str) -> list[str]:
    pieces = SENTENCE_SPLIT_PATTERN.split(text.strip())
    sentences: list[str] = []
    for piece in pieces:
        piece = piece.strip()
        if not piece:
            continue
        if sentences and sentences[-1].endswith(ABBREVIATIONS):
            sentences[-1] = f"{sentences[-1]} {piece}"
        else:
            sentences.append(piece)
    return sentences


def tag_sentences(docs: list[Document]) -> list[tuple[str, int]]:
    tagged: list[tuple[str, int]] = []
    for doc in docs:
        for sentence in split_into_sentences(doc.content):
            tagged.append((sentence, doc.page_num))
    return tagged


def _split_long_sentence(sentence: str, chunk_size: int) -> list[str]:
    # slice the original text by offsets; decoding WordPiece ids would lowercase and alter it
    offsets = get_tokenizer()(
        sentence, add_special_tokens=False, return_offsets_mapping=True
    )["offset_mapping"]
    return [
        sentence[offsets[start][0]:offsets[min(start + chunk_size, len(offsets)) - 1][1]]
        for start in range(0, len(offsets), chunk_size)
    ]


def chunk_document(docs: list[Document], chunk_size: int = Config.CHUNK_SIZE, overlap: int = Config.CHUNK_OVERLAP) -> list[Chunk]:
    if not docs:
        return []

    source = docs[0].source
    tagged = tag_sentences(docs)

    chunks: list[Chunk] = []
    current: list[tuple[str, int]] = []
    current_tokens = 0
    chunk_index = 0

    def flush() -> None:
        nonlocal chunk_index
        if not current:
            return
        text = " ".join(s for s, _ in current)
        chunks.append(Chunk(
            id=f"{source.removesuffix('.pdf')}_chunk_{chunk_index:03d}",
            text=text,
            source=source,
            page_start=current[0][1],
            page_end=current[-1][1],
            token_count=count_tokens(text),
            metadata={},
        ))
        chunk_index += 1

    for sentence, page_num in tagged:
        sentence_tokens = count_tokens(sentence)

        if sentence_tokens > chunk_size:
            # a single sentence too big to ever fit in a normal chunk
            flush()
            current = []
            current_tokens = 0
            for piece in _split_long_sentence(sentence, chunk_size):
                chunks.append(Chunk(
                    id=f"{source.removesuffix('.pdf')}_chunk_{chunk_index:03d}",
                    text=piece,
                    source=source,
                    page_start=page_num,
                    page_end=page_num,
                    token_count=count_tokens(piece),
                    metadata={"hard_split": True},
                ))
                chunk_index += 1
            continue

        if current_tokens + sentence_tokens > chunk_size and current:
            flush()
            # carry trailing sentences from the finished chunk into the next one
            overlap_sentences: list[tuple[str, int]] = []
            overlap_tokens = 0
            for s, p in reversed(current):
                t = count_tokens(s)
                if overlap_tokens + t > overlap:
                    break
                overlap_sentences.insert(0, (s, p))
                overlap_tokens += t
            # the carried overlap must leave room for the sentence about to be added,
            # otherwise the new chunk starts out over budget
            while overlap_sentences and overlap_tokens + sentence_tokens > chunk_size:
                overlap_sentences.pop(0)
                overlap_tokens = sum(count_tokens(s) for s, _ in overlap_sentences)
            current = overlap_sentences
            current_tokens = overlap_tokens

        current.append((sentence, page_num))
        current_tokens += sentence_tokens

    flush()
    return chunks


def validate_chunk_lengths(chunks: list[Chunk], max_seq_length: int = Config.EMBED_MAX_SEQ_LENGTH) -> None:
    # re-tokenize with [CLS]/[SEP]: this is the length the embedder truncates at
    tokenizer = get_tokenizer()
    too_long = [c.id for c in chunks if len(tokenizer.encode(c.text)) > max_seq_length]
    if too_long:
        raise ValueError(
            f"{len(too_long)} chunk(s) exceed the embedder's {max_seq_length}-token limit "
            f"and would be silently truncated (first: {too_long[0]}). Lower CHUNK_SIZE."
        )


def chunk_all( documents: list[list[Document]], chunk_size: int = Config.CHUNK_SIZE, overlap: int = Config.CHUNK_OVERLAP ) -> list[list[Chunk]]:
    return [chunk_document(docs, chunk_size, overlap) for docs in documents]


def save_chunks_json(chunks: list[Chunk], path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    data = [asdict(c) for c in chunks]
    json_path = path / "chunks.json"
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)
    logger.info(f"{len(data)} chunks saved at {json_path}")


def load_processed(path: Path) -> list[list[Document]]:
    documents: list[list[Document]] = []
    for json_path in sorted(path.glob("*.json")):
        with open(json_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        documents.append([Document(**d) for d in data])
    return documents


def load_chunks(path: Path) -> list[Chunk]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as fh:
        return [Chunk(**d) for d in json.load(fh)]


def chunk_data(changed_documents: list[list[Document]]) -> list[Chunk]:
    if not changed_documents:
        return []

    changed_sources = {docs[0].source for docs in changed_documents if docs}

    existing_chunks = load_chunks(Config.chunks_dir / "chunks.json")
    # stale: these files are being re-chunked, so their old chunks no longer apply
    retained_chunks = [c for c in existing_chunks if c.source not in changed_sources]

    new_chunks_grouped = chunk_all(changed_documents)
    new_chunks = [c for pdf_chunks in new_chunks_grouped for c in pdf_chunks]
    validate_chunk_lengths(new_chunks)

    save_chunks_json(retained_chunks + new_chunks, Config.chunks_dir)

    return new_chunks


if __name__ == "__main__":
    Config.configure_logging()
    chunk_data(load_processed(Config.output_dir))
