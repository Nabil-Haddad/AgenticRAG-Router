import json
import logging
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

import tiktoken

try:
    from .Config import Config
    from .Document import Document
except ImportError:
    from Config import Config
    from Document import Document

logger = logging.getLogger(__name__)

ENCODING = tiktoken.get_encoding("cl100k_base")
SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+")


@dataclass
class Chunk:
    id: str
    text: str
    source: str
    page_start: int
    page_end: int
    token_count: int
    metadata: dict = field(default_factory=dict)


def count_tokens(text: str) -> int:
    return len(ENCODING.encode(text))


def split_into_sentences(text: str) -> list[str]:
    sentences = SENTENCE_SPLIT_PATTERN.split(text.strip())
    return [s.strip() for s in sentences if s.strip()]


def tag_sentences(docs: list[Document]) -> list[tuple[str, int]]:
    tagged: list[tuple[str, int]] = []
    for doc in docs:
        for sentence in split_into_sentences(doc.content):
            tagged.append((sentence, doc.page_num))
    return tagged


def _split_long_sentence(sentence: str, chunk_size: int) -> list[str]:
    tokens = ENCODING.encode(sentence)
    return [
        ENCODING.decode(tokens[start:start + chunk_size])
        for start in range(0, len(tokens), chunk_size)
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
            current = overlap_sentences
            current_tokens = overlap_tokens

        current.append((sentence, page_num))
        current_tokens += sentence_tokens

    flush()
    return chunks


def chunk_all( documents: list[list[Document]], chunk_size: int = Config.CHUNK_SIZE, overlap: int = Config.CHUNK_OVERLAP ) -> list[list[Chunk]]:
    return [chunk_document(docs, chunk_size, overlap) for docs in documents]


def save_chunks_json(chunks: list[list[Chunk]], path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    data = [asdict(c) for pdf_chunks in chunks for c in pdf_chunks]
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


def chunk_data()->list[Chunk]:
    documents = load_processed(Config.output_dir)
    chunks = chunk_all(documents)
    save_chunks_json(chunks, Config.chunks_dir)
    flatten_chunks = [c for pdf_chunks in chunks for c in pdf_chunks]
    return flatten_chunks


if __name__ == "__main__":
    Config.configure_logging()
    chunk_data()
