import logging
try:
    from .Config import Config
    from .Document import Document
except ImportError:
    from Config import Config
    from Document import Document
from pathlib import Path
import hashlib
import fitz
import json
import re
from dataclasses import asdict

logger = logging.getLogger(__name__)



def load_manifest(path: Path = Config.manifest_path) -> dict | None:
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (json.JSONDecodeError, PermissionError) as e:
        logger.warning(f"Error reading manifest at {path}: {e}")
        return None


def save_manifest(manifest: dict, path: Path = Config.manifest_path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)


def hash_string_list(strings: list[str]) -> str:
    if not strings:
        return hashlib.sha256(b"").hexdigest()
    sorted_strings = sorted(strings)
    combined_text = "||".join(sorted_strings)
    return hashlib.sha256(combined_text.encode("utf-8")).hexdigest()


def verify_data(docs : list[Document])->bool:
    manifest = load_manifest(Config.manifest_path)
    # Extract text from docs
    text = [d.content for d in docs]
    new_hash = hash_string_list(text)

    if manifest is not None and manifest.get("hashed_text") == new_hash:
        logger.info("Stored hash matches new hash. Data hasn't changed. Skipping build.")
        return False

    save_manifest({
        "schema_version": 1,
        "embed_model": Config.EMBED_MODEL_NAME,
        "chunk_size": Config.CHUNK_SIZE,
        "chunk_overlap": Config.CHUNK_OVERLAP,
        "hashed_text": new_hash,
    }, Config.manifest_path)
    if manifest is None:
        logger.info("No manifest found; created a new one and proceeding with indexing.")
    else:
        logger.info("Data has changed since the last run; reindexing.")
    return True



def log_pdf(docs : list[list[Document]])-> None:
    for i, list_pages in enumerate(docs):
        logger.info(f"This is the pdf number {i} : ")
        for doc in list_pages:
            logger.debug(f"document : {doc.id}\nsource : {doc.source}\npage number: {doc.page_num}\ncontent : {len(doc.content.split())}")
        logger.info("#" * 50)


def Validate(docs: list[Document]) -> list[Document]:
    new_documents : list[Document] = []
    word = "References"
    pattern = r"^\[\d+\]"
    for doc in docs:
        # "References" heading marks the start of the bibliography;
        # keep only the content before it
        if word in doc.content.split():
            before, match, after = doc.content.partition(word)
            new_content = before.strip()
        else :
            phrases = doc.content.split("\n")
            phrases_number = len(phrases)
            references = [phrase for phrase in phrases if re.match(pattern, phrase)]
            references_number = len(references)
            if references_number / phrases_number >= 0.20:
                continue
            new_content = doc.content
        new_documents.append(Document(
                    id = doc.id,
                    content = new_content,
                    source = doc.source,
                    page_num = doc.page_num,
                    metadata = doc.metadata,
                            ))
    return new_documents



                

def save_pdf_json(docs: list[Document], path: Path)->None:
    for d in docs:
        data = []
        json_path : Path = path / (d[0].source.removesuffix(".pdf") + ".json")
        for page in d:
            data.append(asdict(page))

        json_path.parent.mkdir(parents=True, exist_ok=True)

        with open(json_path , "w" , encoding="UTF-8")as fh:
            json.dump(data , fh ,indent=2, ensure_ascii=False)

        logger.info(f"{len(data)} pages saved at {json_path}")


def load_pdf(path: Path)->list[Document]:
    docs: list[Document] = []
    if not path.exists():
        raise ValueError("Path doesn't exits")
    with fitz.open(path) as pdf:
        for page_num, page in enumerate(pdf, start=1):
            text = page.get_text("text").strip()
            if not text:
                continue
            docs.append(Document(
                id = f"{path.stem}_page_{page_num:03d}",
                content = text,
                source = path.name,
                page_num = page_num,
                metadata = {"char_count": len(text)}
            ))
    return docs


def Process_data(path: Path, extensions: list[str] | None = None) -> bool:
    if extensions is None:
        extensions = [".pdf"]
    documents = []

    if not path.exists():
        raise ValueError("Warning Directory doesn't exist")
    for filepath in sorted(path.glob("*")):
        if filepath.suffix.lower() not in extensions:
            continue
        if not filepath.is_file():
            continue
        list_pages = load_pdf(filepath)
        list_pages = Validate(list_pages)
        documents.append(list_pages)
    # until here we would have our clean pdf's
    flat_documents = [doc for pdf_docs in documents for doc in pdf_docs]
    if verify_data(flat_documents):
        # if every thing is ok , save the documents
        save_pdf_json(documents , Config.output_dir)
        return True
    else:
        logger.info("No changes detected in source PDFs; skipping save and reindex.")
        return False
    
        


def main():
    Process_data(Config.data_dir)


if __name__ == "__main__":
    Config.configure_logging()
    main()
