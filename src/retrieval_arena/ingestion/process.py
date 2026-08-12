import logging
from ..config import Config
from .document import Document
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


# returns (bool, list[list[Document]] | None):
# if we have the same data we return (False, None)
# if we have completely new data we return (True, docs)
# if we have partial new data we return (True, only the changed/new pdfs)
def verify_data(documents: list[list[Document]]) -> tuple[bool, list[list[Document]] | None]:
    manifest = load_manifest(Config.manifest_path)
    stored_files = manifest.get("files", {}) if manifest is not None else {}

    # hash each pdf's pages on their own, so we can tell which specific
    # files are new/changed instead of only "something in the corpus changed"
    new_files: dict[str, str] = {}
    changed_documents: list[list[Document]] = []
    for pdf_docs in documents:
        if not pdf_docs:
            continue
        source = pdf_docs[0].source
        file_hash = hash_string_list([d.content for d in pdf_docs])
        new_files[source] = file_hash
        if stored_files.get(source) != file_hash:
            changed_documents.append(pdf_docs)

    if not changed_documents:
        logger.info("All files match the stored manifest. Data hasn't changed. Skipping build.")
        return False, None

    save_manifest({
        "schema_version": 1,
        "embed_model": Config.EMBED_MODEL_NAME,
        "chunk_size": Config.CHUNK_SIZE,
        "chunk_overlap": Config.CHUNK_OVERLAP,
        "files": new_files,
    }, Config.manifest_path)

    if manifest is None:
        logger.info("No manifest found; indexing all files for the first time.")
    else:
        changed_sources = [pdf_docs[0].source for pdf_docs in changed_documents]
        logger.info(f"{len(changed_documents)} file(s) new or changed: {changed_sources}")

    return True, changed_documents



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


def Process_data(path: Path, extensions: list[str] | None = None) -> list[list[Document]] | None:
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
    # now we deal with every case
    valid , new_docs = verify_data(documents)
        # if every thing is ok , save the documents
    if valid:
        save_pdf_json(new_docs , Config.output_dir)
        return new_docs
    else:
        logger.info("No changes detected in source PDFs; skipping save and reindex.")
        return None
    
        


def main():
    Process_data(Config.data_dir)


if __name__ == "__main__":
    Config.configure_logging()
    main()
