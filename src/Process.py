import logging
from Config import Config
from Document import Document
from pathlib import Path
import fitz
import json
from typing import Optional
from dataclasses import asdict

logger = logging.getLogger(__name__)



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


def log_pdf(docs : list[list[Document]])-> None:
    for i, list_pages in enumerate(docs):
        logger.info(f"This is the pdf number {i} : ")
        for doc in list_pages:
            logger.debug(f"document : {doc.id}\nsource : {doc.source}\npage number: {doc.page_num}\ncontent : {len(doc.content.split())}")
        logger.info("#" * 50)


def load_directory(path: Path, extensions: list[str] = None) -> None:
    if extensions is None:
        extensions = [".pdf"]
    documents = []

    if not path.exists():
        raise ValueError("Warning Directory doesn't exist")
    i = 0
    for filepath in sorted(path.glob("*")):
        if filepath.suffix.lower() not in extensions:
            continue
        if not filepath.is_file():
            continue
        list_pages = load_pdf(filepath)
        documents.append(list_pages)
    # log_pdf(documents)
    save_pdf_json(documents , Config.output_dir)
        


def main():
    load_directory(Config.data_dir)


if __name__ == "__main__":
    Config.configure_logging()
    main()
