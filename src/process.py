from Config import Config
from Document import Document
from pathlib import Path
import fitz
from dataclasses import asdict


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
    


def load_directory(path: Path, extensions: list[str] = None) -> None:
    if extensions is None:
        extensions = [".pdf"]

    if not path.exists():
        raise ValueError("Warning Directory doesn't exist")

    for filepath in sorted(path.glob("*")):
        if filepath.suffix.lower() not in extensions:
            continue
        if not filepath.is_file():
            continue
        documents = load_pdf(filepath)
        print(documents)


def main():
    load_directory(Config.data_dir)


if __name__ == "__main__":
    main()
