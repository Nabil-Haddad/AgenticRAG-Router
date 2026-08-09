import json
from dataclasses import asdict
from pathlib import Path

import fitz
import pytest

from src.Document import Document
# Import Config through src.Process (not src.Config) so monkeypatching it
# affects the exact object load_directory reads — src/ is also on the
# import path (see conftest.py), and importing the "same" module via two
# different names creates two independent module/class instances.
from src.Process import Config, Validate, load_directory, load_pdf, save_pdf_json


def make_doc(content: str, page_num: int = 1, doc_id: str = "d1", source: str = "paper.pdf") -> Document:
    return Document(id=doc_id, content=content, source=source, page_num=page_num, metadata={"k": "v"})


def make_pdf(path: Path, pages: list[str]) -> None:
    doc = fitz.open()
    for text in pages:
        page = doc.new_page()
        if text:
            page.insert_text((72, 72), text)
    doc.save(str(path))
    doc.close()


# load_pdf 
def test_load_pdf_extracts_pages_with_metadata(tmp_path):
    pdf_path = tmp_path / "sample.pdf"
    make_pdf(pdf_path, ["Page one text.", "Page two text."])

    docs = load_pdf(pdf_path)

    assert len(docs) == 2
    assert docs[0].id == "sample_page_001"
    assert docs[0].page_num == 1
    assert docs[0].source == "sample.pdf"
    assert "Page one text." in docs[0].content
    assert docs[0].metadata["char_count"] == len(docs[0].content)
    assert docs[1].id == "sample_page_002"
    assert docs[1].page_num == 2


def test_load_pdf_skips_blank_pages(tmp_path):
    pdf_path = tmp_path / "sample.pdf"
    make_pdf(pdf_path, ["", "Only page with text."])

    docs = load_pdf(pdf_path)

    assert len(docs) == 1
    # original page numbering is preserved even though page 1 was skipped
    assert docs[0].page_num == 2


def test_load_pdf_raises_for_missing_file(tmp_path):
    missing = tmp_path / "does_not_exist.pdf"
    with pytest.raises(ValueError):
        load_pdf(missing)


#  Validate

def test_validate_truncates_at_references_heading():
    content = "Real content before.\nReferences\n[1] Some Citation."
    docs = [make_doc(content)]

    result = Validate(docs)

    assert len(result) == 1
    assert result[0].content == "Real content before."
    assert result[0].id == "d1"
    assert result[0].source == "paper.pdf"
    assert result[0].page_num == 1
    assert result[0].metadata == {"k": "v"}


def test_validate_drops_reference_heavy_page():
    lines = [f"[{i}] Some Author, Some Title, Year." for i in range(8)] + ["one more line"]
    content = "\n".join(lines)  # 8/9 lines match the citation pattern
    docs = [make_doc(content)]

    assert Validate(docs) == []


def test_validate_keeps_normal_prose_page():
    content = "This is a normal paragraph.\nIt has a couple of lines.\nNone of them are citations."
    docs = [make_doc(content)]

    result = Validate(docs)

    assert len(result) == 1
    assert result[0].content == content


def test_validate_handles_mixed_batch_preserving_order():
    keep_doc = make_doc("Normal prose here.\nMore normal prose.", doc_id="keep")
    truncate_doc = make_doc("Kept part.\nReferences\n[1] citation", doc_id="trunc")
    drop_doc = make_doc("\n".join(f"[{i}] citation line" for i in range(10)), doc_id="drop")

    result = Validate([keep_doc, truncate_doc, drop_doc])

    assert [d.id for d in result] == ["keep", "trunc"]
    assert result[1].content == "Kept part."


#  save_pdf_json 

def test_save_pdf_json_writes_one_file_per_pdf(tmp_path):
    docs_pdf_a = [make_doc("Page 1 content.", page_num=1, doc_id="a_page_001", source="a.pdf")]
    docs_pdf_b = [make_doc("Page 1 content.", page_num=1, doc_id="b_page_001", source="b.pdf")]

    save_pdf_json([docs_pdf_a, docs_pdf_b], tmp_path)

    file_a = tmp_path / "a.json"
    file_b = tmp_path / "b.json"
    assert file_a.exists()
    assert file_b.exists()

    data_a = json.loads(file_a.read_text(encoding="utf-8"))
    assert data_a == [asdict(docs_pdf_a[0])]


# load_directory 

def test_load_directory_raises_for_missing_dir(tmp_path):
    missing_dir = tmp_path / "does_not_exist"
    with pytest.raises(ValueError):
        load_directory(missing_dir)


def test_load_directory_processes_only_matching_files(tmp_path, monkeypatch):
    output_dir = tmp_path / "out"
    monkeypatch.setattr(Config, "output_dir", output_dir)

    data_dir = tmp_path / "raw"
    data_dir.mkdir()
    make_pdf(data_dir / "paper.pdf", ["Some real content here."])
    (data_dir / "notes.txt").write_text("irrelevant file", encoding="utf-8")
    (data_dir / "subdir").mkdir()

    load_directory(data_dir)

    output_file = output_dir / "paper.json"
    assert output_file.exists()
    data = json.loads(output_file.read_text(encoding="utf-8"))
    assert len(data) == 1
    assert data[0]["source"] == "paper.pdf"
