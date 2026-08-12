from src.retrieval_arena.ingestion.document import Document


def test_document_holds_all_fields():
    # Arrange + Act
    doc = Document(
        id="doc-1",
        content="Some page text.",
        source="paper.pdf",
        page_num=3,
        metadata={"section": "intro"},
    )

    # Assert
    assert doc.id == "doc-1"
    assert doc.content == "Some page text."
    assert doc.source == "paper.pdf"
    assert doc.page_num == 3
    assert doc.metadata == {"section": "intro"}


def test_document_metadata_defaults_are_independent():
    doc_a = Document(id="a", content="x", source="s.pdf", page_num=1, metadata={})
    doc_b = Document(id="b", content="y", source="s.pdf", page_num=1, metadata={})

    doc_a.metadata["touched"] = True

    assert "touched" not in doc_b.metadata