from src.retrive import cosign_simularity, peek_first_5_elements


# peek_first_5_elements

def test_peek_prints_samples_on_success(mocker, capsys):
    fake_collection = mocker.MagicMock()
    fake_collection.peek.return_value = {
        "ids": ["c1", "c2"],
        "documents": ["first document text", "second document text"],
    }
    mocker.patch("src.retrive.get_collection", return_value=fake_collection)

    result = peek_first_5_elements()

    fake_collection.peek.assert_called_once_with(limit=5)
    assert result is None
    out = capsys.readouterr().out
    assert "VectorDB Connection Successful" in out
    assert "Retrieved 2 items" in out
    assert "c1" in out
    assert "c2" in out


def test_peek_handles_failure_without_crashing(mocker, capsys):
    fake_collection = mocker.MagicMock()
    fake_collection.peek.side_effect = Exception("simulated connection failure")
    mocker.patch("src.retrive.get_collection", return_value=fake_collection)

    result = peek_first_5_elements()

    assert result is None
    out = capsys.readouterr().out
    assert "connection failed" in out
    assert "simulated connection failure" in out


# cosign_simularity

def test_cosign_simularity_embeds_query_and_returns_results(mocker):
    fake_collection = mocker.MagicMock()
    fake_collection.query.return_value = {
        "ids": [["c1", "c2"]],
        "documents": [["text one", "text two"]],
        "distances": [[0.1, 0.2]],
    }
    mocker.patch("src.retrive.get_collection", return_value=fake_collection)
    mock_embed = mocker.patch("src.retrive.embed_texts", return_value=[[0.5, 0.6]])

    results = cosign_simularity("what is RAG?", top_k=2)

    mock_embed.assert_called_once_with(["what is RAG?"])
    fake_collection.query.assert_called_once_with(
        query_embeddings=[[0.5, 0.6]],
        n_results=2,
    )
    assert results == [
        {"idx": "c1", "text": "text one", "score": 0.1},
        {"idx": "c2", "text": "text two", "score": 0.2},
    ]


def test_cosign_simularity_defaults_top_k_to_5(mocker):
    fake_collection = mocker.MagicMock()
    fake_collection.query.return_value = {"ids": [[]], "documents": [[]], "distances": [[]]}
    mocker.patch("src.retrive.get_collection", return_value=fake_collection)
    mocker.patch("src.retrive.embed_texts", return_value=[[0.1]])

    cosign_simularity("a query")

    fake_collection.query.assert_called_once_with(
        query_embeddings=[[0.1]],
        n_results=5,
    )


def test_cosign_simularity_returns_empty_list_on_failure(mocker):
    fake_collection = mocker.MagicMock()
    fake_collection.query.side_effect = Exception("simulated connection failure")
    mocker.patch("src.retrive.get_collection", return_value=fake_collection)
    mocker.patch("src.retrive.embed_texts", return_value=[[0.1]])

    assert cosign_simularity("a query") == []
