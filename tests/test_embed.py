import pytest

from src.Embed import Config, embed_texts, get_model


@pytest.fixture(autouse=True)
def clear_model_cache():
    # get_model is @lru_cache'd; without clearing it, mocked
    # SentenceTransformer calls in one test would be skipped in the next.
    get_model.cache_clear()
    yield
    get_model.cache_clear()


# get_model 

def test_get_model_constructs_with_default_name(mocker):
    mock_cls = mocker.patch("src.Embed.SentenceTransformer")

    get_model()

    mock_cls.assert_called_once_with(Config.EMBED_MODEL_NAME)


def test_get_model_accepts_custom_name(mocker):
    mock_cls = mocker.patch("src.Embed.SentenceTransformer")

    get_model("custom-model-name")

    mock_cls.assert_called_once_with("custom-model-name")


def test_get_model_caches_result(mocker):
    mock_cls = mocker.patch("src.Embed.SentenceTransformer")

    first = get_model()
    second = get_model()

    assert first is second
    mock_cls.assert_called_once()


# embed_texts 

def test_embed_texts_passes_texts_and_show_progress_bar(mocker):
    fake_model = mocker.MagicMock()
    fake_model.encode.return_value.tolist.return_value = [[0.1, 0.2], [0.3, 0.4]]
    mocker.patch("src.Embed.get_model", return_value=fake_model)

    result = embed_texts(["a", "b"], show_progress_bar=True)

    fake_model.encode.assert_called_once_with(["a", "b"], show_progress_bar=True)
    assert result == [[0.1, 0.2], [0.3, 0.4]]


def test_embed_texts_defaults_show_progress_bar_to_false(mocker):
    fake_model = mocker.MagicMock()
    fake_model.encode.return_value.tolist.return_value = []
    mocker.patch("src.Embed.get_model", return_value=fake_model)

    embed_texts(["a"])

    fake_model.encode.assert_called_once_with(["a"], show_progress_bar=False)
