from Config import Config
from sentence_transformers import SentenceTransformer
from functools import lru_cache
import logging


logger = logging.getLogger(__name__)

@lru_cache(maxsize=1)
def get_model(name : str = Config.EMBED_MODEL_NAME)->SentenceTransformer:
    return SentenceTransformer(name)

def embed_texts(texts : list[str] , show_progress_bar: bool = False)-> list[list[float]]:
    model = get_model()
    return model.encode(texts, show_progress_bar=show_progress_bar).tolist()


if __name__ == "__main__":
    Config.configure_logging()
    # Quick manual test: `python src/Embed.py`
    vec = embed_texts(["binary search runs in O(log n) time"])[0]
    logger.info(f"Embedded 1 text -> vector of length {len(vec)}")
