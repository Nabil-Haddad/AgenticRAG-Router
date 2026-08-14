import json
import logging
from pathlib import Path

from datasets import load_dataset

from src.retrieval_arena.config import Config

logger = logging.getLogger(__name__)

# benchmark-local, not in Config: this is eval data, not part of the production pipeline's settings
PUBMEDQA_DIR = Config.project_root / "Data" / "pubmedqa"


def _download_pubmedqa() -> tuple[dict[str, str], dict[str, str], dict[str, list[str]]]:
    dataset = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")

    corpus: dict[str, str] = {}
    queries: dict[str, str] = {}
    qrels: dict[str, list[str]] = {}

    for row in dataset:
        # Extract the pubid
        pubid = str(row["pubid"])
        # Add each value to the proper dict
        corpus[pubid] = " ".join(row["context"]["contexts"])
        queries[pubid] = row["question"]
        qrels[pubid] = [pubid]

    return corpus, queries, qrels


def _save(corpus: dict, queries: dict, qrels: dict, path: Path) -> None:
    # make the directory
    path.mkdir(parents=True, exist_ok=True)
    # loop on each dict of data
    for name, data in (("corpus", corpus), ("queries", queries), ("qrels", qrels)):
        # open the filenamme as json
        with open(path / f"{name}.json", "w", encoding="utf-8") as fh:
            # dump the data inside the josn file
            json.dump(data, fh, indent=2, ensure_ascii=False)
    logger.info(f"Saved {len(corpus)} documents, {len(queries)} queries to {path}")


def _load_cached(path: Path) -> tuple[dict, dict, dict] | None:
    filepaths = [path / f"{name}.json" for name in ("corpus", "queries", "qrels")]
    if not all(fp.exists() for fp in filepaths):
        return None

    loaded = []
    for fp in filepaths:
        with open(fp, "r", encoding="utf-8") as fh:
            loaded.append(json.load(fh))
    return tuple(loaded)


def get_pubmedqa_data(path: Path = PUBMEDQA_DIR) -> tuple[dict[str, str], dict[str, str], dict[str, list[str]]]:
    # get the data from the json file (cach)
    cached = _load_cached(path)
    if cached is not None:
        logger.info(f"Loaded cached PubMedQA data from {path}")
        return cached

    # if there is no cach , get the data from datasets
    logger.info("No cached PubMedQA data found; downloading pqa_labeled...")
    corpus, queries, qrels = _download_pubmedqa()
    # save to the .json files
    _save(corpus, queries, qrels, path)
    return corpus, queries, qrels


if __name__ == "__main__":
    Config.configure_logging()
    corpus, queries, qrels = get_pubmedqa_data()
    logger.info(f"{len(corpus)} documents, {len(queries)} queries, {len(qrels)} qrels")
