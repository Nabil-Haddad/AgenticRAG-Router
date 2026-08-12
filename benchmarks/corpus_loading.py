from src.retrieval_arena.index.index import get_collection
from src.retrieval_arena.config import Config
from pathlib import Path
import json
import logging
import random
import statistics
import string
import tempfile
import time
from typing import Callable

import chromadb

logger = logging.getLogger(__name__)


def load_corpus_json(path: Path) -> list[list[str]]:
    try:
        with open(path, "r", encoding="utf-8") as file_handle:
            raw_json_data = json.load(file_handle)
    except Exception as error_exception:
        logger.error("Something went wrong when opening the JSON file.")
        print(error_exception)
        return []

    return [item["text"].lower().split() for item in raw_json_data]


def load_corpus_db(collection) -> list[list[str]]:
    try:
        total_items = collection.count()
        all_retrieved_data = collection.get(
            limit=total_items,
            include=["documents"]
        )
        return [
            document.lower().split() for document in all_retrieved_data["documents"]
        ]
    except Exception as error_exception:
        logger.error("Something went wrong when trying to open the database.")
        print(error_exception)
        return []


def compare(func_a: Callable,func_b: Callable,iterations: int = 10,args_a: tuple = (),kwargs_a: dict | None = None,args_b: tuple = (),kwargs_b: dict | None = None,) -> dict[str, float]:
    kwargs_a = kwargs_a or {}
    kwargs_b = kwargs_b or {}
    name_a, name_b = func_a.__name__, func_b.__name__

    times_a: list[float] = []
    times_b: list[float] = []

    # interleave trials (a, b, a, b, ...) instead of running all of a then
    # all of b - otherwise time-varying system noise (cache warmup,
    # background load) can systematically favor whichever block runs first
    for _ in range(iterations):
        start = time.perf_counter()
        func_a(*args_a, **kwargs_a)
        times_a.append(time.perf_counter() - start)

        start = time.perf_counter()
        func_b(*args_b, **kwargs_b)
        times_b.append(time.perf_counter() - start)

    mean_a, mean_b = statistics.mean(times_a), statistics.mean(times_b)
    stdev_a = statistics.stdev(times_a) if iterations > 1 else 0.0
    stdev_b = statistics.stdev(times_b) if iterations > 1 else 0.0

    print(f"{name_a}: mean={mean_a:.6f}s stdev={stdev_a:.6f}s")
    print(f"{name_b}: mean={mean_b:.6f}s stdev={stdev_b:.6f}s")

    if mean_a < mean_b:
        winner, diff = name_a, mean_b - mean_a
    else:
        winner, diff = name_b, mean_a - mean_b
    print(f"{winner} is faster by an average of {diff:.6f} seconds per run\n")

    return {"mean_a": mean_a, "mean_b": mean_b, "stdev_a": stdev_a, "stdev_b": stdev_b}


def make_fake_chunk(i: int) -> dict:
    text = " ".join("".join(random.choices(string.ascii_lowercase, k=6)) for _ in range(50))
    return {
        "id": f"chunk_{i:06d}",
        "text": text,
        "source": f"doc_{i % 20}.pdf",
        "page_start": 1,
        "page_end": 1,
        "token_count": 50,
        "metadata": {},
    }


def build_synthetic_corpus(size: int, json_path: Path, collection, client) -> None:
    chunks = [make_fake_chunk(i) for i in range(size)]
    json_path.write_text(json.dumps(chunks), encoding="utf-8")

    # load_corpus_db only reads "documents", so the embeddings themselves
    # don't need to be meaningful - just present, with a consistent dimension.
    # chromadb caps how many records a single add() call can take, so batch it.
    max_batch_size = client.get_max_batch_size()
    for start in range(0, size, max_batch_size):
        batch = chunks[start:start + max_batch_size]
        collection.add(
            ids=[c["id"] for c in batch],
            documents=[c["text"] for c in batch],
            embeddings=[[0.0] * 8 for _ in batch],
        )


def run_scaling_benchmark(sizes: list[int]) -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        client = chromadb.PersistentClient(path=str(tmp_path / "vectordb"))

        for size in sizes:
            collection = client.get_or_create_collection(name=f"bench_{size}")
            json_path = tmp_path / f"chunks_{size}.json"
            build_synthetic_corpus(size, json_path, collection, client)

            print(f"--- synthetic corpus size: {size} chunks ---")
            compare(
                func_a=load_corpus_json,
                func_b=load_corpus_db,
                iterations=10,
                kwargs_a={"path": json_path},
                kwargs_b={"collection": collection},
            )


def main() -> None:
    print("real corpus")
    corpus_path = Config.chunks_dir / "chunks.json"
    compare(
        func_a=load_corpus_json,
        func_b=load_corpus_db,
        iterations=10,
        kwargs_a={"path": corpus_path},
        kwargs_b={"collection": get_collection()},
    )

    print("synthetic scaling benchmark (temporary, isolated DB)")
    run_scaling_benchmark(sizes=[150, 1500, 15000])


if __name__ == "__main__":
    Config.configure_logging()
    main()
