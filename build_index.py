import logging
from src.retrieval_arena.config import Config
from src.retrieval_arena.index.index import build_index

logger = logging.getLogger(__name__)



def main():
    chunks_number = build_index(path=Config.data_dir)
    if chunks_number is None:
        logger.info("Nothing to index; data unchanged since the last run.")
    else:
        logger.info(f"{chunks_number} have been stored in the database")


if __name__ == "__main__":
    Config.configure_logging()
    main()
