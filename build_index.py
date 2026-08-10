import logging
from src.Config import Config
from src.Index import build_index

logger = logging.getLogger(__name__)



def main():
    chunks_number = build_index(path=Config.data_dir)
    logger.info(f"{chunks_number} have been stored in the database")


if __name__ == "__main__":
    Config.configure_logging()
    main()
