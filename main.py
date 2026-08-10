import logging
from src.Config import Config

logger = logging.getLogger(__name__)


def main():
    logger.info(f"Data directory: {Config.data_dir}")


if __name__ == "__main__":
    Config.configure_logging()
    main()
