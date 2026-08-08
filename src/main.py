import logging
from Config import Config

logger = logging.getLogger(__name__)


def main():
    logger.info(f"Data directory: {Config.data_dir}")
    #print(text1)
if __name__ == "__main__":
    Config.configure_logging()
    main()
