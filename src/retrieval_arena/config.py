import logging
import os
import warnings
from pathlib import Path
from dotenv import load_dotenv

# noisy third-party loggers that shouldn't drown out the app's own output
# at the default log level - set before any of these libraries are imported
# elsewhere, since huggingface_hub reads this env var once at import time
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")


load_dotenv()

class Config:
    # Keys
    OPEN_API_KEY = os.getenv("OPENAI_API_KEY", "")
    COHERE_API_KEY = os.getenv("COHERE_API_KEY", "")
    ENVIRONMENT : str = os.getenv("ENVIRONMENT", "development")
    LOG_LEVEL : str = os.getenv("LOG_LEVEL", "INFO")

    # paths
    project_root = Path(__file__).resolve().parent.parent.parent
    data_dir = project_root / "Data" / "raw"
    output_dir = project_root / "Data" / "processed"

    # chunking
    chunks_dir = project_root / "Data" / "chunks"
    manifest_path = project_root / "Data" / "manifest.json"
    CHUNK_SIZE : int = int(os.getenv("CHUNK_SIZE", "400"))      # tokens
    CHUNK_OVERLAP : int = int(os.getenv("CHUNK_OVERLAP", "60")) # tokens

    # models
    EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

    # vector db
    VECTORDB_DIR = project_root / "vectordb"
    COLLECTION_NAME = "chunks"

    # retrieval
    RETRIEVAL_METHOD: str = os.getenv("RETRIEVAL_METHOD", "hybrid")
    
    
    IS_PRODUCTION : bool = os.getenv("ENVIRONMENT", "development") == "production"

    @classmethod
    def configure_logging(cls) -> None:
        level = getattr(logging, cls.LOG_LEVEL.upper(), logging.INFO)
        logging.basicConfig(
            level=level,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        )

        # keep vendor request/progress chatter out of normal output; LOG_LEVEL=DEBUG
        # still surfaces it, since that's an explicit ask to see everything
        if level > logging.DEBUG:
            for noisy_logger in ("httpx", "httpcore", "huggingface_hub", "urllib3", "filelock"):
                logging.getLogger(noisy_logger).setLevel(logging.ERROR)
            warnings.filterwarnings("ignore", message=".*unauthenticated requests.*")
            warnings.filterwarnings("ignore", message=".*CUDA initialization.*")

    @classmethod
    def validate(cls)-> None:
        required = ["OPEN_API_KEY"]
        missing = [key for key in required if not getattr(cls, key)]

        if missing:
            raise EnvironmentError(f"Missing required environment variables: {missing}\n")

    @classmethod
    def summary(cls)-> dict:
        return {
            "OPEN_API_KEY" : "set" if cls.OPEN_API_KEY else "MISSING",
            "COHERE_API_KEY" : "set" if cls.COHERE_API_KEY else "not set",
            "ENVIRONMENT" : cls.ENVIRONMENT,
            "LOG_LEVEL" : cls.LOG_LEVEL,
            "IS_PRODUCTION" : cls.IS_PRODUCTION,
        }

