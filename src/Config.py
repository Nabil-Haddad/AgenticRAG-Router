import logging
import os
from pathlib import Path
from dotenv import load_dotenv


load_dotenv()

class Config:
    # Keys
    OPEN_API_KEY = os.getenv("OPENAI_API_KEY", "")
    COHERE_API_KEY = os.getenv("COHERE_API_KEY", "")
    ENVIRONMENT : str = os.getenv("ENVIRONMENT", "development")
    LOG_LEVEL : str = os.getenv("LOG_LEVEL", "INFO")

    # paths
    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / "Data" / "raw"
    output_dir = project_root / "Data" / "processed"
    
    IS_PRODUCTION : bool = os.getenv("ENVIRONMENT", "development") == "production"

    @classmethod
    def configure_logging(cls) -> None:
        logging.basicConfig(
            level=getattr(logging, cls.LOG_LEVEL.upper(), logging.INFO),
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        )

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

