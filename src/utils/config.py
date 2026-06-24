import os
from pathlib import Path

from dotenv import load_dotenv

ENV_LOCAL = Path(__file__).parents[2] / ".env.local"
ENV_EXAMPLE = Path(__file__).parents[2] / ".env.example"


def load_env() -> None:
    if ENV_LOCAL.exists():
        load_dotenv(ENV_LOCAL)
    else:
        load_dotenv(ENV_EXAMPLE)


def get_kaggle_config() -> dict[str, str]:
    return {
        "username": os.getenv("KAGGLE_USERNAME", ""),
        "key": os.getenv("KAGGLE_KEY", ""),
    }


load_env()
