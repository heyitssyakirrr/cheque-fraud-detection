import os
from dataclasses import dataclass, field
from functools import lru_cache

from dotenv import load_dotenv

load_dotenv()


@dataclass
class Settings:
    gateway_url: str = field(default_factory=lambda: os.environ.get("GATEWAY_URL", "http://127.0.0.1:8000"))
    gateway_api_key: str = field(default_factory=lambda: os.environ.get("GATEWAY_API_KEY", ""))


@lru_cache
def get_settings() -> Settings:
    return Settings()