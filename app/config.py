import os
from dataclasses import dataclass, field
from functools import lru_cache

from dotenv import load_dotenv

load_dotenv()


@dataclass
class Settings:
    llm_base_url: str = field(default_factory=lambda: os.environ.get("LLM_BASE_URL", "").rstrip("/"))
    llm_extract_path: str = field(default_factory=lambda: os.environ.get("LLM_EXTRACT_PATH", "/foundry/chat/completions"))
    # strip(): a stray space or Windows line ending makes the header illegal
    helper_id: str = field(default_factory=lambda: os.environ.get("HELPER_ID", "").strip())
    llm_model: str = field(default_factory=lambda: os.environ.get("LLM_MODEL", "gpt-6-luna"))
    llm_max_tokens: int = field(default_factory=lambda: int(os.environ.get("LLM_MAX_TOKENS", "6000")))


@lru_cache
def get_settings() -> Settings:
    return Settings()
