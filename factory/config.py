"""Central configuration from env / .env."""
from __future__ import annotations

import os
from dataclasses import dataclass, field

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


@dataclass(frozen=True)
class Settings:
    openai_api_key: str = field(default_factory=lambda: os.environ.get("OPENAI_API_KEY", ""))
    model: str = field(default_factory=lambda: os.environ.get("SDF_MODEL", "gpt-4o-mini"))
    temperature: float = field(default_factory=lambda: float(os.environ.get("SDF_TEMPERATURE", "0.7")))
    timeout: int = field(default_factory=lambda: int(os.environ.get("SDF_TIMEOUT", "30")))
    min_score: int = field(default_factory=lambda: int(os.environ.get("SDF_MIN_SCORE", "6")))
    dedup_threshold: float = field(default_factory=lambda: float(os.environ.get("SDF_DEDUP_THRESHOLD", "0.92")))

    @property
    def has_key(self) -> bool:
        return bool(self.openai_api_key)


SETTINGS = Settings()
