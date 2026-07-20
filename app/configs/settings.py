"""
Pydantic Settings loaded from config.ini + environment variables.

Usage:
    from configs.settings import settings
    settings.TELEGRAM_ACCESS_TOKEN
"""

from pydantic_settings import BaseSettings
from typing import Optional
import configparser
import json
from pathlib import Path


class Settings(BaseSettings):
    # ------------------------------------------------------------------ #
    #  Telegram                                                          #
    # ------------------------------------------------------------------ #
    TELEGRAM_ACCESS_TOKEN: str = "8517370440:AAEhvWStRQlrW9uFA3aCidb3TAQdQ7YcMPI"

    # ------------------------------------------------------------------ #
    #  Azure OpenAI (app.llm)                                       #
    # ------------------------------------------------------------------ #
    CHATGPT_API_KEY: str = "a053ead5-9ec1-4ca3-ac8d-08dd73a28ab7"
    CHATGPT_BASE_URL: str = "https://genai.hkbu.edu.hk/api/v0/rest"
    CHATGPT_MODEL: str = "gpt-5-mini"
    CHATGPT_API_VER: str = "2024-12-01-preview"

    # ------------------------------------------------------------------ #
    #  SiliconFlow / Wan-AI (Image-to-Video)                              #
    # ------------------------------------------------------------------ #
    WAN_AI_API_KEY: str = "sk-mubvhickdwyfyifhjthqkidqthzdxbvpulkzgyhjhftatqxz"
    WAN_AI_BASE_URL: str = "https://api.siliconflow.cn/v1"
    WAN_AI_MODEL: str = "Wan-AI/Wan2.2-I2V-A14B"

    # ------------------------------------------------------------------ #
    #  Milvus Vector DB                                                   #
    # ------------------------------------------------------------------ #
    # TO-DO: Please provide your Milvus connection details.
    # For a local Milvus instance (Docker), use host="milvus", port="19530".
    # For Zilliz Cloud, set the full URI and token.
    MILVUS_HOST: str = "milvus"
    MILVUS_PORT: str = "19530"
    MILVUS_URI: str = ""                    # e.g. "https://<instance>.zillizcloud.com:443"
    MILVUS_TOKEN: str = ""                  # optional, for Zilliz Cloud
    MILVUS_COLLECTION: str = "course_documents"

    # ------------------------------------------------------------------ #
    #  Embedding model (used by LangChain OpenAIEmbeddings)               #
    # ------------------------------------------------------------------ #
    # TO-DO: Provide a valid OpenAI-compatible embedding endpoint + key.
    # For Azure OpenAI embeddings, set:
    #   EMBEDDING_API_KEY, EMBEDDING_BASE_URL, EMBEDDING_MODEL, EMBEDDING_API_VER
    # To use the HKBU genai gateway, reuse the CHATGPT credentials.
    # If your gateway is Azure-format, set EMBEDDING_PROVIDER="azure"
    EMBEDDING_PROVIDER: str = "openai"       # "openai" or "azure"
    EMBEDDING_API_KEY: str = "sk-ol-v1-fd6a031eed2dc47cbd0aded4da1c21dcd418a5968a1f592e2adb2013a6d6a5ff"
    EMBEDDING_BASE_URL: str = "https://api.aliapi.me/v1"
    EMBEDDING_MODEL: str = "bge-m3"
    EMBEDDING_API_VER: str = "2024-12-01-preview"

    # ------------------------------------------------------------------ #
    #  Hybrid Search (BM25 + Dense Vector RRF)                           #
    # ------------------------------------------------------------------ #
    HYBRID_SEARCH_ENABLED: bool = True        # set False to use pure dense only
    HYBRID_DENSE_WEIGHT: float = 0.7          # dense vector weight in RRF (must sum with SPARSE to 1.0)
    HYBRID_SPARSE_WEIGHT: float = 0.3         # BM25 sparse weight in RRF

    # ------------------------------------------------------------------ #
    #  Redis                                                             #
    # ------------------------------------------------------------------ #
    # Password authentication for Redis. Empty string = no auth.
    # Use docker-compose to set the same password in both Redis and app containers.
    REDIS_HOST: str = "redis"
    REDIS_PORT: str = "6379"
    REDIS_PASSWORD: str = ""

    # ------------------------------------------------------------------ #
    #  Admin Authentication (token → role mapping)                       #
    # ------------------------------------------------------------------ #
    # Pass as a JSON string env var, e.g.:
    #   ADMIN_TOKENS={"a7f3c...":"upload","9d3f...":"view"}
    # Roles: "upload" (can upload + view stats) or "view" (stats only)
    ADMIN_TOKENS_JSON: str = "{}"

    @property
    def admin_tokens(self) -> dict[str, str]:
        """Parse the ADMIN_TOKENS_JSON string into a dict of token → role."""
        try:
            return json.loads(self.ADMIN_TOKENS_JSON)
        except (json.JSONDecodeError, TypeError):
            return {}

    # ------------------------------------------------------------------ #
    #  Logging                                                           #
    # ------------------------------------------------------------------ #
    # INFO: metadata only (user msg truncated to 50 chars)
    # DEBUG: full user message + LLM response logged for troubleshooting
    LOG_DIR: str = "/comp7940-lab/logs"
    BOT_LOG_LEVEL: str = "INFO"
    TEMP_DIR: str = "/comp7940-lab/temp"

    class Config:
        env_prefix = ""
        case_sensitive = True

    @classmethod
    def from_ini(cls, path: str = "config.ini") -> "Settings":
        """Load settings from a config.ini file, overlaying env vars."""
        ini = configparser.ConfigParser()
        ini.read(path)

        kwargs: dict[str, str] = {}

        if ini.has_section("TELEGRAM"):
            kwargs["TELEGRAM_ACCESS_TOKEN"] = ini["TELEGRAM"].get("ACCESS_TOKEN", "")

        if ini.has_section("CHATGPT"):
            kwargs["CHATGPT_API_KEY"] = ini["CHATGPT"].get("API_KEY", "")
            kwargs["CHATGPT_BASE_URL"] = ini["CHATGPT"].get("BASE_URL", "")
            kwargs["CHATGPT_MODEL"] = ini["CHATGPT"].get("MODEL", "")
            kwargs["CHATGPT_API_VER"] = ini["CHATGPT"].get("API_VER", "")

        if ini.has_section("WAN_AI"):
            kwargs["WAN_AI_API_KEY"] = ini["WAN_AI"].get("API_KEY", "")
            kwargs["WAN_AI_BASE_URL"] = ini["WAN_AI"].get("BASE_URL", "")
            kwargs["WAN_AI_MODEL"] = ini["WAN_AI"].get("MODEL", "")

        # Read optional [EMBEDDING] section (if present in config.ini)
        if ini.has_section("EMBEDDING"):
            kwargs["EMBEDDING_API_KEY"] = ini["EMBEDDING"].get("API_KEY", "")
            kwargs["EMBEDDING_BASE_URL"] = ini["EMBEDDING"].get("BASE_URL", "")
            kwargs["EMBEDDING_MODEL"] = ini["EMBEDDING"].get("MODEL", "text-embedding-3-small")
            kwargs["EMBEDDING_API_VER"] = ini["EMBEDDING"].get("API_VER", "2024-12-01-preview")
            kwargs["EMBEDDING_PROVIDER"] = ini["EMBEDDING"].get("PROVIDER", "openai")

        # Read optional [APP] section for misc paths
        if ini.has_section("APP"):
            kwargs["TEMP_DIR"] = ini["APP"].get("TEMP_DIR", "/comp7940-lab/temp")
            kwargs["LOG_DIR"] = ini["APP"].get("LOG_DIR", "/comp7940-lab/logs")

        # Merge with env-var overrides (env vars win)
        return cls(**kwargs)


# Module-level singleton – import `settings` everywhere.
settings = Settings.from_ini(str(Path(__file__).resolve().parent.parent.parent / "config.ini"))
