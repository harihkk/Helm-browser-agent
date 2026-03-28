"""
Centralized Configuration
========================
"""

import os
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    # Groq
    groq_api_key: str = Field(default="", alias="GROQ_API_KEY")
    groq_model: str = Field(default="llama-3.3-70b-versatile", alias="GROQ_MODEL")
    groq_eval_model: str = Field(default="llama-3.1-8b-instant", alias="GROQ_EVAL_MODEL")

    # AI Provider
    ai_provider: str = Field(default="groq", alias="AI_PROVIDER")
    ollama_base_url: str = Field(default="http://localhost:11434", alias="OLLAMA_BASE_URL")
    ollama_model: str = Field(default="llama3.1", alias="OLLAMA_MODEL")
    gemini_api_key: str = Field(default="", alias="GEMINI_API_KEY")

    # Browser
    browser_headless: bool = Field(default=False, alias="BROWSER_HEADLESS")
    max_contexts: int = Field(default=3, alias="MAX_CONTEXTS")

    # Task
    max_steps: int = Field(default=25, alias="MAX_STEPS")
    confidence_threshold: float = Field(default=0.6, alias="CONFIDENCE_THRESHOLD")

    # Database
    database_url: str = Field(default="sqlite:///./agent_memory.db", alias="DATABASE_URL")

    # Server
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    host: str = Field(default="0.0.0.0", alias="HOST")
    port: int = Field(default=8000, alias="PORT")

    # Paths
    screenshots_dir: str = Field(default="./screenshots", alias="SCREENSHOTS_DIR")

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


def get_settings() -> Settings:
    return Settings()
