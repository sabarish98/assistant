"""Configuration management for AI Research Assistant."""

from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppConfig(BaseSettings):
    """Application configuration with environment variable support."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )
    
    # Application settings
    app_name: str = Field(default="AI Research Assistant")
    app_version: str = Field(default="1.0.0")
    
    # Ollama settings
    ollama_model: str = Field(default="gemma4:e4b")
    ollama_base_url: str = Field(default="http://localhost:11434")
    ollama_temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    
    # Vector database settings
    chroma_persist_directory: str = Field(default="./data/chroma_db")
    
    # Logging settings
    log_level: str = Field(default="INFO")
    log_file: str = Field(default="./logs/research_assistant.log")
    
    # Token and performance settings
    max_tokens: Optional[int] = Field(default=None)
    request_timeout: int = Field(default=60)
    
    def __repr__(self) -> str:
        return f"AppConfig(model={self.ollama_model}, temp={self.ollama_temperature})"


# Global configuration instance
config = AppConfig()