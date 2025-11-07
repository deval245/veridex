from typing import Optional, Dict, Any
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
from functools import lru_cache


class LLMConfig(BaseModel):
    provider: str = "openai"
    model: str = "gpt-4-turbo-preview"
    temperature: float = 0.0
    max_tokens: int = 2000
    timeout: int = 30


class AgentConfig(BaseModel):
    timeout: int = 5000
    max_retries: int = 3
    backoff_multiplier: float = 2.0


class CacheConfig(BaseModel):
    enabled: bool = True
    ttl: int = 3600
    max_size: int = 1000


class Settings(BaseSettings):
    project_name: str = "VERIDEX"
    environment: str = "development"
    log_level: str = "INFO"
    
    openai_api_key: Optional[str] = None
    tmdb_api_key: Optional[str] = None
    
    llm: LLMConfig = Field(default_factory=LLMConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    
    class Config:
        env_file = ".env"
        env_nested_delimiter = "__"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    return Settings()

