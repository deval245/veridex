import pytest
from src.config import Settings, get_settings


def test_settings_defaults():
    settings = Settings()
    assert settings.project_name == "VERIDEX"
    assert settings.environment == "development"
    assert settings.log_level == "INFO"


def test_llm_config_defaults():
    settings = Settings()
    assert settings.llm.provider == "openai"
    assert settings.llm.model == "gpt-4-turbo-preview"
    assert settings.llm.temperature == 0.0
    assert settings.llm.max_tokens == 2000


def test_agent_config_defaults():
    settings = Settings()
    assert settings.agent.timeout == 5000
    assert settings.agent.max_retries == 3
    assert settings.agent.backoff_multiplier == 2.0


def test_cache_config_defaults():
    settings = Settings()
    assert settings.cache.enabled is True
    assert settings.cache.ttl == 3600


def test_get_settings_cached():
    settings1 = get_settings()
    settings2 = get_settings()
    assert settings1 is settings2

