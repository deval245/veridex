from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
from openai import AsyncOpenAI
from anthropic import AsyncAnthropic
from src.config import Settings


class LLMProvider(ABC):
    @abstractmethod
    async def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 2000
    ) -> str:
        pass


class OpenAIProvider(LLMProvider):
    def __init__(self, settings: Settings):
        self.client = AsyncOpenAI(api_key=settings.openai_api_key)
        self.model = settings.llm.model
        self.default_temperature = settings.llm.temperature
        self.default_max_tokens = settings.llm.max_tokens
    
    async def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature or self.default_temperature,
            max_tokens=max_tokens or self.default_max_tokens
        )
        return response.choices[0].message.content


class ClaudeProvider(LLMProvider):
    def __init__(self, settings: Settings):
        self.client = AsyncAnthropic(api_key=settings.openai_api_key)
        self.model = "claude-3-sonnet-20240229"
        self.default_temperature = settings.llm.temperature
        self.default_max_tokens = settings.llm.max_tokens
    
    async def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        response = await self.client.messages.create(
            model=self.model,
            system=system or "",
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature or self.default_temperature,
            max_tokens=max_tokens or self.default_max_tokens
        )
        return response.content[0].text


def get_llm_provider(settings: Settings) -> LLMProvider:
    providers = {
        "openai": OpenAIProvider,
        "claude": ClaudeProvider,
    }
    provider_class = providers.get(settings.llm.provider)
    if not provider_class:
        raise ValueError(f"Unknown LLM provider: {settings.llm.provider}")
    return provider_class(settings)

