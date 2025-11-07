from abc import ABC, abstractmethod
from typing import Optional
from src.core.types import Task, Result, Context
from src.core.llm import LLMProvider
from src.config import Settings
from src.evaluation.metrics import MetricsCollector, MetricsContext
import asyncio
from functools import wraps


def with_retry(max_attempts: int = 3, backoff: float = 2.0):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        await asyncio.sleep(backoff ** attempt)
            raise last_exception
        return wrapper
    return decorator


class Agent(ABC):
    def __init__(
        self,
        llm_provider: LLMProvider,
        settings: Settings,
        metrics_collector: Optional[MetricsCollector] = None
    ):
        self.llm = llm_provider
        self.settings = settings
        self.name = self.__class__.__name__
        self.metrics = metrics_collector or MetricsCollector()
    
    @abstractmethod
    def can_handle(self, task: Task) -> bool:
        pass
    
    @abstractmethod
    async def execute(self, task: Task, context: Context) -> Result:
        pass
    
    @with_retry(max_attempts=3, backoff=2.0)
    async def run(self, task: Task, context: Optional[Context] = None) -> Result:
        ctx = context or Context()
        
        if not self.can_handle(task):
            return Result(
                success=False,
                error=f"{self.name} cannot handle task type: {task.task_type}"
            )
        
        with MetricsContext(self.metrics, self.name, task.task_type):
            try:
                return await asyncio.wait_for(
                    self.execute(task, ctx),
                    timeout=self.settings.agent.timeout / 1000
                )
            except asyncio.TimeoutError:
                return Result(
                    success=False,
                    error=f"{self.name} execution timeout"
                )
            except Exception as e:
                return Result(
                    success=False,
                    error=f"{self.name} error: {str(e)}"
                )

