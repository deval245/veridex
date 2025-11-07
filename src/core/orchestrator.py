from typing import List, Dict, Any, Optional
from src.agents.base import Agent
from src.agents.validation import ValidationAgent
from src.core.types import Task, Result, Context, TaskStatus
from src.core.llm import LLMProvider
from src.config import Settings
import asyncio


class Orchestrator:
    def __init__(self, llm_provider: LLMProvider, settings: Settings):
        self.llm = llm_provider
        self.settings = settings
        self.agents: List[Agent] = self._init_agents()
    
    def _init_agents(self) -> List[Agent]:
        return [
            ValidationAgent(self.llm, self.settings),
        ]
    
    def _select_agent(self, task: Task) -> Optional[Agent]:
        for agent in self.agents:
            if agent.can_handle(task):
                return agent
        return None
    
    async def execute_task(self, task: Task, context: Optional[Context] = None) -> Result:
        ctx = context or Context()
        
        agent = self._select_agent(task)
        if not agent:
            return Result(
                success=False,
                error=f"No agent found for task type: {task.task_type}"
            )
        
        task.status = TaskStatus.RUNNING
        result = await agent.run(task, ctx)
        task.status = TaskStatus.COMPLETED if result.success else TaskStatus.FAILED
        
        return result
    
    async def execute_batch(
        self,
        tasks: List[Task],
        context: Optional[Context] = None,
        max_concurrency: int = 10
    ) -> List[Result]:
        semaphore = asyncio.Semaphore(max_concurrency)
        
        async def execute_with_limit(task: Task) -> Result:
            async with semaphore:
                return await self.execute_task(task, context)
        
        results = await asyncio.gather(
            *[execute_with_limit(task) for task in tasks],
            return_exceptions=True
        )
        
        return [
            r if isinstance(r, Result) else Result(success=False, error=str(r))
            for r in results
        ]

