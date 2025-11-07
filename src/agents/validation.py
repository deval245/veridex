from typing import Dict, Any
from src.agents.base import Agent
from src.core.types import Task, Result, Context, ValidationStatus, ValidationResult
import json


class ValidationAgent(Agent):
    def can_handle(self, task: Task) -> bool:
        return task.task_type == "validate"
    
    async def execute(self, task: Task, context: Context) -> Result:
        entity_id = task.payload.get("entity_id")
        expected = task.payload.get("expected")
        actual = task.payload.get("actual")
        domain = task.payload.get("domain", "content_rating")
        
        if domain == "content_rating":
            return await self._validate_content_rating(
                entity_id, expected, actual, context
            )
        
        return Result(
            success=False,
            error=f"Unknown validation domain: {domain}"
        )
    
    async def _validate_content_rating(
        self,
        entity_id: str,
        expected: str,
        actual: str,
        context: Context
    ) -> Result:
        if expected == actual:
            validation = ValidationResult(
                entity_id=entity_id,
                status=ValidationStatus.PASS,
                expected=expected,
                actual=actual,
                confidence=1.0,
                reason="Rating matches expected value"
            )
            return Result(success=True, data={"validation": validation.__dict__})
        
        prompt = f"""Analyze this content rating mismatch:

Entity: {entity_id}
Expected Rating: {expected}
Actual Rating: {actual}

Determine:
1. Is this a critical mismatch that violates policy?
2. What is the severity? (low/medium/high)
3. Explain why the ratings differ in one sentence.

Respond in JSON format:
{{
    "critical": boolean,
    "severity": "low|medium|high",
    "explanation": "one sentence explanation"
}}"""
        
        response = await self.llm.generate(prompt)
        
        try:
            analysis = json.loads(response)
            status = ValidationStatus.FAIL if analysis.get("critical") else ValidationStatus.PASS
            
            validation = ValidationResult(
                entity_id=entity_id,
                status=status,
                expected=expected,
                actual=actual,
                confidence=0.85,
                reason=analysis.get("explanation", "Rating mismatch detected"),
                metadata={"severity": analysis.get("severity")}
            )
            
            return Result(success=True, data={"validation": validation.__dict__})
        except json.JSONDecodeError:
            return Result(
                success=False,
                error="Failed to parse LLM response"
            )

