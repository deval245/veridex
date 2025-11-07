from typing import List, Optional, Dict, Any
from src.config import Settings, get_settings
from src.core.llm import get_llm_provider, LLMProvider
from src.core.orchestrator import Orchestrator
from src.core.types import Task, Result, ValidationResult, Context
from src.adapters.tmdb import TMDbAdapter
import pandas as pd
from uuid import uuid4


class VERIDEX:
    def __init__(
        self,
        settings: Optional[Settings] = None,
        tmdb_api_key: Optional[str] = None
    ):
        self.settings = settings or get_settings()
        if tmdb_api_key:
            self.settings.tmdb_api_key = tmdb_api_key
        
        self.llm_provider: LLMProvider = get_llm_provider(self.settings)
        self.orchestrator = Orchestrator(self.llm_provider, self.settings)
        self.tmdb: Optional[TMDbAdapter] = None
    
    def _init_tmdb(self):
        if not self.tmdb and self.settings.tmdb_api_key:
            self.tmdb = TMDbAdapter(self.settings.tmdb_api_key)
    
    async def validate_content_rating(
        self,
        movie_id: str,
        expected_rating: str,
        country: str = "US"
    ) -> ValidationResult:
        self._init_tmdb()
        
        if not self.tmdb:
            raise ValueError("TMDb API key not configured")
        
        ratings = await self.tmdb.get_content_ratings(movie_id)
        actual_rating = next(
            (r["rating"] for r in ratings if r["country"] == country),
            None
        )
        
        if not actual_rating:
            return ValidationResult(
                entity_id=movie_id,
                status="error",
                expected=expected_rating,
                actual=None,
                reason=f"No rating found for country: {country}"
            )
        
        task = Task(
            task_id=str(uuid4()),
            task_type="validate",
            payload={
                "entity_id": movie_id,
                "expected": expected_rating,
                "actual": actual_rating,
                "domain": "content_rating"
            }
        )
        
        result = await self.orchestrator.execute_task(task)
        
        if result.success and result.data:
            validation_data = result.data.get("validation", {})
            return ValidationResult(**validation_data)
        
        return ValidationResult(
            entity_id=movie_id,
            status="error",
            expected=expected_rating,
            actual=actual_rating,
            reason=result.error or "Validation failed"
        )
    
    async def validate_batch(
        self,
        items: List[Dict[str, Any]],
        country: str = "US"
    ) -> List[ValidationResult]:
        results = []
        for item in items:
            result = await self.validate_content_rating(
                movie_id=item.get("movie_id"),
                expected_rating=item.get("expected_rating"),
                country=country
            )
            results.append(result)
        return results
    
    def results_to_dataframe(self, results: List[ValidationResult]) -> pd.DataFrame:
        data = [
            {
                "entity_id": r.entity_id,
                "status": r.status.value if hasattr(r.status, "value") else r.status,
                "expected": r.expected,
                "actual": r.actual,
                "confidence": r.confidence,
                "reason": r.reason
            }
            for r in results
        ]
        return pd.DataFrame(data)
    
    async def cleanup(self):
        if self.tmdb and self.tmdb.session:
            await self.tmdb.session.close()

