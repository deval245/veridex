import asyncio
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime

from src.adapters.base import ContentRecord


@dataclass
class BaselineResult:
    content_id: str
    title: str
    region: str
    predicted_rating: Optional[str]
    confidence: float
    method: str
    metadata: Dict[str, Any]


class RuleBasedBaseline:
    
    RATING_MAP = {
        "US": {
            "adult": "R",
            "violence_high": "R",
            "violence_med": "PG-13",
            "family": "PG",
            "animation": "G"
        },
        "GB": {
            "adult": "18",
            "violence_high": "18",
            "violence_med": "12",
            "family": "PG",
            "animation": "U"
        },
        "DE": {
            "adult": "18",
            "violence_high": "18",
            "violence_med": "12",
            "family": "6",
            "animation": "0"
        }
    }
    
    GENRE_CATEGORIES = {
        "violence_high": ["war", "crime", "thriller"],
        "violence_med": ["action"],
        "family": ["family", "comedy"],
        "animation": ["animation"]
    }
    
    async def predict(
        self,
        record: ContentRecord,
        region: str = "US"
    ) -> BaselineResult:
        
        genres_lower = [g.lower() for g in record.genres]
        
        category = self._categorize(record, genres_lower)
        rating_map = self.RATING_MAP.get(region, self.RATING_MAP["US"])
        predicted_rating = rating_map.get(category, "PG-13")
        
        return BaselineResult(
            content_id=record.content_id,
            title=record.title,
            region=region,
            predicted_rating=predicted_rating,
            confidence=0.6,
            method="rule_based",
            metadata={"category": category, "genres": record.genres}
        )
    
    def _categorize(self, record: ContentRecord, genres_lower: List[str]) -> str:
        
        if record.metadata.get("adult"):
            return "adult"
        
        for category, keywords in self.GENRE_CATEGORIES.items():
            if any(kw in genres_lower for kw in keywords):
                return category
        
        return "default"
    
    async def predict_batch(
        self,
        records: List[ContentRecord],
        regions: List[str]
    ) -> List[BaselineResult]:
        
        results = []
        for record in records:
            for region in regions:
                result = await self.predict(record, region)
                results.append(result)
        return results


class LLMBaseline:
    
    PROMPT_TEMPLATE = """Analyze this movie and predict its content rating for {region}.

Movie: {title}
Genres: {genres}
Overview: {overview}

Predict the rating (e.g., for US: G, PG, PG-13, R, NC-17).
Return ONLY the rating code, no explanation."""
    
    def __init__(self, llm_provider: Optional[Any] = None):
        self.llm = llm_provider
    
    async def predict(
        self,
        record: ContentRecord,
        region: str = "US"
    ) -> BaselineResult:
        
        if not self.llm:
            return BaselineResult(
                content_id=record.content_id,
                title=record.title,
                region=region,
                predicted_rating=None,
                confidence=0.0,
                method="llm_not_available",
                metadata={}
            )
        
        prompt = self.PROMPT_TEMPLATE.format(
            region=region,
            title=record.title,
            genres=", ".join(record.genres[:3]),
            overview=record.metadata.get("overview", "")[:200]
        )
        
        try:
            response = await self.llm.generate(prompt)
            predicted_rating = response.strip()
            
            return BaselineResult(
                content_id=record.content_id,
                title=record.title,
                region=region,
                predicted_rating=predicted_rating,
                confidence=0.7,
                method="llm_baseline",
                metadata={"prompt_tokens": len(prompt.split())}
            )
        except Exception as e:
            return BaselineResult(
                content_id=record.content_id,
                title=record.title,
                region=region,
                predicted_rating=None,
                confidence=0.0,
                method="llm_error",
                metadata={"error": str(e)}
            )
    
    async def predict_batch(
        self,
        records: List[ContentRecord],
        regions: List[str]
    ) -> List[BaselineResult]:
        
        results = []
        for record in records:
            for region in regions:
                result = await self.predict(record, region)
                results.append(result)
        return results

