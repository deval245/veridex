from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
from src.adapters.base import ContentRecord


@dataclass
class ValidationResult:
    content_id: str
    title: str
    region: str
    status: str  # "pass", "fail", "warning"
    expected_rating: Optional[str]
    actual_rating: Optional[str]
    confidence: float
    reasoning: str
    violations: List[str]
    metadata: Dict[str, Any]
    validated_at: datetime
    data_freshness: Optional[Dict[str, Any]] = None


class UniversalContentValidator:
    """
    Universal validator for ANY OTT platform
    Works with Netflix, Disney+, Hulu, Amazon Prime, etc.
    """
    
    def __init__(self):
        self.rating_hierarchies = {
            "US-MPAA": ["G", "PG", "PG-13", "R", "NC-17"],
            "GB-BBFC": ["U", "PG", "12", "12A", "15", "18", "R18"],
            "DE-FSK": ["0", "6", "12", "16", "18"]
        }
        
        self.content_rules = self._load_rules()
    
    def _load_rules(self) -> Dict[str, List[Dict[str, Any]]]:
        """Load universal content rating rules"""
        return {
            "violence": [
                {
                    "level": "none",
                    "allowed_ratings": ["G", "U", "0"],
                    "description": "No violence"
                },
                {
                    "level": "mild",
                    "allowed_ratings": ["G", "PG", "U", "PG", "0", "6"],
                    "description": "Mild cartoon violence"
                },
                {
                    "level": "moderate",
                    "allowed_ratings": ["PG-13", "12", "12A", "12"],
                    "description": "Moderate violence, no graphic detail"
                },
                {
                    "level": "strong",
                    "allowed_ratings": ["R", "15", "16"],
                    "description": "Strong violence, some gore"
                },
                {
                    "level": "extreme",
                    "allowed_ratings": ["NC-17", "18", "18"],
                    "description": "Extreme graphic violence"
                }
            ],
            "language": [
                {
                    "level": "none",
                    "allowed_ratings": ["G", "U", "0"],
                    "description": "No profanity"
                },
                {
                    "level": "mild",
                    "allowed_ratings": ["PG", "PG", "6"],
                    "description": "Mild language"
                },
                {
                    "level": "moderate",
                    "allowed_ratings": ["PG-13", "12", "12"],
                    "description": "Moderate profanity"
                },
                {
                    "level": "strong",
                    "allowed_ratings": ["R", "15", "16", "18"],
                    "description": "Strong profanity throughout"
                }
            ],
            "sexual_content": [
                {
                    "level": "none",
                    "allowed_ratings": ["G", "PG", "U", "PG", "0", "6"],
                    "description": "No sexual content"
                },
                {
                    "level": "mild",
                    "allowed_ratings": ["PG-13", "12", "12"],
                    "description": "Brief innuendo or kissing"
                },
                {
                    "level": "moderate",
                    "allowed_ratings": ["R", "15", "16"],
                    "description": "Sexual situations, partial nudity"
                },
                {
                    "level": "explicit",
                    "allowed_ratings": ["NC-17", "18", "18"],
                    "description": "Explicit sexual content"
                }
            ]
        }
    
    async def validate_content(
        self,
        record: ContentRecord,
        region: str = "US"
    ) -> ValidationResult:
        """Validate a single content record"""
        
        actual_rating = record.get_rating(region)
        violations = []
        
        if not actual_rating:
            expected_rating = await self._infer_expected_rating(record, region)
            return ValidationResult(
                content_id=record.content_id,
                title=record.title,
                region=region,
                status="pass",
                expected_rating=expected_rating,
                actual_rating=None,
                confidence=0.70,
                reasoning=f"No official rating for {region}. AI predicted: {expected_rating} (based on genres: {', '.join(record.genres[:3])})",
                violations=[],
                metadata=record.metadata,
                validated_at=datetime.now()
            )
        
        expected_rating = await self._infer_expected_rating(record, region)
        
        if actual_rating == expected_rating:
            status = "pass"
            reasoning = f"Rating {actual_rating} matches expected rating for content"
        else:
            status = "fail"
            reasoning = f"Rating mismatch: expected {expected_rating}, got {actual_rating}"
            violations.append(f"rating_mismatch_{expected_rating}_vs_{actual_rating}")
        
        if record.metadata.get("adult") and actual_rating in ["G", "PG", "U", "0", "6"]:
            status = "fail"
            violations.append("adult_content_with_family_rating")
            reasoning += " | Adult content cannot have family-friendly rating"
        
        confidence = 0.85 if status == "pass" else 0.75
        
        return ValidationResult(
            content_id=record.content_id,
            title=record.title,
            region=region,
            status=status,
            expected_rating=expected_rating,
            actual_rating=actual_rating,
            confidence=confidence,
            reasoning=reasoning,
            violations=violations,
            metadata=record.metadata,
            validated_at=datetime.now()
        )
    
    async def _infer_expected_rating(
        self,
        record: ContentRecord,
        region: str
    ) -> str:
        """Infer expected rating based on content metadata"""
        
        adult = record.metadata.get("adult", False)
        vote_average = record.metadata.get("vote_average", 5.0)
        
        genres = [g.lower() for g in record.genres]
        
        if adult:
            return "R" if region == "US" else "18"
        
        if any(g in ["horror", "thriller"] for g in genres):
            return "PG-13" if region == "US" else "12"
        
        if any(g in ["action", "crime", "war"] for g in genres):
            return "PG-13" if region == "US" else "12"
        
        if any(g in ["animation", "family"] for g in genres):
            return "PG" if region == "US" else "U"
        
        return "PG-13" if region == "US" else "12"
    
    async def validate_batch(
        self,
        records: List[ContentRecord],
        regions: Optional[List[str]] = None
    ) -> List[ValidationResult]:
        """Validate multiple content records"""
        
        if not regions:
            regions = ["US"]
        
        results = []
        
        for record in records:
            for region in regions:
                result = await self.validate_content(record, region)
                results.append(result)
        
        return results
    
    def generate_summary(self, results: List[ValidationResult]) -> Dict[str, Any]:
        """Generate summary statistics from validation results"""
        
        total = len(results)
        passed = len([r for r in results if r.status == "pass"])
        failed = len([r for r in results if r.status == "fail"])
        warnings = len([r for r in results if r.status == "warning"])
        
        avg_confidence = sum([r.confidence for r in results]) / total if total > 0 else 0
        
        violation_counts = {}
        for result in results:
            for violation in result.violations:
                violation_counts[violation] = violation_counts.get(violation, 0) + 1
        
        return {
            "total_validations": total,
            "passed": passed,
            "failed": failed,
            "warnings": warnings,
            "pass_rate": passed / total if total > 0 else 0,
            "fail_rate": failed / total if total > 0 else 0,
            "average_confidence": avg_confidence,
            "top_violations": sorted(
                violation_counts.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]
        }

