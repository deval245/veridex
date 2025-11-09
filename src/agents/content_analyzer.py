from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum


class AgentRole(Enum):
    CONTENT_CLASSIFIER = "content_classifier"
    RATING_PREDICTOR = "rating_predictor"
    CONSISTENCY_CHECKER = "consistency_checker"
    AGGREGATOR = "aggregator"


@dataclass
class AgentResponse:
    agent_role: str
    analysis: Dict[str, Any]
    confidence: float
    reasoning: str


class ContentAnalyzerAgent:
    
    def __init__(self, llm_provider: Optional[Any] = None):
        self.llm = llm_provider
        self.role = AgentRole.CONTENT_CLASSIFIER
    
    async def analyze(self, content: Dict[str, Any], region: str) -> AgentResponse:
        
        violence_level = self._analyze_violence(content)
        language_level = self._analyze_language(content)
        mature_themes = self._analyze_themes(content)
        
        analysis = {
            "violence_level": violence_level,
            "language_level": language_level,
            "mature_themes": mature_themes,
            "genres": content.get("genres", []),
            "adult_flag": content.get("metadata", {}).get("adult", False)
        }
        
        confidence = self._compute_confidence(analysis)
        reasoning = self._generate_reasoning(analysis)
        
        return AgentResponse(
            agent_role=self.role.value,
            analysis=analysis,
            confidence=confidence,
            reasoning=reasoning
        )
    
    def _analyze_violence(self, content: Dict[str, Any]) -> str:
        genres = [g.lower() for g in content.get("genres", [])]
        
        if any(g in ["war", "crime"] for g in genres):
            return "high"
        elif any(g in ["action", "thriller"] for g in genres):
            return "medium"
        elif any(g in ["animation", "family", "comedy"] for g in genres):
            return "low"
        return "medium"
    
    def _analyze_language(self, content: Dict[str, Any]) -> str:
        genres = [g.lower() for g in content.get("genres", [])]
        
        if any(g in ["crime", "thriller"] for g in genres):
            return "strong"
        elif any(g in ["drama", "action"] for g in genres):
            return "moderate"
        return "mild"
    
    def _analyze_themes(self, content: Dict[str, Any]) -> List[str]:
        genres = [g.lower() for g in content.get("genres", [])]
        themes = []
        
        if "horror" in genres:
            themes.append("fear")
        if any(g in ["war", "crime"] for g in genres):
            themes.append("violence")
        if "romance" in genres:
            themes.append("romance")
        if content.get("metadata", {}).get("adult"):
            themes.append("adult_content")
        
        return themes
    
    def _compute_confidence(self, analysis: Dict[str, Any]) -> float:
        base_confidence = 0.75
        
        if analysis.get("adult_flag"):
            base_confidence += 0.1
        
        if len(analysis.get("genres", [])) > 2:
            base_confidence += 0.05
        
        return min(base_confidence, 0.95)
    
    def _generate_reasoning(self, analysis: Dict[str, Any]) -> str:
        parts = []
        
        parts.append(f"Violence: {analysis['violence_level']}")
        parts.append(f"Language: {analysis['language_level']}")
        
        if analysis['mature_themes']:
            parts.append(f"Themes: {', '.join(analysis['mature_themes'])}")
        
        return " | ".join(parts)


class RatingPredictorAgent:
    
    RATING_MAP = {
        "US": {
            ("high", "strong"): "R",
            ("high", "moderate"): "PG-13",
            ("medium", "strong"): "PG-13",
            ("medium", "moderate"): "PG-13",
            ("low", "mild"): "PG",
            ("low", "none"): "G"
        }
    }
    
    def __init__(self):
        self.role = AgentRole.RATING_PREDICTOR
    
    async def predict(
        self,
        content_analysis: Dict[str, Any],
        region: str
    ) -> AgentResponse:
        
        violence = content_analysis.get("violence_level", "medium")
        language = content_analysis.get("language_level", "moderate")
        
        if content_analysis.get("adult_flag"):
            predicted_rating = "R"
            confidence = 0.9
        else:
            rating_map = self.RATING_MAP.get(region, self.RATING_MAP["US"])
            predicted_rating = rating_map.get((violence, language), "PG-13")
            confidence = 0.75
        
        analysis = {
            "predicted_rating": predicted_rating,
            "factors": {
                "violence": violence,
                "language": language,
                "adult": content_analysis.get("adult_flag", False)
            }
        }
        
        reasoning = f"Predicted {predicted_rating} based on {violence} violence and {language} language"
        
        return AgentResponse(
            agent_role=self.role.value,
            analysis=analysis,
            confidence=confidence,
            reasoning=reasoning
        )


class ConsistencyCheckerAgent:
    
    def __init__(self):
        self.role = AgentRole.CONSISTENCY_CHECKER
    
    async def check(
        self,
        official_rating: str,
        predicted_rating: str,
        content_analysis: Dict[str, Any],
        region: str
    ) -> AgentResponse:
        
        is_consistent = self._check_consistency(
            official_rating,
            predicted_rating,
            content_analysis
        )
        
        violations = self._detect_violations(official_rating, content_analysis)
        
        analysis = {
            "is_consistent": is_consistent,
            "official_rating": official_rating,
            "predicted_rating": predicted_rating,
            "violations": violations
        }
        
        confidence = 0.85 if is_consistent and not violations else 0.70
        
        reasoning = self._explain_consistency(is_consistent, violations)
        
        return AgentResponse(
            agent_role=self.role.value,
            analysis=analysis,
            confidence=confidence,
            reasoning=reasoning
        )
    
    def _check_consistency(
        self,
        official: str,
        predicted: str,
        content: Dict[str, Any]
    ) -> bool:
        
        if official == predicted:
            return True
        
        rating_hierarchy = ["G", "PG", "PG-13", "R", "NC-17"]
        
        try:
            official_idx = rating_hierarchy.index(official)
            predicted_idx = rating_hierarchy.index(predicted)
            return abs(official_idx - predicted_idx) <= 1
        except ValueError:
            return False
    
    def _detect_violations(
        self,
        rating: str,
        content: Dict[str, Any]
    ) -> List[str]:
        
        violations = []
        
        if content.get("adult_flag") and rating in ["G", "PG"]:
            violations.append("Adult content with family rating")
        
        if content.get("violence_level") == "high" and rating == "G":
            violations.append("High violence with G rating")
        
        return violations
    
    def _explain_consistency(self, is_consistent: bool, violations: List[str]) -> str:
        if is_consistent and not violations:
            return "Official rating is consistent with content analysis"
        elif violations:
            return f"Potential issues: {'; '.join(violations)}"
        else:
            return "Rating differs from prediction but within acceptable range"


class MultiAgentOrchestrator:
    
    def __init__(self, llm_provider: Optional[Any] = None):
        self.content_analyzer = ContentAnalyzerAgent(llm_provider)
        self.rating_predictor = RatingPredictorAgent()
        self.consistency_checker = ConsistencyCheckerAgent()
    
    async def analyze_and_validate(
        self,
        content: Dict[str, Any],
        official_rating: str,
        region: str = "US"
    ) -> Dict[str, Any]:
        
        content_response = await self.content_analyzer.analyze(content, region)
        
        rating_response = await self.rating_predictor.predict(
            content_response.analysis,
            region
        )
        
        consistency_response = await self.consistency_checker.check(
            official_rating,
            rating_response.analysis["predicted_rating"],
            content_response.analysis,
            region
        )
        
        final_confidence = (
            content_response.confidence * 0.3 +
            rating_response.confidence * 0.3 +
            consistency_response.confidence * 0.4
        )
        
        return {
            "agents": {
                "content_analyzer": {
                    "analysis": content_response.analysis,
                    "confidence": content_response.confidence,
                    "reasoning": content_response.reasoning
                },
                "rating_predictor": {
                    "analysis": rating_response.analysis,
                    "confidence": rating_response.confidence,
                    "reasoning": rating_response.reasoning
                },
                "consistency_checker": {
                    "analysis": consistency_response.analysis,
                    "confidence": consistency_response.confidence,
                    "reasoning": consistency_response.reasoning
                }
            },
            "final_decision": {
                "status": "pass" if consistency_response.analysis["is_consistent"] else "warning",
                "confidence": final_confidence,
                "official_rating": official_rating,
                "predicted_rating": rating_response.analysis["predicted_rating"]
            }
        }

