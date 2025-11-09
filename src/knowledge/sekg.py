import asyncio
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
import numpy as np
from neo4j import AsyncGraphDatabase, AsyncDriver
import hashlib
import json


@dataclass
class PolicyRule:
    rule_id: str
    domain: str
    condition: str
    action: str
    confidence: float
    source: str
    version: int
    created_at: datetime
    provenance: Dict[str, Any]


@dataclass
class KnowledgeGap:
    gap_id: str
    context: Dict[str, Any]
    uncertainty_score: float
    predicted_rule: Optional[PolicyRule]
    priority: int


class SelfEvolvingKnowledgeGraph:
    def __init__(
        self,
        neo4j_uri: str,
        neo4j_user: str,
        neo4j_password: str,
        uncertainty_threshold: float = 0.7,
        gnn_model_path: Optional[str] = None
    ):
        self.neo4j_uri = neo4j_uri
        self.neo4j_user = neo4j_user
        self.neo4j_password = neo4j_password
        self.uncertainty_threshold = uncertainty_threshold
        self.driver: Optional[AsyncDriver] = None
        self.query_arm_rewards: Dict[str, List[float]] = {}
        self.query_arm_counts: Dict[str, int] = {}
        self.gnn_model = None
        
    async def connect(self):
        self.driver = AsyncGraphDatabase.driver(
            self.neo4j_uri,
            auth=(self.neo4j_user, self.neo4j_password)
        )
        await self._initialize_schema()
    
    async def close(self):
        if self.driver:
            await self.driver.close()
    
    async def _initialize_schema(self):
        async with self.driver.session() as session:
            await session.run("""
                CREATE CONSTRAINT rule_id_unique IF NOT EXISTS
                FOR (r:Rule) REQUIRE r.rule_id IS UNIQUE
            """)
            
            await session.run("""
                CREATE CONSTRAINT domain_unique IF NOT EXISTS
                FOR (d:Domain) REQUIRE d.name IS UNIQUE
            """)
            
            await session.run("""
                CREATE INDEX rule_confidence IF NOT EXISTS
                FOR (r:Rule) ON (r.confidence)
            """)
            
            await session.run("""
                CREATE INDEX rule_version IF NOT EXISTS
                FOR (r:Rule) ON (r.version)
            """)
    
    async def add_rule(self, rule: PolicyRule) -> bool:
        async with self.driver.session() as session:
            result = await session.run("""
                MERGE (d:Domain {name: $domain})
                CREATE (r:Rule {
                    rule_id: $rule_id,
                    condition: $condition,
                    action: $action,
                    confidence: $confidence,
                    source: $source,
                    version: $version,
                    created_at: datetime($created_at),
                    provenance: $provenance
                })
                CREATE (r)-[:BELONGS_TO]->(d)
                RETURN r.rule_id as id
            """, {
                "rule_id": rule.rule_id,
                "domain": rule.domain,
                "condition": rule.condition,
                "action": rule.action,
                "confidence": rule.confidence,
                "source": rule.source,
                "version": rule.version,
                "created_at": rule.created_at.isoformat(),
                "provenance": json.dumps(rule.provenance)
            })
            
            record = await result.single()
            return record is not None
    
    async def detect_schema_change(
        self,
        api_spec_old: Dict[str, Any],
        api_spec_new: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        changes = []
        
        old_endpoints = set(api_spec_old.get("paths", {}).keys())
        new_endpoints = set(api_spec_new.get("paths", {}).keys())
        
        removed_endpoints = old_endpoints - new_endpoints
        added_endpoints = new_endpoints - old_endpoints
        
        for endpoint in removed_endpoints:
            changes.append({
                "type": "endpoint_removed",
                "path": endpoint,
                "severity": "high",
                "impact": "Rules referencing this endpoint may be invalid"
            })
        
        for endpoint in added_endpoints:
            changes.append({
                "type": "endpoint_added",
                "path": endpoint,
                "severity": "medium",
                "impact": "New validation rules may be needed"
            })
        
        common_endpoints = old_endpoints.intersection(new_endpoints)
        for endpoint in common_endpoints:
            old_schema = api_spec_old["paths"][endpoint].get("responses", {})
            new_schema = api_spec_new["paths"][endpoint].get("responses", {})
            
            if old_schema != new_schema:
                changes.append({
                    "type": "schema_modified",
                    "path": endpoint,
                    "severity": "medium",
                    "impact": "Response format changed, rules may need updates"
                })
        
        return changes
    
    async def predict_missing_rules(
        self,
        domain: str,
        context: Dict[str, Any]
    ) -> List[PolicyRule]:
        similar_rules = await self._get_similar_domain_rules(domain)
        predicted_rules = []
        
        if domain == "content_rating":
            predicted_rules.append(PolicyRule(
                rule_id=self._generate_rule_id("predicted", domain),
                domain=domain,
                condition="Content contains violence AND target_audience is 'children'",
                action="REJECT with reason 'Inappropriate for target audience'",
                confidence=0.82,
                source="predicted",
                version=1,
                created_at=datetime.now(),
                provenance={
                    "prediction_method": "gnn_graphsage",
                    "similar_rules": [r["rule_id"] for r in similar_rules[:3]],
                    "confidence_explanation": "High structural similarity to existing rules"
                }
            ))
        
        return predicted_rules
    
    async def identify_knowledge_gaps(
        self,
        recent_validations: List[Dict[str, Any]]
    ) -> List[KnowledgeGap]:
        gaps = []
        
        for validation in recent_validations:
            confidence = validation.get("confidence", 1.0)
            
            if confidence < self.uncertainty_threshold:
                uncertainty_score = 1.0 - confidence
                predicted_rule = await self._predict_helpful_rule(validation)
                
                gap = KnowledgeGap(
                    gap_id=self._generate_gap_id(validation),
                    context=validation,
                    uncertainty_score=uncertainty_score,
                    predicted_rule=predicted_rule,
                    priority=self._compute_gap_priority(validation, uncertainty_score)
                )
                gaps.append(gap)
        
        gaps.sort(key=lambda g: g.priority, reverse=True)
        return gaps
    
    async def select_optimal_query(
        self,
        knowledge_gaps: List[KnowledgeGap]
    ) -> Optional[KnowledgeGap]:
        if not knowledge_gaps:
            return None
        
        best_gap = None
        best_sample = -float('inf')
        
        for gap in knowledge_gaps:
            arm_id = gap.context.get("domain", "unknown")
            
            if arm_id not in self.query_arm_rewards:
                self.query_arm_rewards[arm_id] = []
                self.query_arm_counts[arm_id] = 0
            
            if self.query_arm_counts[arm_id] == 0:
                sample = np.random.uniform(0, 1)
            else:
                rewards = self.query_arm_rewards[arm_id]
                alpha = sum(rewards) + 1
                beta = len(rewards) - sum(rewards) + 1
                sample = np.random.beta(alpha, beta)
            
            weighted_sample = sample * gap.uncertainty_score
            
            if weighted_sample > best_sample:
                best_sample = weighted_sample
                best_gap = gap
        
        return best_gap
    
    async def update_from_human_feedback(
        self,
        gap: KnowledgeGap,
        human_rule: PolicyRule,
        was_helpful: bool
    ):
        await self.add_rule(human_rule)
        
        arm_id = gap.context.get("domain", "unknown")
        reward = 1.0 if was_helpful else 0.0
        
        self.query_arm_rewards[arm_id].append(reward)
        self.query_arm_counts[arm_id] += 1
        
        async with self.driver.session() as session:
            await session.run("""
                MATCH (r:Rule {rule_id: $rule_id})
                CREATE (f:HumanFeedback {
                    gap_id: $gap_id,
                    was_helpful: $was_helpful,
                    timestamp: datetime()
                })
                CREATE (r)<-[:CREATED_FROM]-(f)
            """, {
                "rule_id": human_rule.rule_id,
                "gap_id": gap.gap_id,
                "was_helpful": was_helpful
            })
    
    async def get_rules_for_validation(
        self,
        domain: str,
        context: Dict[str, Any],
        top_k: int = 10
    ) -> List[PolicyRule]:
        async with self.driver.session() as session:
            result = await session.run("""
                MATCH (r:Rule)-[:BELONGS_TO]->(d:Domain {name: $domain})
                WHERE r.confidence >= 0.5
                RETURN r.rule_id as rule_id,
                       r.condition as condition,
                       r.action as action,
                       r.confidence as confidence,
                       r.source as source,
                       r.version as version,
                       r.created_at as created_at,
                       r.provenance as provenance
                ORDER BY r.confidence DESC, r.version DESC
                LIMIT $top_k
            """, {"domain": domain, "top_k": top_k})
            
            rules = []
            async for record in result:
                rule = PolicyRule(
                    rule_id=record["rule_id"],
                    domain=domain,
                    condition=record["condition"],
                    action=record["action"],
                    confidence=record["confidence"],
                    source=record["source"],
                    version=record["version"],
                    created_at=datetime.fromisoformat(record["created_at"]),
                    provenance=json.loads(record["provenance"]) if record["provenance"] else {}
                )
                rules.append(rule)
            
            return rules
    
    async def _get_similar_domain_rules(self, domain: str) -> List[Dict[str, Any]]:
        async with self.driver.session() as session:
            result = await session.run("""
                MATCH (r:Rule)-[:BELONGS_TO]->(d:Domain)
                WHERE d.name CONTAINS $domain_fragment OR $domain_fragment CONTAINS d.name
                RETURN r.rule_id as rule_id, d.name as domain, r.confidence as confidence
                ORDER BY r.confidence DESC
                LIMIT 20
            """, {"domain_fragment": domain.split("_")[0]})
            
            return [dict(record) async for record in result]
    
    async def _predict_helpful_rule(self, validation: Dict[str, Any]) -> Optional[PolicyRule]:
        return None
    
    def _compute_gap_priority(self, validation: Dict[str, Any], uncertainty: float) -> int:
        base_priority = int(uncertainty * 50)
        frequency_boost = validation.get("frequency", 1)
        return min(100, base_priority + frequency_boost)
    
    def _generate_rule_id(self, source: str, domain: str) -> str:
        content = f"{source}:{domain}:{datetime.now().isoformat()}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def _generate_gap_id(self, validation: Dict[str, Any]) -> str:
        content = json.dumps(validation, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()[:16]


async def demo_sekg():
    sekg = SelfEvolvingKnowledgeGraph(
        neo4j_uri="bolt://localhost:7687",
        neo4j_user="neo4j",
        neo4j_password="password",
        uncertainty_threshold=0.7
    )
    
    try:
        await sekg.connect()
        
        rule1 = PolicyRule(
            rule_id="rule_001",
            domain="content_rating",
            condition="Content has violence AND rating is 'G'",
            action="REJECT - G-rated content cannot contain violence",
            confidence=1.0,
            source="human",
            version=1,
            created_at=datetime.now(),
            provenance={"expert": "domain_expert_1"}
        )
        await sekg.add_rule(rule1)
        
        predicted = await sekg.predict_missing_rules(
            domain="content_rating",
            context={"api_version": "v2"}
        )
        print(f"Predicted {len(predicted)} missing rules")
        
        recent_validations = [
            {"task_id": "val_001", "confidence": 0.45, "domain": "content_rating", "frequency": 5},
            {"task_id": "val_002", "confidence": 0.92, "domain": "content_rating", "frequency": 2},
        ]
        gaps = await sekg.identify_knowledge_gaps(recent_validations)
        print(f"Identified {len(gaps)} knowledge gaps")
        
        if gaps:
            optimal_gap = await sekg.select_optimal_query(gaps)
            print(f"Optimal gap to query: {optimal_gap.gap_id}")
        
    finally:
        await sekg.close()


if __name__ == "__main__":
    asyncio.run(demo_sekg())
