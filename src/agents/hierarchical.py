import asyncio
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from datetime import datetime


class AgentTier(Enum):
    META = 1
    STRATEGIC = 2
    TACTICAL = 3


@dataclass
class AgentDecision:
    agent_id: str
    decision_type: str
    target_agent: Optional[str]
    confidence: float
    reasoning: str
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskExecution:
    task_id: str
    task_type: str
    start_time: datetime
    end_time: Optional[datetime] = None
    decisions: List[AgentDecision] = field(default_factory=list)
    result: Optional[Any] = None
    latency_ms: Optional[float] = None
    accuracy: Optional[float] = None
    cost: float = 0.0


class MetaAgent:
    def __init__(
        self,
        agent_id: str = "meta_agent",
        learning_rate: float = 0.001,
        gamma: float = 0.99,
        epsilon: float = 0.2
    ):
        self.agent_id = agent_id
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.policy_weights: Dict[str, np.ndarray] = {}
        self.value_weights: Dict[str, np.ndarray] = {}
        self.experience_buffer: List[Dict[str, Any]] = []
        self.agent_stats: Dict[str, Dict[str, List[float]]] = {}
    
    async def decide_strategy(
        self,
        task: Dict[str, Any],
        available_agents: List[str]
    ) -> AgentDecision:
        task_features = self._extract_task_features(task)
        action_probs = self._compute_policy(task_features, available_agents)
        action = self._sample_action(action_probs, available_agents)
        value_estimate = self._compute_value(task_features)
        
        return AgentDecision(
            agent_id=self.agent_id,
            decision_type="delegate" if action != "self" else "execute",
            target_agent=action if action != "self" else None,
            confidence=action_probs.get(action, 0.0),
            reasoning=self._generate_reasoning(task, action, action_probs),
            timestamp=datetime.now(),
            metadata={
                "task_features": task_features.tolist(),
                "action_probs": action_probs,
                "value_estimate": float(value_estimate)
            }
        )
    
    async def update_policy(self, execution: TaskExecution, reward: float):
        self.experience_buffer.append({
            "task_id": execution.task_id,
            "decisions": execution.decisions,
            "reward": reward,
            "latency_ms": execution.latency_ms,
            "accuracy": execution.accuracy,
            "cost": execution.cost
        })
        
        if execution.decisions:
            first_decision = execution.decisions[0]
            if first_decision.target_agent:
                agent_id = first_decision.target_agent
                if agent_id not in self.agent_stats:
                    self.agent_stats[agent_id] = {
                        "rewards": [],
                        "latencies": [],
                        "accuracies": []
                    }
                
                self.agent_stats[agent_id]["rewards"].append(reward)
                self.agent_stats[agent_id]["latencies"].append(execution.latency_ms or 0)
                self.agent_stats[agent_id]["accuracies"].append(execution.accuracy or 0)
        
        if len(self.experience_buffer) >= 32:
            await self._ppo_update()
    
    async def _ppo_update(self):
        batch = self.experience_buffer[-32:]
        
        returns = []
        for i, exp in enumerate(batch):
            G = sum([self.gamma ** j * batch[i+j]["reward"] 
                    for j in range(min(10, len(batch) - i))])
            returns.append(G)
        
        returns = np.array(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        self.experience_buffer = self.experience_buffer[-32:]
    
    def _extract_task_features(self, task: Dict[str, Any]) -> np.ndarray:
        features = []
        
        task_types = ["validate_content", "retrieve_policy", "verify_compliance"]
        task_type = task.get("task_type", "unknown")
        type_encoding = [1.0 if tt == task_type else 0.0 for tt in task_types]
        features.extend(type_encoding)
        
        complexity = len(str(task.get("payload", ""))) / 1000.0
        features.append(complexity)
        
        timeout = task.get("timeout_ms", 60000) / 60000.0
        features.append(timeout)
        
        historical_success = self._get_historical_success_rate(task_type)
        features.append(historical_success)
        
        return np.array(features)
    
    def _compute_policy(
        self,
        state: np.ndarray,
        available_agents: List[str]
    ) -> Dict[str, float]:
        action_probs = {}
        action_probs["self"] = 0.1
        
        for agent_id in available_agents:
            if agent_id in self.agent_stats and self.agent_stats[agent_id]["rewards"]:
                mean_reward = np.mean(self.agent_stats[agent_id]["rewards"])
                std_reward = np.std(self.agent_stats[agent_id]["rewards"])
                sampled_reward = np.random.normal(mean_reward, std_reward + 0.1)
                action_probs[agent_id] = max(0, sampled_reward)
            else:
                action_probs[agent_id] = 0.5
        
        total = sum(action_probs.values())
        if total > 0:
            action_probs = {k: v / total for k, v in action_probs.items()}
        
        return action_probs
    
    def _compute_value(self, state: np.ndarray) -> float:
        all_rewards = [r for stats in self.agent_stats.values() 
                      for r in stats.get("rewards", [])]
        return np.mean(all_rewards) if all_rewards else 0.0
    
    def _sample_action(
        self,
        action_probs: Dict[str, float],
        available_agents: List[str]
    ) -> str:
        actions = list(action_probs.keys())
        probs = list(action_probs.values())
        return np.random.choice(actions, p=probs)
    
    def _generate_reasoning(
        self,
        task: Dict[str, Any],
        selected_action: str,
        action_probs: Dict[str, float]
    ) -> str:
        if selected_action == "self":
            return f"Direct execution (confidence: {action_probs['self']:.2f})"
        prob = action_probs.get(selected_action, 0.0)
        return f"Delegate to {selected_action} (expected reward: {prob:.2f})"
    
    def _get_historical_success_rate(self, task_type: str) -> float:
        return 0.5


class StrategicAgent:
    def __init__(self, agent_id: str, expertise: List[str]):
        self.agent_id = agent_id
        self.expertise = expertise
        self.tactical_agent_stats: Dict[str, Dict[str, List[float]]] = {}
    
    async def route_task(
        self,
        task: Dict[str, Any],
        tactical_agents: List[str]
    ) -> AgentDecision:
        task_type = task.get("task_type", "unknown")
        
        agent_scores = {}
        for agent_id in tactical_agents:
            if agent_id not in self.tactical_agent_stats:
                self.tactical_agent_stats[agent_id] = {"successes": [], "attempts": []}
            
            stats = self.tactical_agent_stats[agent_id]
            successes = len([s for s in stats.get("successes", []) if s])
            attempts = len(stats.get("attempts", []))
            
            alpha = successes + 1
            beta = attempts - successes + 1
            sampled_prob = np.random.beta(alpha, beta)
            agent_scores[agent_id] = sampled_prob
        
        best_agent = max(agent_scores.items(), key=lambda x: x[1])[0]
        
        return AgentDecision(
            agent_id=self.agent_id,
            decision_type="route",
            target_agent=best_agent,
            confidence=agent_scores[best_agent],
            reasoning=f"Thompson Sampling: {best_agent} (score: {agent_scores[best_agent]:.2f})",
            timestamp=datetime.now(),
            metadata={"agent_scores": agent_scores}
        )
    
    async def update_stats(self, tactical_agent_id: str, success: bool):
        if tactical_agent_id not in self.tactical_agent_stats:
            self.tactical_agent_stats[tactical_agent_id] = {"successes": [], "attempts": []}
        
        self.tactical_agent_stats[tactical_agent_id]["successes"].append(success)
        self.tactical_agent_stats[tactical_agent_id]["attempts"].append(True)


class HierarchicalOrchestrator:
    def __init__(self):
        self.meta_agent = MetaAgent()
        self.strategic_agents = {
            "planner_agent": StrategicAgent("planner_agent", ["planning", "decomposition"]),
            "knowledge_agent": StrategicAgent("knowledge_agent", ["retrieval", "search"]),
            "reasoner_agent": StrategicAgent("reasoner_agent", ["validation", "reasoning"])
        }
        self.tactical_agents = {
            "retriever": None,
            "validator": None,
            "verifier": None
        }
    
    async def execute_task(self, task: Dict[str, Any]) -> TaskExecution:
        execution = TaskExecution(
            task_id=task.get("task_id", "unknown"),
            task_type=task.get("task_type", "unknown"),
            start_time=datetime.now()
        )
        
        available_strategic = list(self.strategic_agents.keys())
        meta_decision = await self.meta_agent.decide_strategy(task, available_strategic)
        execution.decisions.append(meta_decision)
        
        if meta_decision.decision_type == "execute":
            result = await self._execute_directly(task)
            execution.result = result
        else:
            strategic_agent_id = meta_decision.target_agent
            strategic_agent = self.strategic_agents[strategic_agent_id]
            
            available_tactical = list(self.tactical_agents.keys())
            route_decision = await strategic_agent.route_task(task, available_tactical)
            execution.decisions.append(route_decision)
            
            tactical_agent_id = route_decision.target_agent
            result = await self._execute_tactical(tactical_agent_id, task)
            execution.result = result
            
            success = result.get("success", False)
            await strategic_agent.update_stats(tactical_agent_id, success)
        
        execution.end_time = datetime.now()
        execution.latency_ms = (execution.end_time - execution.start_time).total_seconds() * 1000
        execution.accuracy = execution.result.get("accuracy", 0.0) if execution.result else 0.0
        execution.cost = execution.result.get("cost", 0.0) if execution.result else 0.0
        
        reward = self._compute_reward(execution)
        await self.meta_agent.update_policy(execution, reward)
        
        return execution
    
    async def _execute_directly(self, task: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "success": True,
            "output": "Direct execution result",
            "accuracy": 0.75,
            "cost": 0.001
        }
    
    async def _execute_tactical(
        self,
        tactical_agent_id: str,
        task: Dict[str, Any]
    ) -> Dict[str, Any]:
        await asyncio.sleep(0.1)
        return {
            "success": True,
            "output": f"Result from {tactical_agent_id}",
            "accuracy": np.random.uniform(0.7, 0.95),
            "cost": 0.002
        }
    
    def _compute_reward(self, execution: TaskExecution) -> float:
        latency_penalty = -(execution.latency_ms or 0) / 1000.0
        accuracy_reward = 10.0 * (execution.accuracy or 0)
        cost_penalty = -0.5 * execution.cost
        return latency_penalty + accuracy_reward + cost_penalty


async def benchmark_hierarchical_vs_flat():
    orchestrator = HierarchicalOrchestrator()
    
    tasks = [
        {
            "task_id": f"task_{i}",
            "task_type": "validate_content",
            "payload": {"content_id": f"content_{i}"},
            "timeout_ms": 30000
        }
        for i in range(100)
    ]
    
    results = []
    for task in tasks:
        execution = await orchestrator.execute_task(task)
        results.append({
            "latency_ms": execution.latency_ms,
            "accuracy": execution.accuracy,
            "cost": execution.cost,
            "num_decisions": len(execution.decisions)
        })
    
    avg_latency = np.mean([r["latency_ms"] for r in results])
    avg_accuracy = np.mean([r["accuracy"] for r in results])
    total_cost = sum([r["cost"] for r in results])
    
    print("=" * 60)
    print("HIERARCHICAL ORCHESTRATION BENCHMARK")
    print("=" * 60)
    print(f"Tasks executed: {len(results)}")
    print(f"Average latency: {avg_latency:.2f} ms")
    print(f"Average accuracy: {avg_accuracy:.2%}")
    print(f"Total cost: ${total_cost:.4f}")
    print(f"Decisions per task: {np.mean([r['num_decisions'] for r in results]):.1f}")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(benchmark_hierarchical_vs_flat())
