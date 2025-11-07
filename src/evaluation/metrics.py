from typing import Dict, Any, Optional
from prometheus_client import Counter, Histogram, Gauge
from dataclasses import dataclass
from datetime import datetime
import time


@dataclass
class MetricLabels:
    agent_name: str
    task_type: str
    status: str


class MetricsCollector:
    def __init__(self, enable_prometheus: bool = True):
        self.enable_prometheus = enable_prometheus
        
        if enable_prometheus:
            self._init_prometheus_metrics()
        
        self._internal_metrics: Dict[str, Any] = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_latency_ms": 0
        }
    
    def _init_prometheus_metrics(self):
        self.request_counter = Counter(
            'veridex_requests_total',
            'Total number of validation requests',
            ['agent', 'task_type', 'status']
        )
        
        self.latency_histogram = Histogram(
            'veridex_latency_seconds',
            'Request latency in seconds',
            ['agent', 'task_type']
        )
        
        self.active_requests = Gauge(
            'veridex_active_requests',
            'Number of active requests',
            ['agent']
        )
    
    def record_request(
        self,
        agent_name: str,
        task_type: str,
        status: str,
        latency_ms: float
    ):
        self._internal_metrics["total_requests"] += 1
        self._internal_metrics["total_latency_ms"] += latency_ms
        
        if status == "success":
            self._internal_metrics["successful_requests"] += 1
        else:
            self._internal_metrics["failed_requests"] += 1
        
        if self.enable_prometheus:
            self.request_counter.labels(
                agent=agent_name,
                task_type=task_type,
                status=status
            ).inc()
            
            self.latency_histogram.labels(
                agent=agent_name,
                task_type=task_type
            ).observe(latency_ms / 1000)
    
    def get_metrics(self) -> Dict[str, Any]:
        total = self._internal_metrics["total_requests"]
        if total == 0:
            return {
                "total_requests": 0,
                "success_rate": 0.0,
                "avg_latency_ms": 0.0
            }
        
        return {
            "total_requests": total,
            "successful_requests": self._internal_metrics["successful_requests"],
            "failed_requests": self._internal_metrics["failed_requests"],
            "success_rate": self._internal_metrics["successful_requests"] / total,
            "avg_latency_ms": self._internal_metrics["total_latency_ms"] / total
        }
    
    def reset(self):
        self._internal_metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_latency_ms": 0
        }


class MetricsContext:
    def __init__(
        self,
        collector: MetricsCollector,
        agent_name: str,
        task_type: str
    ):
        self.collector = collector
        self.agent_name = agent_name
        self.task_type = task_type
        self.start_time: Optional[float] = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        latency_ms = (time.time() - self.start_time) * 1000
        status = "success" if exc_type is None else "failure"
        
        self.collector.record_request(
            self.agent_name,
            self.task_type,
            status,
            latency_ms
        )

