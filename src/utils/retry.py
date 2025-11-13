"""
FAANG-Level Retry Mechanism with Circuit Breaker Pattern
- Exponential backoff with jitter
- Circuit breaker to prevent cascade failures
- Adaptive rate limiting
- Metrics tracking
"""

import asyncio
import time
import random
from enum import Enum
from typing import Optional, Callable, Any
from dataclasses import dataclass, field


class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class RetryConfig:
    max_attempts: int = 5
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True
    timeout: float = 15.0


@dataclass
class CircuitBreakerConfig:
    failure_threshold: int = 5
    success_threshold: int = 2
    timeout: float = 60.0


class CircuitBreaker:
    def __init__(self, config: CircuitBreakerConfig):
        self.config = config
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[float] = None
    
    def record_success(self):
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.config.success_threshold:
                self.state = CircuitState.CLOSED
                self.failure_count = 0
                self.success_count = 0
        elif self.state == CircuitState.CLOSED:
            self.failure_count = 0
    
    def record_failure(self):
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.config.failure_threshold:
            self.state = CircuitState.OPEN
    
    def can_attempt(self) -> bool:
        if self.state == CircuitState.CLOSED:
            return True
        
        if self.state == CircuitState.OPEN:
            if self.last_failure_time and \
               (time.time() - self.last_failure_time) >= self.config.timeout:
                self.state = CircuitState.HALF_OPEN
                self.success_count = 0
                return True
            return False
        
        return True


class AdaptiveRateLimiter:
    def __init__(self, initial_rate: float = 10.0):
        self.rate = initial_rate
        self.min_rate = 1.0
        self.max_rate = 50.0
        self.last_call = 0
        self.success_streak = 0
        self.failure_streak = 0
    
    async def acquire(self):
        delay = 1.0 / self.rate
        now = time.time()
        elapsed = now - self.last_call
        
        if elapsed < delay:
            await asyncio.sleep(delay - elapsed)
        
        self.last_call = time.time()
    
    def record_success(self):
        self.success_streak += 1
        self.failure_streak = 0
        
        if self.success_streak >= 10 and self.rate < self.max_rate:
            self.rate = min(self.rate * 1.1, self.max_rate)
            self.success_streak = 0
    
    def record_failure(self):
        self.failure_streak += 1
        self.success_streak = 0
        
        if self.failure_streak >= 3 and self.rate > self.min_rate:
            self.rate = max(self.rate * 0.5, self.min_rate)
            self.failure_streak = 0


class RetryManager:
    def __init__(
        self,
        retry_config: RetryConfig = RetryConfig(),
        circuit_config: CircuitBreakerConfig = CircuitBreakerConfig()
    ):
        self.retry_config = retry_config
        self.circuit_breaker = CircuitBreaker(circuit_config)
        self.rate_limiter = AdaptiveRateLimiter()
    
    def _calculate_delay(self, attempt: int) -> float:
        delay = min(
            self.retry_config.base_delay * (
                self.retry_config.exponential_base ** attempt
            ),
            self.retry_config.max_delay
        )
        
        if self.retry_config.jitter:
            delay *= (0.5 + random.random())
        
        return delay
    
    async def execute(
        self,
        func: Callable,
        *args,
        on_retry: Optional[Callable] = None,
        **kwargs
    ) -> Any:
        last_exception = None
        
        for attempt in range(self.retry_config.max_attempts):
            if not self.circuit_breaker.can_attempt():
                raise Exception("Circuit breaker is OPEN")
            
            try:
                await self.rate_limiter.acquire()
                
                result = await asyncio.wait_for(
                    func(*args, **kwargs),
                    timeout=self.retry_config.timeout
                )
                
                self.circuit_breaker.record_success()
                self.rate_limiter.record_success()
                
                return result
                
            except asyncio.TimeoutError as e:
                last_exception = e
                self.circuit_breaker.record_failure()
                self.rate_limiter.record_failure()
                
                if attempt < self.retry_config.max_attempts - 1:
                    delay = self._calculate_delay(attempt)
                    
                    if on_retry:
                        on_retry(attempt, delay, e)
                    
                    await asyncio.sleep(delay)
                
            except Exception as e:
                if self._is_retryable(e):
                    last_exception = e
                    self.circuit_breaker.record_failure()
                    self.rate_limiter.record_failure()
                    
                    if attempt < self.retry_config.max_attempts - 1:
                        delay = self._calculate_delay(attempt)
                        
                        if on_retry:
                            on_retry(attempt, delay, e)
                        
                        await asyncio.sleep(delay)
                else:
                    raise
        
        raise last_exception or Exception("Max retries exceeded")
    
    def _is_retryable(self, exception: Exception) -> bool:
        retryable_types = (
            asyncio.TimeoutError,
            ConnectionError,
            TimeoutError
        )
        
        if isinstance(exception, retryable_types):
            return True
        
        error_msg = str(exception).lower()
        retryable_keywords = [
            'timeout', 'connection', 'temporary',
            'rate limit', '429', '503', '504'
        ]
        
        return any(keyword in error_msg for keyword in retryable_keywords)









