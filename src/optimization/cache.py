import asyncio
import hashlib
import json
from typing import Any, Optional, Dict
from datetime import datetime, timedelta
from functools import wraps


class InMemoryCache:
    
    def __init__(self, ttl_seconds: int = 3600):
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._ttl = ttl_seconds
    
    def _generate_key(self, *args, **kwargs) -> str:
        key_data = json.dumps({"args": args, "kwargs": kwargs}, sort_keys=True)
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _is_expired(self, entry: Dict[str, Any]) -> bool:
        expiry = entry["expiry"]
        return datetime.now() > expiry
    
    def get(self, key: str) -> Optional[Any]:
        if key in self._cache:
            entry = self._cache[key]
            if not self._is_expired(entry):
                return entry["value"]
            else:
                del self._cache[key]
        return None
    
    def set(self, key: str, value: Any) -> None:
        expiry = datetime.now() + timedelta(seconds=self._ttl)
        self._cache[key] = {"value": value, "expiry": expiry}
    
    def clear(self) -> None:
        self._cache.clear()
    
    def size(self) -> int:
        self._cleanup_expired()
        return len(self._cache)
    
    def _cleanup_expired(self) -> None:
        expired_keys = [
            k for k, v in self._cache.items()
            if self._is_expired(v)
        ]
        for key in expired_keys:
            del self._cache[key]


def cached(ttl_seconds: int = 3600):
    cache = InMemoryCache(ttl_seconds)
    
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            cache_key = cache._generate_key(*args, **kwargs)
            
            cached_result = cache.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            result = await func(*args, **kwargs)
            cache.set(cache_key, result)
            
            return result
        
        wrapper.cache = cache
        return wrapper
    
    return decorator


class BatchProcessor:
    
    def __init__(self, batch_size: int = 50, max_wait_time: float = 1.0):
        self.batch_size = batch_size
        self.max_wait_time = max_wait_time
        self._queue = []
        self._results = {}
        self._processing = False
    
    async def add_to_batch(self, item_id: str, item_data: Any) -> Any:
        future = asyncio.Future()
        self._queue.append((item_id, item_data, future))
        
        if not self._processing:
            asyncio.create_task(self._process_batch())
        
        return await future
    
    async def _process_batch(self):
        self._processing = True
        
        await asyncio.sleep(self.max_wait_time)
        
        if len(self._queue) >= self.batch_size or self._queue:
            batch = self._queue[:self.batch_size]
            self._queue = self._queue[self.batch_size:]
            
            for item_id, item_data, future in batch:
                result = await self._process_single(item_id, item_data)
                future.set_result(result)
        
        self._processing = False
        
        if self._queue:
            asyncio.create_task(self._process_batch())
    
    async def _process_single(self, item_id: str, item_data: Any) -> Any:
        return item_data


class RateLimiter:
    
    def __init__(self, max_requests: int, time_window: float):
        self.max_requests = max_requests
        self.time_window = time_window
        self._requests = []
    
    async def acquire(self):
        now = datetime.now()
        
        self._requests = [
            req for req in self._requests
            if (now - req).total_seconds() < self.time_window
        ]
        
        if len(self._requests) >= self.max_requests:
            sleep_time = self.time_window - (now - self._requests[0]).total_seconds()
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
        
        self._requests.append(now)


def rate_limited(max_requests: int, time_window: float):
    limiter = RateLimiter(max_requests, time_window)
    
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            await limiter.acquire()
            return await func(*args, **kwargs)
        return wrapper
    
    return decorator

