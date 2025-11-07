from typing import Optional, Any, Dict
from functools import wraps
import hashlib
import json
from datetime import datetime, timedelta


class SimpleCache:
    def __init__(self, ttl_seconds: int = 3600):
        self._cache: Dict[str, tuple[Any, datetime]] = {}
        self.ttl = ttl_seconds
    
    def _make_key(self, *args, **kwargs) -> str:
        key_str = json.dumps({"args": args, "kwargs": kwargs}, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def get(self, key: str) -> Optional[Any]:
        if key in self._cache:
            value, expires_at = self._cache[key]
            if datetime.utcnow() < expires_at:
                return value
            del self._cache[key]
        return None
    
    def set(self, key: str, value: Any):
        expires_at = datetime.utcnow() + timedelta(seconds=self.ttl)
        self._cache[key] = (value, expires_at)
    
    def clear(self):
        self._cache.clear()
    
    def size(self) -> int:
        return len(self._cache)


def cached(ttl: int = 3600):
    cache = SimpleCache(ttl)
    
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            key = cache._make_key(*args, **kwargs)
            result = cache.get(key)
            
            if result is not None:
                return result
            
            result = await func(*args, **kwargs)
            cache.set(key, result)
            return result
        
        wrapper.cache = cache
        return wrapper
    return decorator

