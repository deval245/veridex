from typing import Any, Dict, List, Optional, TypeVar, Generic
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime
from uuid import uuid4


class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class ValidationStatus(Enum):
    PASS = "pass"
    FAIL = "fail"
    ERROR = "error"
    SKIP = "skip"


@dataclass
class Context:
    request_id: str = field(default_factory=lambda: str(uuid4()))
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class Task:
    task_id: str
    task_type: str
    payload: Dict[str, Any]
    status: TaskStatus = TaskStatus.PENDING
    context: Context = field(default_factory=Context)


@dataclass
class Result:
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationResult:
    entity_id: str
    status: ValidationStatus
    expected: Any
    actual: Any
    confidence: float = 1.0
    reason: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


T = TypeVar('T')


@dataclass
class Response(Generic[T]):
    data: Optional[T] = None
    error: Optional[str] = None
    request_id: str = field(default_factory=lambda: str(uuid4()))
    
    @property
    def is_success(self) -> bool:
        return self.error is None

