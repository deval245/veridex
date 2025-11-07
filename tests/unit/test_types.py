import pytest
from src.core.types import (
    Context, Task, Result, ValidationResult, Response,
    TaskStatus, ValidationStatus
)


def test_context_creation():
    ctx = Context()
    assert ctx.request_id is not None
    assert isinstance(ctx.metadata, dict)
    assert ctx.created_at is not None


def test_task_creation():
    task = Task(
        task_id="test-1",
        task_type="validate",
        payload={"key": "value"}
    )
    assert task.task_id == "test-1"
    assert task.status == TaskStatus.PENDING


def test_result_success():
    result = Result(success=True, data={"result": "ok"})
    assert result.success is True
    assert result.data["result"] == "ok"
    assert result.error is None


def test_result_failure():
    result = Result(success=False, error="Something failed")
    assert result.success is False
    assert result.error == "Something failed"


def test_validation_result():
    vr = ValidationResult(
        entity_id="123",
        status=ValidationStatus.PASS,
        expected="R",
        actual="R",
        confidence=1.0,
        reason="Match"
    )
    assert vr.status == ValidationStatus.PASS
    assert vr.confidence == 1.0


def test_response_is_success():
    resp = Response(data={"ok": True})
    assert resp.is_success is True
    
    resp_err = Response(error="Failed")
    assert resp_err.is_success is False

