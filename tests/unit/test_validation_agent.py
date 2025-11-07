import pytest
from unittest.mock import AsyncMock, Mock
from src.agents.validation import ValidationAgent
from src.core.types import Task, Context, ValidationStatus
from src.config import Settings


@pytest.fixture
def mock_llm():
    llm = AsyncMock()
    llm.generate = AsyncMock(return_value='{"critical": false, "severity": "low", "explanation": "Minor mismatch"}')
    return llm


@pytest.fixture
def validation_agent(mock_llm):
    settings = Settings()
    return ValidationAgent(mock_llm, settings)


def test_agent_can_handle_validate_task(validation_agent):
    task = Task(task_id="1", task_type="validate", payload={})
    assert validation_agent.can_handle(task) is True


def test_agent_cannot_handle_other_task(validation_agent):
    task = Task(task_id="1", task_type="discover", payload={})
    assert validation_agent.can_handle(task) is False


@pytest.mark.asyncio
async def test_validation_exact_match(validation_agent):
    task = Task(
        task_id="1",
        task_type="validate",
        payload={
            "entity_id": "123",
            "expected": "R",
            "actual": "R",
            "domain": "content_rating"
        }
    )
    ctx = Context()
    
    result = await validation_agent.execute(task, ctx)
    
    assert result.success is True
    validation = result.data["validation"]
    assert validation["status"] == ValidationStatus.PASS
    assert validation["confidence"] == 1.0


@pytest.mark.asyncio
async def test_validation_mismatch(validation_agent, mock_llm):
    task = Task(
        task_id="1",
        task_type="validate",
        payload={
            "entity_id": "123",
            "expected": "R",
            "actual": "PG-13",
            "domain": "content_rating"
        }
    )
    ctx = Context()
    
    result = await validation_agent.execute(task, ctx)
    
    assert result.success is True
    assert mock_llm.generate.called

