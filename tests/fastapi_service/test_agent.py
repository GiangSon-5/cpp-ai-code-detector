import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from src.fastapi_service.services.agent_service import AgentService

@pytest.fixture
def agent_service():
    # Mocking internal components so we don't need real Models or GPU
    with patch("src.fastapi_service.services.agent_service.ModelManager"), \
         patch("src.fastapi_service.services.agent_service.RoBERTaEngine"), \
         patch("src.fastapi_service.services.agent_service.LLMHandler") as mock_llm_class:
        
        service = AgentService()
        
        # Setup async mocks for LLMHandler
        service._llm_handler.classify_code = AsyncMock()
        service._llm_handler.compute_perplexity = AsyncMock()
        service._llm_handler.generate_critique = AsyncMock()
        service._llm_handler.generate_global_critique = AsyncMock()
        
        # Setup mock for RoBERTaEngine
        service._roberta_engine.analyze = MagicMock()
        
        yield service


@pytest.mark.asyncio
async def test_node_router_llm_agrees_with_heuristic(agent_service):
    """Test router logic when LLM and Heuristic both say OOP"""
    code = "class MyClass { public: int x; };"
    
    # Mock LLM to return OOP
    agent_service._llm_handler.classify_code.return_value = {
        "classification": "OOP",
        "provider": "mocked"
    }
    
    result = await agent_service._node_router(code)
    
    assert result["classification"] == "OOP"
    assert result["heuristic_result"]["classification"] == "OOP"


@pytest.mark.asyncio
async def test_node_router_llm_fallback(agent_service):
    """Test router logic when LLM is unavailable (fallback), it should trust heuristic"""
    code = "int main() { return 0; }"
    
    # Mock LLM to return fallback
    agent_service._llm_handler.classify_code.return_value = {
        "classification": "UNKNOWN",
        "provider": "fallback"
    }
    
    result = await agent_service._node_router(code)
    
    # Heuristic should classify "int main()" as NORMAL
    assert result["classification"] == "NORMAL"
    assert result["heuristic_result"]["classification"] == "NORMAL"


@pytest.mark.asyncio
async def test_node_judge_ambiguous_zone(agent_service):
    """Test Judge node when score is ambiguous (0.50). Should retry with opposite model."""
    
    # Setup RoBERTa mock to return a different score on retry
    agent_service._roberta_engine.analyze.return_value = {
        "mean_score": 0.85, # Highly confident AI on retry
        "chunks": []
    }
    
    result = await agent_service._node_judge(
        mean_score=0.50, # Ambiguous
        perplexity=2.0,
        raw_code="int x = 0;",
        current_model_type="NORMAL",
        chunks=[]
    )
    
    assert result["is_ambiguous"] is True
    assert result["retry_count"] == 1
    assert result["model_switched"] is True
    assert result["new_model_type"] == "OOP"
    assert result["final_score"] == 0.85


@pytest.mark.asyncio
async def test_node_judge_ppl_conflict(agent_service):
    """Test Judge node when PPL conflicts with Score. Score is high (0.80) but PPL is very high (6.0)."""
    
    # On retry, let's say the opposite model says 0.10 (Human) -> 0.4 diff from 0.5
    agent_service._roberta_engine.analyze.return_value = {
        "mean_score": 0.10,
        "chunks": []
    }
    
    result = await agent_service._node_judge(
        mean_score=0.80, # Confident AI... -> 0.3 diff from 0.5
        perplexity=6.0,  # ...but very human-like PPL
        raw_code="int x = 0;",
        current_model_type="OOP",
        chunks=[]
    )
    
    assert result["is_ambiguous"] is True
    assert result["model_switched"] is True
    assert result["new_model_type"] == "NORMAL"
    assert result["final_score"] == 0.10 # It picked the retry score because |0.2 - 0.5| = 0.3, same as |0.8 - 0.5| = 0.3. Wait, is it?
