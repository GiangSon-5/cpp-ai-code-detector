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
    assert result["final_score"] == 0.10


@pytest.mark.asyncio
async def test_node_judge_ml_dl_conflict_with_fusion(agent_service):
    """Test when DL and ML have opposite labels, and gap >= 0.40, DL is in [0.45, 0.75]. Fusion applies."""
    result = await agent_service._node_judge(
        mean_score=0.71, # AI
        perplexity=2.0,
        raw_code="int x = 0;",
        current_model_type="NORMAL",
        chunks=[],
        ml_score=0.20 # Human, gap = 0.51
    )
    
    assert result["ml_dl_conflict"] is True
    assert result["ml_dl_gap"] == 0.51
    assert result["fusion_applied"] is True
    # final_score = 0.70 * 0.71 + 0.30 * 0.20 = 0.497 + 0.060 = 0.557
    assert abs(result["final_score"] - 0.557) < 1e-6


@pytest.mark.asyncio
async def test_node_judge_ml_dl_conflict_no_fusion(agent_service):
    """Test when DL and ML have opposite labels and gap >= 0.40, but DL is confident (not in [0.45, 0.75]). No fusion."""
    result = await agent_service._node_judge(
        mean_score=0.95, # AI
        perplexity=2.0,
        raw_code="int x = 0;",
        current_model_type="NORMAL",
        chunks=[],
        ml_score=0.10 # Human, gap = 0.85
    )
    
    assert result["ml_dl_conflict"] is True
    assert result["ml_dl_gap"] == 0.85
    assert result["fusion_applied"] is False
    assert result["final_score"] == 0.95


@pytest.mark.asyncio
async def test_node_judge_ml_dl_no_conflict_same_label(agent_service):
    """Test when DL and ML have gap >= 0.40, but both predict the same class (both AI). No conflict."""
    result = await agent_service._node_judge(
        mean_score=0.95, # AI
        perplexity=2.0,
        raw_code="int x = 0;",
        current_model_type="NORMAL",
        chunks=[],
        ml_score=0.55 # AI, gap = 0.40
    )
    
    assert result["ml_dl_conflict"] is False
    assert result["ml_dl_gap"] == 0.40
    assert result["fusion_applied"] is False


@pytest.mark.asyncio
async def test_node_judge_ml_dl_no_conflict_small_gap(agent_service):
    """Test when DL and ML have opposite labels, but gap < 0.40. No conflict."""
    result = await agent_service._node_judge(
        mean_score=0.55, # AI
        perplexity=2.0,
        raw_code="int x = 0;",
        current_model_type="NORMAL",
        chunks=[],
        ml_score=0.45 # Human, gap = 0.10
    )
    
    assert result["ml_dl_conflict"] is False
    assert result["ml_dl_gap"] == 0.10
    assert result["fusion_applied"] is False
