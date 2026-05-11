import pytest
from fastapi.testclient import TestClient
from src.fastapi_service.main import app

client = TestClient(app)

def test_fastapi_health_check():
    """Kiểm tra API có đang hoạt động không"""
    response = client.get("/health")
    assert response.status_code == 200
    assert "status" in response.json()
