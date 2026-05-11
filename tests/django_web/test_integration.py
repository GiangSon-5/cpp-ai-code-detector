import json
import base64
import pytest
from unittest.mock import patch, MagicMock
from django.urls import reverse
from django.contrib.auth.models import User
from src.django_web.apps.submissions.models import BronzeSubmission

@pytest.fixture
def auth_client(client, db):
    user = User.objects.create_user(username="testuser", password="password")
    client.force_login(user)
    return client, user

@pytest.fixture
def sample_submission(db, auth_client):
    _, user = auth_client
    return BronzeSubmission.objects.create(
        code_hash="fakehash_integration",
        raw_code="int main() {}",
        user=user
    )


@pytest.mark.django_db
@patch("src.django_web.apps.submissions.views.httpx.Client.stream")
def test_sse_proxy_integration(mock_stream, auth_client, sample_submission):
    """
    IT_03: Django SSE Proxy nhận yêu cầu, gọi FastAPI và yield stream data.
    """
    client, _ = auth_client
    
    # Mocking FastAPI SSE stream
    mock_context_manager = MagicMock()
    mock_response = MagicMock()
    
    # Simulate FastAPI yielding lines as strings (httpx iter_lines yields str)
    mock_lines = [
        'data: {"step": "router", "progress": 5, "message": "Classifying"}',
        'data: {"step": "complete", "progress": 100, "data": {"final_pred": "AI GENERATED", "final_score": 0.85}}'
    ]
    mock_response.iter_lines.return_value = mock_lines
    mock_context_manager.__enter__.return_value = mock_response
    mock_stream.return_value = mock_context_manager
    
    # URL to proxy view
    url = reverse('submissions:sse_proxy', kwargs={'code_hash': sample_submission.code_hash})
    
    # Gọi Django View
    response = client.get(url)
    assert response.status_code == 200
    assert response['Content-Type'] == 'text/event-stream'
    
    # Đọc luồng StreamingHttpResponse
    content = b"".join(response.streaming_content)
    
    # Kiểm tra stream trả về cho browser có đúng không
    assert b'"step": "router"' in content
    assert b'"step": "complete"' in content
    
    # Đảm bảo BronzeSubmission được cập nhật khi có step="complete"
    sample_submission.refresh_from_db()
    assert sample_submission.prediction == "AI GENERATED"
    assert sample_submission.confidence == 0.85
    assert sample_submission.result_json is not None
    assert sample_submission.result_json["final_score"] == 0.85
