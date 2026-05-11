import pytest
from src.django_web.apps.submissions.models import BronzeSubmission

@pytest.mark.django_db
def test_submission_creation():
    """UT_DJ_04: Kiểm tra lưu BronzeSubmission xuống DB"""
    submission = BronzeSubmission.objects.create(
        code_hash="fakehash123",
        raw_code="int main() {}",
        confidence=0.95
    )
    assert submission.code_hash == "fakehash123"
    assert BronzeSubmission.objects.count() == 1
