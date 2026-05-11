"""
submissions/models.py — Bronze layer ORM model.

Schema matches metadata_implementation_plan.md exactly.
"""

from django.contrib.auth.models import User
from django.db import models


class BronzeSubmission(models.Model):
    """Raw C++ code submission — Bronze layer (PostgreSQL)."""

    code_hash = models.CharField(max_length=64, unique=True, db_index=True)
    user = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True, related_name="submissions")
    raw_code = models.TextField()
    language = models.CharField(max_length=10, default="cpp")
    source = models.CharField(max_length=50, default="web_upload")  # web_upload | api | batch
    file_size_bytes = models.IntegerField(default=0)
    timestamp = models.DateTimeField(auto_now_add=True)
    schema_version = models.CharField(max_length=10, default="1.0")

    # Cached result (populated after FastAPI returns)
    prediction = models.CharField(max_length=20, blank=True, default="")
    confidence = models.FloatField(null=True, blank=True)
    result_json = models.JSONField(null=True, blank=True)
    
    # Batch Support
    filename = models.CharField(max_length=255, blank=True, default="")
    batch = models.ForeignKey('BatchSession', on_delete=models.CASCADE, null=True, blank=True, related_name="submissions")

    class Meta:
        app_label = "submissions"
        ordering = ["-timestamp"]
        verbose_name = "Bronze Submission"
        verbose_name_plural = "Bronze Submissions"

    def __str__(self):
        return f"[{self.code_hash[:12]}] {self.prediction or 'pending'} ({self.timestamp:%Y-%m-%d %H:%M})"

class BatchSession(models.Model):
    """A batch of code submissions uploaded together."""
    id = models.BigAutoField(primary_key=True)
    name = models.CharField(max_length=255, help_text="Batch name or folder name")
    user = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True, related_name="batch_sessions")
    total_files = models.IntegerField(default=0)
    timestamp = models.DateTimeField(auto_now_add=True)

    class Meta:
        app_label = "submissions"
        ordering = ["-timestamp"]
        verbose_name = "Batch Session"
        verbose_name_plural = "Batch Sessions"

    def __str__(self):
        return f"[{self.id}] {self.name} ({self.total_files} files)"
