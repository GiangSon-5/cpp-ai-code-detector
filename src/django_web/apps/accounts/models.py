"""
accounts/models.py — User profile extension.
"""

from django.contrib.auth.models import User
from django.db import models


class UserProfile(models.Model):
    """Extended user profile for the C++ AI Code Detector."""

    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name="profile")
    organization = models.CharField(max_length=200, blank=True, default="")
    role = models.CharField(
        max_length=20,
        choices=[("student", "Student"), ("developer", "Developer"), ("admin", "Admin")],
        default="student",
    )
    total_submissions = models.IntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        app_label = "accounts"

    def __str__(self):
        return f"{self.user.username} ({self.role})"
