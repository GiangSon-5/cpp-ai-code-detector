"""submissions/admin.py"""

from django.contrib import admin
from src.django_web.apps.submissions.models import BronzeSubmission


@admin.register(BronzeSubmission)
class BronzeSubmissionAdmin(admin.ModelAdmin):
    list_display = ("code_hash", "user", "prediction", "confidence", "source", "file_size_bytes", "timestamp")
    list_filter = ("prediction", "source", "language")
    search_fields = ("code_hash", "user__username")
    readonly_fields = ("code_hash", "timestamp")
    ordering = ("-timestamp",)


