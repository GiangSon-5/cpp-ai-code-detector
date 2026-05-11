"""accounts/admin.py"""

from django.contrib import admin
from src.django_web.apps.accounts.models import UserProfile


@admin.register(UserProfile)
class UserProfileAdmin(admin.ModelAdmin):
    list_display = ("user", "role", "organization", "total_submissions", "created_at")
    list_filter = ("role",)
    search_fields = ("user__username", "organization")
