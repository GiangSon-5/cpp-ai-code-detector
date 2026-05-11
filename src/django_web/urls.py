"""
Root URL configuration for Django Web.
"""

from django.contrib import admin
from django.urls import include, path

from src.django_web.apps.dashboard.views import dashboard_view

urlpatterns = [
    path("admin/", admin.site.urls),
    path("accounts/", include("src.django_web.apps.accounts.urls")),
    path("submit/", include("src.django_web.apps.submissions.urls")),
    path("dashboard/", include("src.django_web.apps.dashboard.urls")),
    path("", dashboard_view, name="root_dashboard"),  # Root → dashboard view directly
]
