"""dashboard/urls.py"""

from django.urls import path
from src.django_web.apps.dashboard import views

app_name = "dashboard"

urlpatterns = [
    # User-facing
    path("", views.dashboard_view, name="dashboard"),
    path("api/stats/", views.stats_api_view, name="stats_api"),

    # Admin-only (require @staff_member_required inside views)
    path("api/metrics/", views.metrics_api_view, name="metrics_api"),
    path("metrics/", views.metrics_view, name="metrics"),
    path("infra/", views.infra_view, name="infra"),
    path("database/", views.database_view, name="database"),
    path("users/", views.users_view, name="users"),
    path("models/", views.models_view, name="models"),
]
