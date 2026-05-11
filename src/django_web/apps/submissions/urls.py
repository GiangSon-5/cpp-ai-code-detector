"""submissions/urls.py"""

from django.urls import path
from src.django_web.apps.submissions import views

app_name = "submissions"

urlpatterns = [
    path("", views.submit_view, name="submit"),
    path("history/", views.history_view, name="history"),
    path("result/<str:code_hash>/", views.result_view, name="result"),
    path("batch/<int:batch_id>/", views.batch_result_view, name="batch_result"),
    path("sse/<str:code_hash>/", views.sse_proxy_view, name="sse_proxy"),
    path("api/status/<str:code_hash>/", views.status_api_view, name="status_api"),
]
