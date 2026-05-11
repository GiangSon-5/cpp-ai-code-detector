"""
Root URL configuration for Django Web.
"""

from django.contrib import admin
from django.contrib.auth.decorators import login_required
from django.shortcuts import redirect
from django.urls import include, path


@login_required
def root_redirect(request):
    """Smart root redirect: admin → dashboard, regular user → submit."""
    if request.user.is_staff:
        return redirect("/dashboard/")
    return redirect("/submit/")


urlpatterns = [
    path("admin/", admin.site.urls),
    path("accounts/", include("src.django_web.apps.accounts.urls")),
    path("submit/", include("src.django_web.apps.submissions.urls")),
    path("dashboard/", include("src.django_web.apps.dashboard.urls")),
    path("", root_redirect, name="root_redirect"),  # Smart root redirect
]

