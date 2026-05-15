"""
accounts/views.py — User authentication views with deep logging.
"""

import time

from django.contrib import messages
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from django.shortcuts import redirect, render

from src.django_web.apps.accounts.models import UserProfile
from src.shared.logger import AppLogger

logger = AppLogger()


def login_view(request):
    """Handle user login."""
    t0 = time.perf_counter()

    if request.method == "POST":
        username = request.POST.get("username", "").strip()
        password = request.POST.get("password", "")

        if not username or not password:
            messages.error(request, "Vui lòng nhập đầy đủ thông tin.")
            logger.warning(
                module="accounts.views",
                function="login_view",
                message="Empty credentials submitted",
                latency_ms=(time.perf_counter() - t0) * 1000,
            )
            return render(request, "accounts/login.html")

        user = authenticate(request, username=username, password=password)
        if user is not None:
            login(request, user)
            logger.info(
                module="accounts.views",
                function="login_view",
                message=f"User '{username}' logged in successfully (is_staff={user.is_staff})",
                input_data={"username": username},
                output_data={"user_id": user.id, "is_staff": user.is_staff},
                latency_ms=(time.perf_counter() - t0) * 1000,
            )
            # Redirect: admin → dashboard, user thường → submit page
            next_url = request.GET.get("next", "")
            if user.is_staff:
                # Admin: dùng ?next nếu hợp lệ, không thì vào dashboard
                if next_url and not next_url.startswith("/accounts/"):
                    return redirect(next_url)
                return redirect("/dashboard/")
            else:
                # User thường: luôn vào trang submit (không cho vào dashboard)
                if next_url and not next_url.startswith("/dashboard/") and not next_url.startswith("/accounts/"):
                    return redirect(next_url)
                return redirect("/submit/")
        else:
            messages.error(request, "Sai tên đăng nhập hoặc mật khẩu.")
            logger.warning(
                module="accounts.views",
                function="login_view",
                message=f"Failed login attempt for '{username}'",
                input_data={"username": username},
                latency_ms=(time.perf_counter() - t0) * 1000,
            )

    return render(request, "accounts/login.html")


def register_view(request):
    """Handle user registration."""
    t0 = time.perf_counter()

    if request.method == "POST":
        username = request.POST.get("username", "").strip()
        email = request.POST.get("email", "").strip()
        # register.html uses name="password1" and name="password2"
        password = request.POST.get("password1", "")  # matches <input name="password1">
        password2 = request.POST.get("password2", "")  # matches <input name="password2">
        organization = request.POST.get("organization", "").strip()

        # Validation
        errors = []
        if not username:
            errors.append("Tên đăng nhập không được để trống.")
        if not password:
            errors.append("Mật khẩu không được để trống.")
        if password != password2:
            errors.append("Mật khẩu xác nhận không khớp.")
        if password and len(password) < 6:
            errors.append("Mật khẩu phải có ít nhất 6 ký tự.")
        if User.objects.filter(username=username).exists():
            errors.append("Tên đăng nhập đã tồn tại.")
        if email and User.objects.filter(email=email).exists():
            errors.append("Email đã được sử dụng.")

        if errors:
            for err in errors:
                messages.error(request, err)
            logger.warning(
                module="accounts.views",
                function="register_view",
                message=f"Registration validation failed: {errors}",
                input_data={"username": username},
                latency_ms=(time.perf_counter() - t0) * 1000,
            )
            return render(request, "accounts/register.html")

        try:
            user = User.objects.create_user(
                username=username,
                email=email,
                password=password,
            )
            UserProfile.objects.create(
                user=user,
                organization=organization,
                role="student",
            )
            login(request, user)
            messages.success(request, "Đăng ký thành công! Chào mừng bạn đến với AI code detection system.")
            logger.info(
                module="accounts.views",
                function="register_view",
                message=f"User '{username}' registered successfully",
                input_data={"username": username, "email": email},
                output_data={"user_id": user.id},
                latency_ms=(time.perf_counter() - t0) * 1000,
            )
            # Sau đăng ký, user thường → trang submit để bắt đầu dùng ngay
            return redirect("/submit/")
        except Exception as exc:
            messages.error(request, f"Lỗi đăng ký: {exc}")
            logger.error(
                module="accounts.views",
                function="register_view",
                error=f"Registration error: {exc}",
                input_data={"username": username},
                latency_ms=(time.perf_counter() - t0) * 1000,
            )

    return render(request, "accounts/register.html")


def logout_view(request):
    """Log user out."""
    t0 = time.perf_counter()
    username = request.user.username if request.user.is_authenticated else "anonymous"
    logout(request)
    logger.info(
        module="accounts.views",
        function="logout_view",
        message=f"User '{username}' logged out",
        latency_ms=(time.perf_counter() - t0) * 1000,
    )
    return redirect("/accounts/login/")


@login_required
def profile_view(request):
    """Display user profile."""
    t0 = time.perf_counter()
    profile, _ = UserProfile.objects.get_or_create(user=request.user)
    logger.info(
        module="accounts.views",
        function="profile_view",
        message=f"Profile viewed for user_id={request.user.id}",
        latency_ms=(time.perf_counter() - t0) * 1000,
    )
    return render(request, "accounts/profile.html", {"profile": profile})
