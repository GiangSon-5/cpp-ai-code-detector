"""
dashboard/views.py — Gold layer analytics, MLOps admin views.
All admin views require @staff_member_required.
"""

from __future__ import annotations

import time

from django.contrib.admin.views.decorators import staff_member_required
from django.contrib.auth import get_user_model
from django.contrib.auth.decorators import login_required
from django.db.models import Avg, Count, Q
from django.http import JsonResponse
from django.shortcuts import render

from src.django_web.apps.submissions.models import BronzeSubmission
from src.shared.logger import AppLogger

logger = AppLogger()
User = get_user_model()


# ──────────────────────────────────────────────────────────────────────────────
# USER-FACING DASHBOARD
# ──────────────────────────────────────────────────────────────────────────────

@login_required
def dashboard_view(request):
    """Main dashboard with aggregate statistics. Admin only."""
    t0 = time.perf_counter()
    user = request.user

    # Non-staff users should use the submit page, not the admin dashboard
    if not user.is_staff:
        from django.shortcuts import redirect as _redirect
        return _redirect("/submit/")

    user_submissions = BronzeSubmission.objects.filter(user=user)
    user_total = user_submissions.count()
    user_ai_count = user_submissions.filter(prediction="AI GENERATED").count()
    user_human_count = user_submissions.filter(prediction="HUMAN WRITTEN").count()
    user_pending = user_submissions.filter(prediction="").count()
    user_avg_confidence = (
        user_submissions.filter(confidence__isnull=False)
        .aggregate(avg=Avg("confidence"))["avg"] or 0.0
    )

    global_total = BronzeSubmission.objects.count()
    global_ai_count = BronzeSubmission.objects.filter(prediction="AI GENERATED").count()
    global_human_count = BronzeSubmission.objects.filter(prediction="HUMAN WRITTEN").count()
    global_avg_confidence = (
        BronzeSubmission.objects.filter(confidence__isnull=False)
        .aggregate(avg=Avg("confidence"))["avg"] or 0.0
    )

    recent_submissions = list(user_submissions.order_by("-timestamp")[:10])
    ai_ratio = (user_ai_count / user_total * 100) if user_total > 0 else 0

    context = {
        "active_nav": "overview",
        "user_stats": {
            "total": user_total,
            "ai_count": user_ai_count,
            "human_count": user_human_count,
            "pending": user_pending,
            "avg_confidence": round(user_avg_confidence, 4),
            "ai_ratio": round(ai_ratio, 1),
        },
        "global_stats": {
            "total": global_total,
            "ai_count": global_ai_count,
            "human_count": global_human_count,
            "avg_confidence": round(global_avg_confidence, 4),
        },
        "recent_submissions": recent_submissions,
        "metrics": {},  # populated by JS polling
    }

    logger.info(
        module="dashboard.views",
        function="dashboard_view",
        message=f"Dashboard loaded for user_id={user.id}",
        latency_ms=(time.perf_counter() - t0) * 1000,
    )
    return render(request, "dashboard/dashboard.html", context)


@login_required
def stats_api_view(request):
    """JSON API for user-facing charts (AJAX)."""
    user = request.user
    submissions = (
        BronzeSubmission.objects.filter(user=user, confidence__isnull=False)
        .order_by("timestamp")[:100]
    )
    data_points = [
        {
            "timestamp": s.timestamp.isoformat(),
            "confidence": float(s.confidence) if s.confidence else 0,
            "prediction": s.prediction,
            "code_hash": s.code_hash[:12],
        }
        for s in submissions
    ]
    return JsonResponse({"data": data_points})


# ──────────────────────────────────────────────────────────────────────────────
# ADMIN — METRICS API (used by JS polling on all admin pages)
# ──────────────────────────────────────────────────────────────────────────────

@staff_member_required
def metrics_api_view(request):
    """JSON endpoint polled by admin JS every 30s.

    Merges Django DB stats with FastAPI /health stats.
    """
    t0 = time.perf_counter()

    # ── Django DB stats ──────────────────────────────────────────
    total = BronzeSubmission.objects.count()
    ai_count = BronzeSubmission.objects.filter(prediction="AI GENERATED").count()
    human_count = BronzeSubmission.objects.filter(prediction="HUMAN WRITTEN").count()
    avg_conf = (
        BronzeSubmission.objects.filter(confidence__isnull=False)
        .aggregate(avg=Avg("confidence"))["avg"] or 0.0
    )
    from django.utils import timezone
    today_start = timezone.now().replace(hour=0, minute=0, second=0, microsecond=0)
    submissions_today = BronzeSubmission.objects.filter(timestamp__gte=today_start).count()

    # ── Celery queue depth ───────────────────────────────────────
    celery_backlog = 0
    try:
        from src.celery_workers.celery_app import celery_app
        insp = celery_app.control.inspect(timeout=0.5)
        reserved = insp.reserved() or {}
        celery_backlog = sum(len(v) for v in reserved.values())
    except Exception:
        pass

    # ── FastAPI /health stats ────────────────────────────────────
    fastapi_health: dict = {}
    try:
        import httpx as _httpx
        with _httpx.Client(timeout=2.0) as client:
            resp = client.get("http://localhost:8001/health")
            if resp.status_code == 200:
                fastapi_health = resp.json()
    except Exception:
        pass

    # ── Redis healthcheck ────────────────────────────────────────
    redis_ok = False
    try:
        import redis as redis_lib
        r = redis_lib.Redis(host="localhost", port=6379, socket_timeout=1)
        r.ping()
        redis_ok = True
    except Exception:
        pass

    data = {
        "total": total,
        "ai_count": ai_count,
        "human_count": human_count,
        "avg_confidence": round(float(avg_conf), 4),
        "submissions_today": submissions_today,
        "celery_backlog": celery_backlog,
        "inference_p95_ms": fastapi_health.get("p95_latency_ms"),
        "last_latency_ms": fastapi_health.get("last_latency_ms"),
        "request_count": fastapi_health.get("request_count", 0),
        "cache_size": fastapi_health.get("cache_size", 0),
        "gpu_vram_used_gb": fastapi_health.get("gpu_vram_used_gb"),
        "gpu_vram_total_gb": fastapi_health.get("gpu_vram_total_gb"),
        "gpu_vram_pct": fastapi_health.get("gpu_vram_pct"),
        "models_loaded": fastapi_health.get("models_loaded", False),
        "gpu_available": fastapi_health.get("gpu_available", False),
        "fastapi_uptime_s": fastapi_health.get("uptime_seconds"),
        "redis_ok": redis_ok,
        "api_latency_ms": round((time.perf_counter() - t0) * 1000, 2),
    }
    return JsonResponse(data)



# ──────────────────────────────────────────────────────────────────────────────
# ADMIN — METRICS PAGE
# ──────────────────────────────────────────────────────────────────────────────

@staff_member_required
def metrics_view(request):
    """Observability stack — real data from DB + FastAPI health."""
    t0 = time.perf_counter()
    from django.utils import timezone
    from datetime import timedelta
    import json

    # ── Basic counts ─────────────────────────────────────────────
    total = BronzeSubmission.objects.count()
    ai_count = BronzeSubmission.objects.filter(prediction="AI GENERATED").count()
    human_count = BronzeSubmission.objects.filter(prediction="HUMAN WRITTEN").count()
    pending = BronzeSubmission.objects.filter(prediction="").count()
    avg_conf = (
        BronzeSubmission.objects.filter(confidence__isnull=False)
        .aggregate(avg=Avg("confidence"))["avg"] or 0.0
    )

    # ── Confidence histogram (10 buckets × 10%) ──────────────────
    conf_data = []
    for i in range(10):
        low, high = i / 10, (i + 1) / 10
        count = BronzeSubmission.objects.filter(
            confidence__gte=low, confidence__lt=high if i < 9 else 1.0001
        ).count()
        conf_data.append(count)
    conf_labels = [f"{i*10}–{(i+1)*10}%" for i in range(10)]

    # ── AI vs Human per day — last 7 days ────────────────────────
    now = timezone.now()
    daily_ai = []
    daily_human = []
    daily_labels = []
    for d in range(6, -1, -1):
        day_start = (now - timedelta(days=d)).replace(hour=0, minute=0, second=0, microsecond=0)
        day_end   = day_start + timedelta(days=1)
        day_ai = BronzeSubmission.objects.filter(
            timestamp__gte=day_start, timestamp__lt=day_end,
            prediction="AI GENERATED"
        ).count()
        day_hm = BronzeSubmission.objects.filter(
            timestamp__gte=day_start, timestamp__lt=day_end,
            prediction="HUMAN WRITTEN"
        ).count()
        daily_ai.append(day_ai)
        daily_human.append(day_hm)
        daily_labels.append((now - timedelta(days=d)).strftime("%d/%m"))

    # ── Model usage (OOP vs NORMAL) from result_json ──────────────
    oop_count = 0
    normal_count = 0
    dl_scores = []
    ml_scores = []
    hybrid_scores = []

    recent_results = BronzeSubmission.objects.filter(
        result_json__isnull=False
    ).exclude(result_json={})

    for sub in recent_results:
        rj = sub.result_json or {}
        model = rj.get("model_used", "")
        if "OOP" in model:
            oop_count += 1
        elif "Normal" in model or "NORMAL" in model:
            normal_count += 1
        if rj.get("dl_score") is not None:
            dl_scores.append(rj["dl_score"])
        if rj.get("ml_score") is not None:
            ml_scores.append(rj["ml_score"])
        if rj.get("hybrid_score") is not None:
            hybrid_scores.append(rj["hybrid_score"])

    avg_dl = round(sum(dl_scores) / len(dl_scores) * 100, 1) if dl_scores else None
    avg_ml = round(sum(ml_scores) / len(ml_scores) * 100, 1) if ml_scores else None
    avg_hybrid = round(sum(hybrid_scores) / len(hybrid_scores) * 100, 1) if hybrid_scores else None

    # ── Cache hit rate: duplicate hashes submitted more than once ─
    from django.db.models import Count as _Count
    dup_hashes = (
        BronzeSubmission.objects
        .values("code_hash")
        .annotate(cnt=_Count("id"))
        .filter(cnt__gt=1)
        .count()
    )
    cache_hit_rate = round(dup_hashes / total * 100, 1) if total > 0 else 0

    # ── Source breakdown (pasted vs file_upload vs batch) ─────────
    pasted = BronzeSubmission.objects.filter(source="pasted_code").count()
    file_up = BronzeSubmission.objects.filter(source="file_upload").count()
    batch = BronzeSubmission.objects.filter(source="batch").count()

    # ── Recent error submissions (pending > 10min) ────────────────
    stale_cutoff = now - timedelta(minutes=10)
    stale_pending = BronzeSubmission.objects.filter(
        prediction="", timestamp__lt=stale_cutoff
    ).count()

    # ── FastAPI health ────────────────────────────────────────────
    fastapi_health: dict = {}
    try:
        import httpx as _httpx
        with _httpx.Client(timeout=2.0) as client:
            resp = client.get("http://localhost:8001/health")
            if resp.status_code == 200:
                fastapi_health = resp.json()
    except Exception:
        pass

    context = {
        "active_nav": "metrics",
        "global_stats": {
            "total": total,
            "ai_count": ai_count,
            "human_count": human_count,
            "pending": pending,
            "avg_confidence": round(float(avg_conf), 4),
            "avg_confidence_pct": round(float(avg_conf) * 100, 1),
        },
        # Histogram
        "conf_data_json": json.dumps(conf_data),
        "conf_labels_json": json.dumps(conf_labels),
        # Daily trend
        "daily_ai_json": json.dumps(daily_ai),
        "daily_human_json": json.dumps(daily_human),
        "daily_labels_json": json.dumps(daily_labels),
        # Model breakdown
        "oop_count": oop_count,
        "normal_count": normal_count,
        "model_total": oop_count + normal_count,
        # Per-model scores
        "avg_dl": avg_dl,
        "avg_ml": avg_ml,
        "avg_hybrid": avg_hybrid,
        # Cache
        "cache_hit_rate": cache_hit_rate,
        "dup_hashes": dup_hashes,
        # Source breakdown
        "src_pasted": pasted,
        "src_file": file_up,
        "src_batch": batch,
        "src_json": json.dumps([pasted, file_up, batch]),
        # Health
        "stale_pending": stale_pending,
        "fastapi_health": fastapi_health,
        "inference_p95_ms": fastapi_health.get("p95_latency_ms"),
        "gpu_vram_used": fastapi_health.get("gpu_vram_used_gb", 0),
        "gpu_vram_total": fastapi_health.get("gpu_vram_total_gb", 4),
        "gpu_vram_pct": fastapi_health.get("gpu_vram_pct", 0),
        "request_count": fastapi_health.get("request_count", 0),
    }

    logger.info(
        module="dashboard.views",
        function="metrics_view",
        message="Metrics page loaded",
        latency_ms=(time.perf_counter() - t0) * 1000,
    )
    return render(request, "dashboard/metrics.html", context)



# ──────────────────────────────────────────────────────────────────────────────
# ADMIN — INFRASTRUCTURE
# ──────────────────────────────────────────────────────────────────────────────

@staff_member_required
def infra_view(request):
    """Infrastructure control panel — real service healthchecks."""
    import os, socket

    def _check_tcp(host: str, port: int, timeout: float = 1.0) -> bool:
        try:
            s = socket.create_connection((host, port), timeout=timeout)
            s.close()
            return True
        except Exception:
            return False

    # ── FastAPI healthcheck ──────────────────────────────────────
    fastapi_health: dict = {}
    fastapi_ok = False
    fastapi_latency = None
    try:
        import httpx as _httpx, time as _time
        t = _time.perf_counter()
        with _httpx.Client(timeout=2.0) as client:
            r = client.get("http://localhost:8001/health")
            fastapi_latency = round((_time.perf_counter() - t) * 1000, 1)
            if r.status_code == 200:
                fastapi_health = r.json()
                fastapi_ok = True
    except Exception:
        pass

    # ── Redis ────────────────────────────────────────────────────
    redis_ok = _check_tcp("localhost", 6379)

    # ── PostgreSQL ───────────────────────────────────────────────
    pg_ok = False
    pg_latency = None
    try:
        import time as _time
        from django.db import connection
        t = _time.perf_counter()
        connection.ensure_connection()
        pg_latency = round((_time.perf_counter() - t) * 1000, 1)
        pg_ok = True
    except Exception:
        pass

    # ── Celery ───────────────────────────────────────────────────
    celery_ok = False
    celery_workers = 0
    try:
        from src.celery_workers.celery_app import celery_app
        insp = celery_app.control.inspect(timeout=1.0)
        ping = insp.ping() or {}
        celery_workers = len(ping)
        celery_ok = celery_workers > 0
    except Exception:
        pass

    # ── Redpanda ─────────────────────────────────────────────────
    redpanda_ok = _check_tcp("localhost", 9092)

    # Build service list
    services = [
        {
            "name": "FastAPI AI Service",
            "description": f"LangGraph Orchestrator — models {'loaded ✓' if fastapi_health.get('models_loaded') else 'not loaded'}",
            "endpoint": "localhost:8001",
            "status": "online" if fastapi_ok else "offline",
            "status_label": f"OK ({fastapi_latency}ms)" if fastapi_ok else "Offline",
            "color": "#22d3ee" if fastapi_ok else "#f87171",
            "bg": "rgba(34,211,238,0.1)" if fastapi_ok else "rgba(248,113,113,0.1)",
            "icon": "🤖",
        },
        {
            "name": "Django Web Server",
            "description": "Auth, Templates, SSE Proxy",
            "endpoint": "localhost:8000",
            "status": "online",
            "status_label": "Serving this page",
            "color": "#60a5fa",
            "bg": "rgba(96,165,250,0.1)",
            "icon": "🌐",
        },
        {
            "name": "PostgreSQL",
            "description": f"Primary DB — Bronze layer{f' ({pg_latency}ms)' if pg_latency else ''}",
            "endpoint": "localhost:5432",
            "status": "online" if pg_ok else "offline",
            "status_label": "Connected" if pg_ok else "Offline",
            "color": "#4ade80" if pg_ok else "#f87171",
            "bg": "rgba(74,222,128,0.1)" if pg_ok else "rgba(248,113,113,0.1)",
            "icon": "🐘",
        },
        {
            "name": "Redis",
            "description": "Celery broker + result backend",
            "endpoint": "localhost:6379",
            "status": "online" if redis_ok else "offline",
            "status_label": "Connected" if redis_ok else "Offline",
            "color": "#fb923c" if redis_ok else "#64748b",
            "bg": "rgba(251,146,60,0.1)" if redis_ok else "rgba(255,255,255,0.04)",
            "icon": "⚡",
        },
        {
            "name": "Celery Worker",
            "description": f"Background ETL tasks — {celery_workers} worker(s)",
            "endpoint": "celery -A src.celery_workers.celery_app",
            "status": "online" if celery_ok else "warning",
            "status_label": f"{celery_workers} worker(s)" if celery_ok else "No workers",
            "color": "#c084fc" if celery_ok else "#fbbf24",
            "bg": "rgba(192,132,252,0.1)" if celery_ok else "rgba(251,191,36,0.1)",
            "icon": "⚙️",
        },
        {
            "name": "Redpanda",
            "description": "Kafka-compatible message broker (optional)",
            "endpoint": "localhost:9092",
            "status": "online" if redpanda_ok else "warning",
            "status_label": "Running" if redpanda_ok else "Not started",
            "color": "#fbbf24" if redpanda_ok else "#64748b",
            "bg": "rgba(251,191,36,0.1)" if redpanda_ok else "rgba(255,255,255,0.04)",
            "icon": "📨",
        },
    ]

    obs_tools = [
        {
            "icon": "📊",
            "name": "FastAPI Swagger",
            "desc": "Live API docs + manual testing",
            "url": "http://localhost:8001/docs",
        },
        {
            "icon": "📈",
            "name": "FastAPI ReDoc",
            "desc": "Full API reference documentation",
            "url": "http://localhost:8001/redoc",
        },
        {
            "icon": "🛠️",
            "name": "Django Admin",
            "desc": "Built-in model admin interface",
            "url": "/admin/",
        },
    ]

    context = {
        "active_nav": "infra",
        "services": services,
        "obs_tools": obs_tools,
        "healing_events": [],
        "fastapi_health": fastapi_health,
    }
    return render(request, "dashboard/infra.html", context)



# ──────────────────────────────────────────────────────────────────────────────
# ADMIN — DATABASE EXPLORER
# ──────────────────────────────────────────────────────────────────────────────

@staff_member_required
def database_view(request):
    """Database explorer — all submissions."""
    t0 = time.perf_counter()

    all_submissions = (
        BronzeSubmission.objects.select_related("user")
        .order_by("-timestamp")[:200]
    )
    total = BronzeSubmission.objects.count()
    with_results = BronzeSubmission.objects.exclude(prediction="").count()
    ai_count = BronzeSubmission.objects.filter(prediction="AI GENERATED").count()
    human_count = BronzeSubmission.objects.filter(prediction="HUMAN WRITTEN").count()
    pending = BronzeSubmission.objects.filter(prediction="").count()
    avg_conf = (
        BronzeSubmission.objects.filter(confidence__isnull=False)
        .aggregate(avg=Avg("confidence"))["avg"] or 0.0
    )

    # Build schema info for display
    schema_bronze = [
        {"name": "id", "type": "BIGINT PK"},
        {"name": "code_hash", "type": "VARCHAR(64) UNIQUE"},
        {"name": "user_id", "type": "INT FK → auth_user"},
        {"name": "raw_code", "type": "TEXT"},
        {"name": "language", "type": "VARCHAR(10)"},
        {"name": "source", "type": "VARCHAR(50)"},
        {"name": "prediction", "type": "VARCHAR(50)"},
        {"name": "confidence", "type": "FLOAT NULL"},
        {"name": "file_size_bytes", "type": "INT"},
        {"name": "timestamp", "type": "TIMESTAMPTZ"},
    ]

    context = {
        "active_nav": "database",
        "all_submissions": all_submissions,
        "db_stats": {
            "total": total,
            "with_results": with_results,
            "ai_count": ai_count,
            "human_count": human_count,
            "pending": pending,
            "avg_confidence": round(float(avg_conf), 4),
        },
        "schema_bronze": schema_bronze,
    }

    logger.info(
        module="dashboard.views",
        function="database_view",
        message=f"DB explorer loaded: {total} records",
        latency_ms=(time.perf_counter() - t0) * 1000,
    )
    return render(request, "dashboard/database.html", context)


# ──────────────────────────────────────────────────────────────────────────────
# ADMIN — USER MANAGEMENT
# ──────────────────────────────────────────────────────────────────────────────

@staff_member_required
def users_view(request):
    """User management panel."""
    t0 = time.perf_counter()
    users = User.objects.all().order_by("-date_joined")

    context = {
        "active_nav": "users",
        "users": users,
    }

    logger.info(
        module="dashboard.views",
        function="users_view",
        message=f"Users panel: {users.count()} users",
        latency_ms=(time.perf_counter() - t0) * 1000,
    )
    return render(request, "dashboard/users.html", context)


# ──────────────────────────────────────────────────────────────────────────────
# ADMIN — MODEL REGISTRY
# ──────────────────────────────────────────────────────────────────────────────

@staff_member_required
def models_view(request):
    """Model registry panel."""
    import os
    t0 = time.perf_counter()

    # Try to load GoldModelPerformance if available
    model_history = []
    try:
        from src.django_web.apps.submissions.models import GoldModelPerformance
        model_history = list(GoldModelPerformance.objects.order_by("-timestamp")[:20])
    except Exception:
        pass

    context = {
        "active_nav": "models",
        "active_model": {
            "alpha": 0.48,
            "threshold": 0.5,
            "val_accuracy": "92.4%",
        },
        "model_history": model_history,
        "colab_vram": 0,
        "fastapi_url": os.environ.get("FASTAPI_AI_URL", "http://localhost:8001"),
    }

    logger.info(
        module="dashboard.views",
        function="models_view",
        message="Model registry loaded",
        latency_ms=(time.perf_counter() - t0) * 1000,
    )
    return render(request, "dashboard/models.html", context)
