"""
submissions/views.py — Code submission, result display, and history views.

All views have deep logging with input/output/latency.
"""

from __future__ import annotations

import base64
import json
import time

import httpx
import threading
from django.conf import settings as django_settings
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse, StreamingHttpResponse
from django.shortcuts import get_object_or_404, redirect, render

from src.django_web.apps.submissions.models import BronzeSubmission, BatchSession
from src.django_web.apps.submissions.repositories import BronzeRepository
from src.shared.logger import AppLogger

logger = AppLogger()

# Max upload size (500 KB)
MAX_CODE_SIZE = 500 * 1024


@login_required
def submit_view(request):
    """Display submission form and handle code submission."""
    t0 = time.perf_counter()

    if request.method == "POST":
        raw_code = ""

        uploaded_files = request.FILES.getlist("file_upload")
        ai_url = "http://127.0.0.1:8001"
        repo = BronzeRepository()
        # Detect AJAX request — JS frontend sends this header
        is_ajax = request.headers.get("X-Requested-With") == "XMLHttpRequest"

        # --- BATCH UPLOAD (Multiple files) ---
        if len(uploaded_files) > 1:
            batch_name = request.POST.get("batch_name", f"Batch {time.strftime('%Y-%m-%d %H:%M')}")
            batch = BatchSession.objects.create(
                name=batch_name,
                user=request.user,
                total_files=len(uploaded_files)
            )
            
            for f in uploaded_files:
                if f.size > MAX_CODE_SIZE:
                    continue
                try:
                    raw_code = f.read().decode("utf-8")
                    repo.save(user=request.user, raw_code=raw_code, source="batch", filename=f.name, batch=batch)
                except Exception:
                    pass
            
            # Process in background thread
            def _process_batch(b_id):
                b = BatchSession.objects.get(id=b_id)
                subs = b.submissions.filter(prediction="")
                with httpx.Client(timeout=120.0) as client:
                    for sub in subs:
                        try:
                            code_b64 = base64.b64encode(sub.raw_code.encode("utf-8")).decode("ascii")
                            resp = client.post(f"{ai_url}/api/analyze", json={"code_base64": code_b64}, headers={"Content-Type": "application/json"})
                            if resp.status_code == 200:
                                res_data = resp.json()
                                repo.update_result(code_hash=sub.code_hash, prediction=res_data.get("final_pred",""), confidence=res_data.get("final_score",0.0), result_json=res_data)
                        except Exception:
                            pass
            
            threading.Thread(target=_process_batch, args=(batch.id,)).start()
            messages.success(request, f"Đã tải lên {len(uploaded_files)} files. Đang xử lý nền...")
            if is_ajax:
                return JsonResponse({"batch_id": batch.id, "redirect": f"/submit/batch/{batch.id}/"})
            return redirect("submissions:batch_result", batch_id=batch.id)

        # --- SINGLE UPLOAD ---
        if uploaded_files:
            uploaded_file = uploaded_files[0]
            if uploaded_file.size > MAX_CODE_SIZE:
                if is_ajax:
                    return JsonResponse({"error": f"File quá lớn ({uploaded_file.size / 1024:.0f}KB). Giới hạn 500KB."}, status=400)
                messages.error(request, f"File quá lớn ({uploaded_file.size / 1024:.0f}KB). Giới hạn 500KB.")
                logger.warning(
                    module="submissions.views",
                    function="submit_view",
                    message=f"File too large: {uploaded_file.size} bytes",
                    latency_ms=(time.perf_counter() - t0) * 1000,
                )
                return render(request, "submissions/submit.html")
            try:
                raw_code = uploaded_file.read().decode("utf-8")
                filename = uploaded_file.name
                source = "file_upload"
            except UnicodeDecodeError:
                if is_ajax:
                    return JsonResponse({"error": "File không phải text UTF-8."}, status=400)
                messages.error(request, "File không phải text UTF-8.")
                return render(request, "submissions/submit.html")
        else:
            raw_code = request.POST.get("code_input", "")
            filename = "Pasted Code"
            source = "pasted_code"

        # Validate
        if not raw_code.strip():
            if is_ajax:
                return JsonResponse({"error": "Vui lòng nhập code hoặc upload file."}, status=400)
            messages.error(request, "Vui lòng nhập code hoặc upload file.")
            logger.warning(
                module="submissions.views",
                function="submit_view",
                message="Empty code submitted",
                latency_ms=(time.perf_counter() - t0) * 1000,
            )
            return render(request, "submissions/submit.html")

        if len(raw_code.encode("utf-8")) > MAX_CODE_SIZE:
            if is_ajax:
                return JsonResponse({"error": "Code quá lớn. Giới hạn 500KB."}, status=400)
            messages.error(request, "Code quá lớn. Giới hạn 500KB.")
            return render(request, "submissions/submit.html")

        # Save to Bronze
        submission, created = repo.save(
            user=request.user,
            raw_code=raw_code,
            source=source,
            filename=filename
        )

        # If duplicate with existing result → show cached result immediately
        if not created and submission.result_json:
            logger.info(
                module="submissions.views",
                function="submit_view",
                message=f"Duplicate submission, showing cached result for {submission.code_hash[:12]}",
                latency_ms=(time.perf_counter() - t0) * 1000,
            )
            if is_ajax:
                return JsonResponse({"hash": submission.code_hash, "status": "cached"})
            return redirect("submissions:result", code_hash=submission.code_hash)

        # ── AJAX path: return hash immediately, SSE will handle analysis ──
        if is_ajax:
            logger.info(
                module="submissions.views",
                function="submit_view",
                message=f"AJAX submit: returning hash {submission.code_hash[:12]} for SSE analysis",
                latency_ms=(time.perf_counter() - t0) * 1000,
            )
            return JsonResponse({"hash": submission.code_hash, "status": "pending"})

        # ── Fallback sync path (non-AJAX, e.g. direct form submit) ──
        try:
            code_base64 = base64.b64encode(raw_code.encode("utf-8")).decode("ascii")
            # Django calls the local brain, local brain calls Colab
            ai_url = "http://127.0.0.1:8001"

            with httpx.Client(timeout=120.0) as client:
                response = client.post(
                    f"{ai_url}/api/analyze",
                    json={"code_base64": code_base64},
                    headers={"Content-Type": "application/json"},
                )

            if response.status_code == 200:
                result_data = response.json()
                # Update Bronze with result
                repo.update_result(
                    code_hash=submission.code_hash,
                    prediction=result_data.get("final_pred", ""),
                    confidence=result_data.get("final_score", 0.0),
                    result_json=result_data,
                )
                logger.info(
                    module="submissions.views",
                    function="submit_view",
                    message=f"Analysis complete: {result_data.get('final_pred')} ({result_data.get('final_score', 0):.4f})",
                    input_data={"code_hash": submission.code_hash[:12]},
                    output_data={"prediction": result_data.get("final_pred")},
                    latency_ms=(time.perf_counter() - t0) * 1000,
                )
                return redirect("submissions:result", code_hash=submission.code_hash)
            else:
                error_detail = response.text[:200]
                messages.error(request, f"AI Service lỗi ({response.status_code}): {error_detail}")
                logger.error(
                    module="submissions.views",
                    function="submit_view",
                    error=f"FastAPI returned {response.status_code}: {error_detail}",
                    latency_ms=(time.perf_counter() - t0) * 1000,
                )

        except httpx.ConnectError:
            messages.error(request, "Không thể kết nối AI Service. Vui lòng thử lại sau.")
            logger.error(
                module="submissions.views",
                function="submit_view",
                error="FastAPI connection refused",
                latency_ms=(time.perf_counter() - t0) * 1000,
            )
        except httpx.TimeoutException:
            messages.error(request, "AI Service timeout. Vui lòng thử lại.")
            logger.error(
                module="submissions.views",
                function="submit_view",
                error="FastAPI timeout",
                latency_ms=(time.perf_counter() - t0) * 1000,
            )
        except Exception as exc:
            messages.error(request, f"Lỗi không xác định: {exc}")
            logger.error(
                module="submissions.views",
                function="submit_view",
                error=f"Unexpected error: {exc}",
                latency_ms=(time.perf_counter() - t0) * 1000,
            )

        # If AI service failed, redirect to pending page
        return redirect("submissions:result", code_hash=submission.code_hash)

    return render(request, "submissions/submit.html")


@login_required
def result_view(request, code_hash):
    """Display analysis result for a submission."""
    t0 = time.perf_counter()
    submission = get_object_or_404(BronzeSubmission, code_hash=code_hash)

    result = submission.result_json or {}
    if result:
        fs = result.get("final_score", 0)
        result["final_score_pct"] = fs * 100

        # Individual model score percentages (backward-compat: fallback to final_score)
        dl  = result.get("dl_score",     fs)
        ml  = result.get("ml_score",     result.get("fingerprint", {}).get("lgbm_score", 0.5) if result.get("fingerprint") else fs)
        hyb = result.get("hybrid_score", round(0.6 * dl + 0.4 * ml, 4))
        result["dl_score_pct"]     = round(dl  * 100, 1)
        result["ml_score_pct"]     = round(ml  * 100, 1)
        result["hybrid_score_pct"] = round(hyb * 100, 1)
        result["ml_dl_gap_pct"]     = round(result.get("ml_dl_gap", 0.0) * 100, 1)

        # Calculate max absolute SHAP value for rendering scale
        fingerprint = result.get("fingerprint", {})
        if fingerprint:
            top_features = fingerprint.get("top_features", [])
            max_shap = max(abs(feat.get("shap_value", 0.0)) for feat in top_features) if top_features else 1.0
            if max_shap == 0.0:
                max_shap = 1.0
            result["max_shap"] = max_shap
            
            for feat in top_features:
                sval = feat.get("shap_value", 0.0)
                feat["bar_width"] = round((abs(sval) / max_shap) * 100, 1)
                feat["is_positive"] = sval > 0

        # Precompute ML/DL info for template context
        ml_score_pct = result.get("ml_score_pct", 0.0)
        result["ml_human_pct"] = round(100.0 - ml_score_pct, 1)
        result["ml_pred_label"] = "AI GENERATED" if ml_score_pct >= 50.0 else "HUMAN WRITTEN"
        
        dl_score_pct = result.get("dl_score_pct", 0.0)
        result["dl_human_pct"] = round(100.0 - dl_score_pct, 1)
        result["dl_pred_label"] = "AI GENERATED" if dl_score_pct >= 50.0 else "HUMAN WRITTEN"
        
        # Calculate distance from threshold for DL
        dl_threshold_gap = dl_score_pct - 50.0
        result["dl_threshold_gap"] = round(dl_threshold_gap, 1)
        result["dl_threshold_gap_abs"] = round(abs(dl_threshold_gap), 1)
        
        # Check router type (OOP vs Normal)
        model_used = result.get("model_used", "")
        classification = result.get("classification", "")
        result["is_oop"] = "OOP" in model_used or "OOP" in classification

        for chunk in result.get("chunks", []):
            chunk["score_pct"] = chunk.get("score", 0) * 100

    logger.info(
        module="submissions.views",
        function="result_view",
        message=f"Viewing result for {code_hash[:12]}",
        input_data={"code_hash": code_hash},
        output_data={"has_result": bool(result)},
        latency_ms=(time.perf_counter() - t0) * 1000,
    )

    return render(request, "submissions/result.html", {
        "submission": submission,
        "result": result,
        "chunks": result.get("chunks", []),
        "raw_code": submission.raw_code,  # for source code panel
    })



@login_required
def history_view(request):
    """Display submission history for the current user."""
    t0 = time.perf_counter()
    repo = BronzeRepository()
    # Get normal submissions (not part of a batch)
    single_submissions = BronzeSubmission.objects.filter(user=request.user, batch__isnull=True).order_by("-timestamp")[:50]
    
    # Get batch sessions
    batch_sessions = BatchSession.objects.filter(user=request.user).order_by("-timestamp")[:20]

    logger.info(
        module="submissions.views",
        function="history_view",
        message=f"History viewed: {len(single_submissions)} single, {len(batch_sessions)} batches",
        input_data={"user_id": request.user.id},
        latency_ms=(time.perf_counter() - t0) * 1000,
    )

    return render(request, "submissions/history.html", {
        "submissions": single_submissions,
        "batch_sessions": batch_sessions,
    })

@login_required
def batch_result_view(request, batch_id):
    """Display results for a batch session."""
    t0 = time.perf_counter()
    batch = get_object_or_404(BatchSession, id=batch_id, user=request.user)
    
    # Calculate stats
    submissions = batch.submissions.all()
    total = submissions.count()
    ai_count = submissions.filter(prediction="AI GENERATED").count()
    human_count = submissions.filter(prediction="HUMAN WRITTEN").count()
    pending = total - ai_count - human_count

    return render(request, "submissions/batch_result.html", {
        "batch": batch,
        "submissions": submissions,
        "stats": {
            "total": total,
            "ai_count": ai_count,
            "human_count": human_count,
            "pending": pending,
        }
    })


@login_required
def status_api_view(request, code_hash):
    """Lightweight JSON endpoint for polling analysis status from result page."""
    submission = get_object_or_404(BronzeSubmission, code_hash=code_hash)
    ready = bool(submission.result_json)
    return JsonResponse({
        "ready": ready,
        "prediction": submission.prediction if ready else None,
    })


@login_required
def sse_proxy_view(request, code_hash):
    """Proxy SSE events from FastAPI to the browser."""
    t0 = time.perf_counter()

    submission = get_object_or_404(BronzeSubmission, code_hash=code_hash)
    code_base64 = base64.b64encode(submission.raw_code.encode("utf-8")).decode("ascii")
    # Django calls the local brain, local brain calls Colab
    ai_url = "http://127.0.0.1:8001"

    def event_stream():
        try:
            with httpx.Client(timeout=180.0) as client:
                with client.stream(
                    "POST",
                    f"{ai_url}/api/analyze_stream",
                    json={"code_base64": code_base64},
                ) as response:
                    for line in response.iter_lines():
                        if line.startswith("data:"):
                            yield f"{line}\n\n"

                            # Check if final event
                            try:
                                payload = json.loads(line[5:].strip())
                                if payload.get("step") == "complete" and payload.get("data"):
                                    # Update Bronze with result
                                    result_data = payload["data"]
                                    BronzeRepository.update_result(
                                        code_hash=submission.code_hash,
                                        prediction=result_data.get("final_pred", ""),
                                        confidence=result_data.get("final_score", 0.0),
                                        result_json=result_data,
                                    )
                            except (json.JSONDecodeError, KeyError):
                                pass
        except Exception as exc:
            error_event = json.dumps({"step": "error", "progress": 0, "message": str(exc)})
            yield f"data: {error_event}\n\n"
            logger.error(
                module="submissions.views",
                function="sse_proxy_view",
                error=f"SSE proxy error: {exc}",
                latency_ms=(time.perf_counter() - t0) * 1000,
            )

    logger.info(
        module="submissions.views",
        function="sse_proxy_view",
        message=f"SSE proxy started for {code_hash[:12]}",
        latency_ms=(time.perf_counter() - t0) * 1000,
    )

    return StreamingHttpResponse(
        event_stream(),
        content_type="text/event-stream",
    )
