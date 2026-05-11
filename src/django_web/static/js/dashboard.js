/* ==============================================================
   CodeGuard AI — dashboard.js
   Admin: Chart.js live polling, metrics updates
   ============================================================== */

// ─── Fetch admin metrics every 30s ───────────────────────────
function fetchAdminMetrics() {
  fetch('/dashboard/api/metrics/')
    .then(r => r.json())
    .then(data => {
      // KPI cards
      const p95 = document.getElementById('kpi-p95');
      if (p95 && data.inference_p95_ms) p95.textContent = data.inference_p95_ms + 'ms';

      const celery = document.getElementById('kpi-celery');
      if (celery && data.celery_backlog !== undefined) celery.textContent = data.celery_backlog;

      const total = document.getElementById('kpi-total');
      if (total && data.total !== undefined) total.textContent = data.total;

      // Last-updated timestamp
      const ts = document.getElementById('last-updated');
      if (ts) ts.textContent = new Date().toLocaleTimeString('vi-VN');

      // Vram label on metrics page
      const vramLabel = document.getElementById('vram-label');
      if (vramLabel && data.gpu_vram_gb !== undefined) vramLabel.textContent = data.gpu_vram_gb;

      const metricP95 = document.getElementById('metric-p95');
      if (metricP95 && data.inference_p95_ms) metricP95.textContent = data.inference_p95_ms;
    })
    .catch(() => { /* silently fail — admin may be offline */ });
}

// Start polling when DOM ready
document.addEventListener('DOMContentLoaded', () => {
  fetchAdminMetrics();
  setInterval(fetchAdminMetrics, 30000);
});
