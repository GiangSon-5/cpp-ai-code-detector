/* ==============================================================
   CodeGuard AI — main.js
   Core UX: toast notifications, shared utilities
   ============================================================== */

// ─── Toast notification system ────────────────────────────────
window.showToast = function(message, type = 'info', duration = 4000) {
  const container = document.getElementById('toast-container');
  if (!container) return;

  const colors = {
    success: { bg: 'rgba(74,222,128,0.12)',  border: 'rgba(74,222,128,0.25)',  text: '#86efac' },
    error:   { bg: 'rgba(248,113,113,0.12)', border: 'rgba(248,113,113,0.25)', text: '#fca5a5' },
    warning: { bg: 'rgba(251,146,60,0.12)',  border: 'rgba(251,146,60,0.25)',  text: '#fdba74' },
    info:    { bg: 'rgba(34,211,238,0.12)',  border: 'rgba(34,211,238,0.25)',  text: '#67e8f9' },
  };
  const c = colors[type] || colors.info;

  const icons = { success: '✓', error: '✕', warning: '⚠', info: 'ℹ' };

  const toast = document.createElement('div');
  toast.style.cssText = `
    background: ${c.bg};
    border: 1px solid ${c.border};
    border-radius: 16px;
    padding: 14px 18px;
    font-size: 0.85rem;
    font-weight: 600;
    color: ${c.text};
    min-width: 260px;
    max-width: 360px;
    display: flex;
    align-items: center;
    gap: 10px;
    animation: slide-up 0.3s ease;
    backdrop-filter: blur(16px);
    cursor: pointer;
  `;
  toast.innerHTML = `<span style="font-size:1rem;">${icons[type]}</span><span style="color:#e2e8f0;">${message}</span>`;
  toast.addEventListener('click', () => toast.remove());
  container.appendChild(toast);
  setTimeout(() => { toast.style.opacity = '0'; toast.style.transition = 'opacity 0.4s'; setTimeout(() => toast.remove(), 400); }, duration);
};

// ─── Auto-dismiss flash messages ──────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  document.querySelectorAll('.alert').forEach(alert => {
    setTimeout(() => {
      alert.style.transition = 'opacity 0.5s';
      alert.style.opacity = '0';
      setTimeout(() => alert.remove(), 500);
    }, 5000);
  });
});

// ─── Active nav highlight fallback ────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  const path = window.location.pathname;
  document.querySelectorAll('nav a[href]').forEach(link => {
    const href = link.getAttribute('href');
    if (href && href !== '/' && path.startsWith(href)) {
      link.style.color = '#22d3ee';
      link.style.borderBottom = '2px solid #22d3ee';
    }
  });
});
