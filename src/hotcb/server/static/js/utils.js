/**
 * hotcb dashboard — shared utilities
 */

function $(sel) { return document.querySelector(sel); }
function $$(sel) { return document.querySelectorAll(sel); }

async function api(method, path, body) {
  const opts = { method, headers: {'Content-Type': 'application/json'} };
  if (body) opts.body = JSON.stringify(body);
  try {
    const r = await fetch(path, opts);
    if (!r.ok) {
      console.warn('API', r.status, path);
      try { return await r.json(); } catch(_) { return null; }
    }
    return await r.json();
  } catch (e) { console.error('API error:', path, e); return null; }
}

function openModal(id) { document.getElementById(id).classList.add('open'); }
function closeModal(id) { document.getElementById(id).classList.remove('open'); }

function fmtNum(v, prec) {
  prec = prec || 4;
  if (v === null || v === undefined) return '--';
  if (Math.abs(v) < 0.01 || Math.abs(v) > 1e4) return v.toExponential(2);
  return v.toFixed(prec);
}

// Make closeModal globally available for inline onclick handlers
window.closeModal = closeModal;
