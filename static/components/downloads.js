// Downloads dashboard — folder-per-tag view for exporting images.
//
// Folder structure is fixed (tag_id → images); filters just narrow which
// tags render. Per-tag download, per-selection download, or a
// "download everything shown" button.

import { api } from "../api.js";
import { toast } from "./toast.js";
import {
  html, raw, escapeHtml, refreshIcons, icon,
  decisionChip, statusChip, formatDate,
} from "../lib/format.js";


export async function mountDownloads(el) {
  el.innerHTML = html`
    <div class="page-header">
      <div>
        <h1>Downloads</h1>
        <p class="muted" style="margin:4px 0 0">
          One folder per tag. Filter the list, pick what you want, and download a zip
          with the original filenames preserved.
        </p>
      </div>
      <div class="row" id="header-actions" style="gap:8px"></div>
    </div>

    <div class="twocol">
      <aside class="card filter-rail" aria-label="Filters">
        <h3>Filters</h3>
        <div class="group">
          <div class="muted" style="margin-bottom:6px">Batch</div>
          <select class="select" id="f-batch"><option value="">All batches</option></select>
        </div>
        <div class="group">
          <div class="muted" style="margin-bottom:6px">Status</div>
          ${raw(filterChecks("status", [
            { v: "pending",       l: "Pending" },
            { v: "processing",    l: "Processing" },
            { v: "completed",     l: "Completed" },
            { v: "failed",        l: "Failed" },
            { v: "manual_review", l: "Manual review" },
          ]))}
        </div>
        <div class="group">
          <div class="muted" style="margin-bottom:6px">Decision</div>
          ${raw(filterChecks("decision", [
            { v: "approved",       l: "Approved" },
            { v: "auto_approved",  l: "Auto-approved" },
            { v: "manual_review",  l: "Manual review" },
            { v: "rejected",       l: "Rejected" },
            { v: "pending",        l: "Pending" },
          ]))}
        </div>
        <div class="group">
          <div class="muted" style="margin-bottom:6px">HUID match</div>
          ${raw(filterChecks("match", [
            { v: "yes", l: "Match" },
            { v: "no",  l: "Mismatch" },
          ]))}
        </div>
        <div class="group">
          <div class="muted" style="margin-bottom:6px">Search tag_id</div>
          <div class="search" style="min-width:0"><input id="f-q" placeholder="TAG…" /></div>
        </div>
      </aside>

      <div>
        <div class="row" id="bulk-bar" style="justify-content:space-between; margin-bottom: 12px; gap:8px;">
          <div class="row" style="gap:8px">
            <label class="row" style="gap:6px; font-size:13px;">
              <input type="checkbox" id="select-all" />
              <span>Select all visible</span>
            </label>
            <span class="muted" id="selection-label">0 selected</span>
          </div>
          <div class="row" style="gap:8px">
            <span class="muted" id="totals-label" style="font-size:12px">Loading…</span>
            <button class="btn btn-secondary btn-sm" id="download-selected" disabled>${icon("package")}<span>Download selected</span></button>
            <button class="btn btn-primary btn-sm" id="download-visible">${icon("download")}<span>Download all visible</span></button>
          </div>
        </div>

        <div id="folders"></div>
      </div>
    </div>
  `;
  refreshIcons();

  const state = {
    filters: { batch_id: "", status: new Set(), decision: new Set(), match: new Set(), q: "" },
    tags: [],
    selection: new Set(),
    loading: false,
  };

  await loadBatchOptions(el);
  wireFilters(el, state);
  wireBulkBar(el, state);

  await reload(el, state);
}

// --- helpers --------------------------------------------------------------

function filterChecks(group, options) {
  return options
    .map(o => `<label><input type="checkbox" data-group="${group}" value="${escapeHtml(o.v)}" /> ${escapeHtml(o.l)}</label>`)
    .join("");
}

async function loadBatchOptions(root) {
  try {
    const resp = await api.get("/stage1/batches");
    const batches = (resp && (resp.batches || resp)) || [];
    const select = root.querySelector("#f-batch");
    batches.forEach(b => {
      const opt = document.createElement("option");
      opt.value = b.id;
      opt.textContent = `#${b.id} · ${b.batch_name}`;
      select.appendChild(opt);
    });
  } catch { /* non-fatal */ }
}

function wireFilters(root, state) {
  const schedule = debounce(() => reload(root, state), 200);

  root.querySelector("#f-batch").addEventListener("change", (e) => {
    state.filters.batch_id = e.target.value;
    schedule();
  });
  root.querySelectorAll(".filter-rail input[type='checkbox']").forEach(cb => {
    cb.addEventListener("change", () => {
      const set = state.filters[cb.dataset.group];
      if (cb.checked) set.add(cb.value); else set.delete(cb.value);
      schedule();
    });
  });
  root.querySelector("#f-q").addEventListener("input", (e) => {
    state.filters.q = e.target.value;
    schedule();
  });
}

function wireBulkBar(root, state) {
  root.querySelector("#select-all").addEventListener("change", (e) => {
    if (e.target.checked) {
      state.tags.forEach(t => state.selection.add(t.tag_id));
    } else {
      state.tags.forEach(t => state.selection.delete(t.tag_id));
    }
    renderFolders(root, state);
    updateSelectionLabel(root, state);
  });
  root.querySelector("#download-selected").addEventListener("click", () => {
    const tagIds = [...state.selection];
    if (!tagIds.length) return;
    downloadBulk(tagIds);
  });
  root.querySelector("#download-visible").addEventListener("click", () => {
    const tagIds = state.tags.map(t => t.tag_id);
    if (!tagIds.length) { toast.error("Nothing to download"); return; }
    downloadBulk(tagIds);
  });
}

async function reload(root, state) {
  if (state.loading) return;
  state.loading = true;
  const folders = root.querySelector("#folders");
  folders.innerHTML = skeletonRows();
  try {
    const qs = buildQs(state.filters);
    const data = await api.get(`/stage3/downloads/manifest?${qs}`);
    state.tags = data.tags || [];
    // Drop stale selections that no longer apply.
    const visible = new Set(state.tags.map(t => t.tag_id));
    [...state.selection].forEach(id => { if (!visible.has(id)) state.selection.delete(id); });
    renderTotals(root, data);
    renderFolders(root, state);
    updateSelectionLabel(root, state);
  } catch (e) {
    folders.innerHTML = `<div class="empty-state"><div class="icon-wrap"><i data-lucide="alert-circle"></i></div><p>${escapeHtml(e.message || "Failed to load")}</p></div>`;
    refreshIcons();
  } finally {
    state.loading = false;
  }
}

function buildQs(f) {
  const parts = [];
  if (f.batch_id) parts.push(`batch_id=${encodeURIComponent(f.batch_id)}`);
  f.status.forEach(v => parts.push(`status=${encodeURIComponent(v)}`));
  f.decision.forEach(v => parts.push(`decision=${encodeURIComponent(v)}`));
  // Match is single-select semantically — if both are picked we send neither (same as "any").
  if (f.match.size === 1) parts.push(`match=${encodeURIComponent([...f.match][0])}`);
  if (f.q && f.q.trim()) parts.push(`q=${encodeURIComponent(f.q.trim())}`);
  return parts.join("&");
}

function renderTotals(root, data) {
  const tags = data.count || 0;
  const imgs = data.total_images || 0;
  const bytes = data.total_size_bytes || 0;
  root.querySelector("#totals-label").textContent =
    `${tags} ${tags === 1 ? "tag" : "tags"} · ${imgs} images · ${formatBytes(bytes)}`;
  root.querySelector("#download-visible").toggleAttribute("disabled", tags === 0);
}

function renderFolders(root, state) {
  const host = root.querySelector("#folders");
  if (!state.tags.length) {
    host.innerHTML = `<div class="empty-state"><div class="icon-wrap"><i data-lucide="folder-search"></i></div><h2 style="margin-bottom:4px">No tags match</h2><p>Relax a filter or upload more images.</p></div>`;
    refreshIcons(); return;
  }
  host.innerHTML = state.tags.map(t => renderFolder(t, state.selection.has(t.tag_id))).join("");

  host.querySelectorAll("[data-tag-check]").forEach(cb => {
    cb.addEventListener("change", () => {
      const tagId = cb.dataset.tagCheck;
      if (cb.checked) state.selection.add(tagId);
      else state.selection.delete(tagId);
      updateSelectionLabel(root, state);
    });
  });
  host.querySelectorAll("[data-tag-download]").forEach(btn => {
    btn.addEventListener("click", (e) => {
      e.stopPropagation();
      downloadSingle(btn.dataset.tagDownload);
    });
  });
  host.querySelectorAll(".folder-head").forEach(head => {
    head.addEventListener("click", (e) => {
      if (e.target.closest("input, button, a")) return;
      head.closest(".folder").classList.toggle("open");
    });
  });
  refreshIcons();
}

function renderFolder(tag, selected) {
  const size = formatBytes(tag.total_size_bytes || 0);
  const count = tag.image_count || 0;
  const matchChip =
    tag.huid_match === true  ? '<span class="chip chip-success">Match</span>' :
    tag.huid_match === false ? '<span class="chip chip-danger">Mismatch</span>' :
                               '';
  const chips = [
    matchChip,
    decisionChip(tag.decision),
    statusChip(tag.status),
  ].filter(Boolean).join(" ");

  const imageRows = (tag.images || []).map(img => `
    <div class="image-row">
      <span class="image-icon">${icon(img.image_type === "huid" ? "badge-check" : "image").value}</span>
      <span class="image-name mono">${escapeHtml(img.filename)}</span>
      <span class="image-type muted">${escapeHtml(img.image_type)}${img.image_type === "artifact" ? `-${img.slot}` : ""}</span>
      <span class="image-size muted">${formatBytes(img.size_bytes || 0)}</span>
      <a class="btn btn-tertiary btn-sm" href="${escapeHtml(img.url || "")}" target="_blank" rel="noreferrer">
        ${icon("external-link").value}<span>Open</span>
      </a>
    </div>`).join("");

  return `
    <article class="folder ${selected ? "open" : ""}">
      <header class="folder-head">
        <label class="folder-pick">
          <input type="checkbox" data-tag-check="${escapeHtml(tag.tag_id)}" ${selected ? "checked" : ""} />
        </label>
        <span class="folder-icon">${icon("folder").value}</span>
        <div class="folder-title">
          <div class="row" style="gap:8px; align-items: baseline;">
            <strong class="mono">${escapeHtml(tag.tag_id)}</strong>
            <span class="muted" style="font-size:12px">${escapeHtml(tag.batch_name || "")}</span>
          </div>
          <div class="row" style="gap:6px; margin-top:4px; font-size:12px;">${chips}</div>
        </div>
        <span class="folder-meta mono muted">${count} ${count === 1 ? "image" : "images"} · ${size}</span>
        <button class="btn btn-secondary btn-sm" data-tag-download="${escapeHtml(tag.tag_id)}">
          ${icon("download").value}<span>Download</span>
        </button>
        <span class="folder-chevron">${icon("chevron-right").value}</span>
      </header>
      <div class="folder-body">
        ${imageRows || '<div class="muted" style="padding:10px; font-size:12px">No images.</div>'}
      </div>
    </article>`;
}

function updateSelectionLabel(root, state) {
  const n = state.selection.size;
  root.querySelector("#selection-label").textContent = `${n} selected`;
  root.querySelector("#download-selected").disabled = n === 0;
  const allCb = root.querySelector("#select-all");
  if (state.tags.length && state.tags.every(t => state.selection.has(t.tag_id))) {
    allCb.checked = true;
  } else {
    allCb.checked = false;
  }
}

function skeletonRows() {
  return [...Array(4)].map(() =>
    `<article class="folder"><header class="folder-head" style="cursor:default">
      <span class="skeleton" style="width: 70%; height: 18px"></span>
     </header></article>`
  ).join("");
}

// --- download handlers ----------------------------------------------------

async function downloadSingle(tagId) {
  try {
    const resp = await fetch(`/stage3/downloads/zip/${encodeURIComponent(tagId)}`);
    if (!resp.ok) throw new Error(`Download failed (${resp.status})`);
    saveBlob(await resp.blob(), `hallmark-${tagId}.zip`);
    toast.success(`Downloaded ${tagId}`);
  } catch (e) {
    toast.error(e.message || "Download failed");
  }
}

async function downloadBulk(tagIds) {
  const btn = document.getElementById("download-selected") || document.getElementById("download-visible");
  try {
    if (btn) { btn.setAttribute("aria-disabled", "true"); btn.querySelector("span").textContent = "Preparing…"; }
    const resp = await fetch("/stage3/downloads/zip", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ tag_ids: tagIds }),
    });
    if (!resp.ok) {
      const text = await resp.text();
      throw new Error(text || `Download failed (${resp.status})`);
    }
    const stamp = new Date().toISOString().slice(0, 19).replace(/[:T]/g, "-");
    saveBlob(await resp.blob(), `hallmark-images-${stamp}.zip`);
    toast.success(`Downloaded ${tagIds.length} ${tagIds.length === 1 ? "tag" : "tags"}`);
  } catch (e) {
    toast.error(e.message || "Download failed");
  } finally {
    if (btn) {
      btn.removeAttribute("aria-disabled");
      const label = btn.id === "download-selected" ? "Download selected" : "Download all visible";
      btn.querySelector("span").textContent = label;
    }
  }
}

function saveBlob(blob, filename) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url; a.download = filename;
  document.body.appendChild(a); a.click(); a.remove();
  URL.revokeObjectURL(url);
}

// --- misc -----------------------------------------------------------------

function debounce(fn, ms) {
  let t;
  return (...args) => {
    clearTimeout(t);
    t = setTimeout(() => fn(...args), ms);
  };
}

function formatBytes(bytes) {
  if (!bytes) return "0 B";
  const units = ["B", "KB", "MB", "GB"];
  let i = 0, n = bytes;
  while (n >= 1024 && i < units.length - 1) { n /= 1024; i++; }
  return `${n.toFixed(i ? 1 : 0)} ${units[i]}`;
}
