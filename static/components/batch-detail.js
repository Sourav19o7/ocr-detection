import { api } from "../api.js";
import { drawer } from "./drawer.js";
import { mountItemPreview } from "./item-preview.js";
import {
  html, raw, refreshIcons, icon, escapeHtml,
  decisionChip, statusChip,
} from "../lib/format.js";

export async function mountBatchDetail(el, { batchId }) {
  el.innerHTML = html`
    <div class="page-header">
      <div>
        <h1 id="batch-title">Batch</h1>
        <p class="muted" id="batch-sub" style="margin:4px 0 0">Loading…</p>
      </div>
    </div>

    <div class="twocol">
      <aside class="card filter-rail" aria-label="Filters">
        <h3>Filters</h3>
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
          <div class="muted" style="margin-bottom:6px">HUID match</div>
          ${raw(filterChecks("match", [
            { v: "yes", l: "Yes" }, { v: "no", l: "No" },
          ]))}
        </div>
        <div class="group">
          <div class="muted" style="margin-bottom:6px">Search tag_id</div>
          <div class="search" style="min-width:0"><input id="f-q" placeholder="TAG…" /></div>
        </div>
      </aside>

      <div class="table-wrap">
        <table class="table">
          <thead>
            <tr>
              <th></th>
              <th>Tag ID</th>
              <th>Expected</th>
              <th>Actual</th>
              <th>Match</th>
              <th>Purity</th>
              <th>Decision</th>
              <th>Status</th>
            </tr>
          </thead>
          <tbody id="items-body">
            ${raw([...Array(6)].map(()=>`<tr><td colspan="8"><span class="skeleton" style="width:92%"></span></td></tr>`).join(""))}
          </tbody>
        </table>
      </div>
    </div>
  `;

  const data = await api.get(`/stage3/batch/${encodeURIComponent(batchId)}/results`);
  document.getElementById("batch-title").textContent = data.batch_name || `Batch ${batchId}`;
  document.getElementById("batch-sub").textContent =
    `${data.processed_items || 0} processed of ${data.total_items || 0} items`;

  const allRows = data.results || [];
  const body = document.getElementById("items-body");
  const filters = { status: new Set(), match: new Set(), q: "" };

  function render() {
    const rows = allRows.filter(r => {
      if (filters.status.size && !filters.status.has(r.status)) return false;
      if (filters.match.size) {
        const wantYes = filters.match.has("yes");
        const wantNo  = filters.match.has("no");
        if (r.huid_match && !wantYes) return false;
        if (r.huid_match === false && !wantNo) return false;
        if (r.huid_match == null && !(wantYes && wantNo)) return false;
      }
      if (filters.q && !(r.tag_id || "").toLowerCase().includes(filters.q.toLowerCase())) return false;
      return true;
    });

    if (!rows.length) {
      body.innerHTML = `<tr><td colspan="8">
        <div class="empty-state">
          <div class="icon-wrap"><i data-lucide="filter-x"></i></div>
          <h2>No items match the filters</h2>
        </div></td></tr>`;
    } else {
      body.innerHTML = rows.map(renderItemRow).join("");
      body.querySelectorAll("tr[data-tag]").forEach((tr) => {
        tr.addEventListener("click", () => openDrawer(tr.dataset.tag));
        tr.setAttribute("role", "button"); tr.setAttribute("tabindex", "0");
        tr.addEventListener("keydown", (e) => { if (e.key === "Enter") tr.click(); });
      });
    }
    refreshIcons(el);
  }

  // Wire filters
  el.querySelectorAll('.filter-rail input[type="checkbox"]').forEach((cb) => {
    cb.addEventListener("change", () => {
      const set = filters[cb.dataset.group];
      if (cb.checked) set.add(cb.value); else set.delete(cb.value);
      render();
    });
  });
  el.querySelector("#f-q").addEventListener("input", (e) => { filters.q = e.target.value; render(); });

  render();
}

function openDrawer(tagId) {
  drawer.open((wrap) => {
    wrap.innerHTML = `
      <div class="drawer-header">
        <div><strong>${escapeHtml(tagId)}</strong><div class="muted" style="font-size:12px">Preview</div></div>
        <div class="row">
          <a class="btn btn-secondary btn-sm" href="#/item/${encodeURIComponent(tagId)}">Open page</a>
          <button class="icon-btn" aria-label="Close drawer" id="drawer-close"><i data-lucide="x"></i></button>
        </div>
      </div>
      <div id="drawer-body" style="padding: 16px 20px 24px"></div>
    `;
    refreshIcons(wrap);
    wrap.querySelector("#drawer-close").addEventListener("click", () => drawer.close());
    mountItemPreview(wrap.querySelector("#drawer-body"), { tagId, compact: true });
  });
}

function filterChecks(group, options) {
  return options
    .map(o => `<label><input type="checkbox" data-group="${group}" value="${escapeHtml(o.v)}" /> ${escapeHtml(o.l)}</label>`)
    .join("");
}

function renderItemRow(r) {
  const thumb = r.thumbnail_url
    ? `<span class="thumb" style="background-image:url('${escapeHtml(r.thumbnail_url)}')"></span>`
    : `<span class="thumb empty"></span>`;
  const match =
    r.huid_match === true  ? `<span class="chip chip-success">Match</span>` :
    r.huid_match === false ? `<span class="chip chip-danger">Mismatch</span>` :
                             `<span class="chip">—</span>`;
  return `<tr data-tag="${escapeHtml(r.tag_id)}">
    <td>${thumb}</td>
    <td class="mono">${escapeHtml(r.tag_id)}</td>
    <td class="mono">${escapeHtml(r.expected_huid || "")}</td>
    <td class="mono">${escapeHtml(r.actual_huid || "—")}</td>
    <td>${match}</td>
    <td>${escapeHtml(r.purity_code || "—")}</td>
    <td>${decisionChip(r.decision)}</td>
    <td>${statusChip(r.status)}</td>
  </tr>`;
}
