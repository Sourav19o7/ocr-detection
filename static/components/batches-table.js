import { api } from "../api.js";
import { html, raw, refreshIcons, icon, formatDate, escapeHtml } from "../lib/format.js";

export async function mountBatches(el) {
  el.innerHTML = html`
    <div class="page-header">
      <h1>Batches</h1>
    </div>

    <div class="kpi-strip" id="kpi-strip">
      ${raw([...Array(5)].map(() => `<div class="kpi"><span class="skeleton" style="width:60%"></span><div class="value">&nbsp;</div></div>`).join(""))}
    </div>

    <div class="table-wrap">
      <table class="table" id="batches-table">
        <thead>
          <tr>
            <th>Name</th>
            <th>Created</th>
            <th class="numeric">Items</th>
            <th>Progress</th>
            <th>Status</th>
          </tr>
        </thead>
        <tbody id="batches-body">
          ${raw([...Array(5)].map(() => `<tr><td colspan="5"><span class="skeleton" style="width:80%"></span></td></tr>`).join(""))}
        </tbody>
      </table>
    </div>
  `;

  const batches = await api.get("/stage1/batches");
  const list = (batches && (batches.batches || batches)) || [];

  // KPI: for now derive from batches totals. Detailed per-decision KPIs would
  // require joining batch results — we defer that to a lightweight aggregate
  // and keep this view snappy.
  const totals = list.reduce((acc, b) => {
    acc.items += b.total_items || 0;
    acc.processed += b.processed_items || 0;
    acc.batches += 1;
    return acc;
  }, { items: 0, processed: 0, batches: 0 });

  document.getElementById("kpi-strip").innerHTML = [
    kpi("Batches", totals.batches),
    kpi("Items", totals.items),
    kpi("Processed", totals.processed),
    kpi("Pending", Math.max(0, totals.items - totals.processed)),
    kpi("Match rate", "—"),
  ].join("");

  const body = document.getElementById("batches-body");
  if (!list.length) {
    body.innerHTML = `<tr><td colspan="5">
      <div class="empty-state">
        <div class="icon-wrap"><i data-lucide="inbox"></i></div>
        <h2 style="margin-bottom:4px">No batches yet</h2>
        <p>Upload an Excel or CSV to get started.</p>
        <p style="margin-top:12px"><a class="btn btn-primary" href="#/upload">Upload batch</a></p>
      </div></td></tr>`;
  } else {
    body.innerHTML = list.map(renderRow).join("");
    body.querySelectorAll("tr[data-id]").forEach((tr) => {
      tr.addEventListener("click", () => {
        location.hash = `#/batch/${tr.dataset.id}`;
      });
      tr.setAttribute("role", "button");
      tr.setAttribute("tabindex", "0");
      tr.addEventListener("keydown", (e) => {
        if (e.key === "Enter") tr.click();
      });
    });
  }

  refreshIcons(el);
}

function kpi(label, value) {
  return `<div class="kpi"><div class="label">${escapeHtml(label)}</div><div class="value">${escapeHtml(String(value))}</div></div>`;
}

function renderRow(b) {
  const total = b.total_items || 0;
  const processed = b.processed_items || 0;
  const pct = total ? Math.round((processed / total) * 100) : 0;
  return `<tr data-id="${escapeHtml(b.id)}">
    <td>${escapeHtml(b.batch_name || "—")}</td>
    <td class="muted">${escapeHtml(formatDate(b.created_at))}</td>
    <td class="numeric">${processed}/${total}</td>
    <td><div class="progress" aria-label="Progress ${pct}%"><span style="width:${pct}%"></span></div></td>
    <td><span class="chip">${escapeHtml(b.status || "pending")}</span></td>
  </tr>`;
}
