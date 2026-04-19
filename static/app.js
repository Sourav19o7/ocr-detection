// Router + bootstrap. Hash-based so we don't need history-mode rewrites.

import { mountTopbar } from "./components/topbar.js";
import { mountBatches } from "./components/batches-table.js";
import { mountBatchDetail } from "./components/batch-detail.js";
import { mountItemPreview } from "./components/item-preview.js";
import { mountUpload } from "./components/upload-flow.js";
import { toast } from "./components/toast.js";
import { drawer } from "./components/drawer.js";

const topbarEl = document.getElementById("topbar");
const appEl = document.getElementById("app");

const routes = [
  { pattern: /^#\/batches$/,               view: () => render({ crumbs: [{label:"Batches"}] }, mountBatches) },
  { pattern: /^#\/batch\/([^/]+)$/,        view: (m) => render({ crumbs: [{label:"Batches", href:"#/batches"}, {label:`Batch ${decodeURIComponent(m[1])}`}] }, (el)=>mountBatchDetail(el, { batchId: decodeURIComponent(m[1]) })) },
  { pattern: /^#\/item\/([^/]+)$/,         view: (m) => render({ crumbs: [{label:"Batches", href:"#/batches"}, {label:decodeURIComponent(m[1])}] }, (el)=>mountItemPreview(el, { tagId: decodeURIComponent(m[1]) })) },
  { pattern: /^#\/upload$/,                view: () => render({ crumbs: [{label:"Batches", href:"#/batches"}, {label:"Upload"}] }, mountUpload) },
];

function render(opts, mountFn) {
  mountTopbar(topbarEl, { breadcrumbs: opts.crumbs });
  appEl.innerHTML = "";
  mountFn(appEl);
}

function route() {
  if (!location.hash || location.hash === "#" || location.hash === "#/") {
    location.hash = "#/batches";
    return;
  }
  drawer.close();
  for (const r of routes) {
    const m = location.hash.match(r.pattern);
    if (m) { r.view(m); return; }
  }
  appEl.innerHTML = `<div class="empty-state"><h1>Page not found</h1><p class="muted">${location.hash}</p><p><a href="#/batches" class="btn btn-secondary">Back to batches</a></p></div>`;
}

window.addEventListener("hashchange", route);
window.addEventListener("load", route);

// Global keyboard shortcuts
window.addEventListener("keydown", (e) => {
  if (e.target && /^(INPUT|TEXTAREA|SELECT)$/.test(e.target.tagName)) return;
  if (e.key === "/") {
    const input = document.getElementById("topbar-search");
    if (input) { e.preventDefault(); input.focus(); input.select(); }
  }
});

// Surface uncaught fetch errors quietly.
window.addEventListener("unhandledrejection", (ev) => {
  const msg = ev.reason && (ev.reason.message || String(ev.reason));
  if (msg) toast.error(msg, { timeout: 5000 });
});
