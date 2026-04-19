import { html, raw, refreshIcons, icon } from "../lib/format.js";

export function mountTopbar(el, { breadcrumbs = [] } = {}) {
  const crumbs = breadcrumbs
    .map((c, i) => (i === breadcrumbs.length - 1
      ? `<span>${c.label}</span>`
      : `<a href="${c.href}">${c.label}</a><span class="sep">›</span>`))
    .join("");

  el.innerHTML = html`
    <a href="#/batches" class="brand">
      <span class="dot"></span>Hallmark QC
    </a>
    <nav class="breadcrumbs">${raw(crumbs)}</nav>
    <div class="row" style="gap: 12px;">
      <label class="search" aria-label="Search">
        ${icon("search")}
        <input id="topbar-search" placeholder="Search by tag id…" />
        <kbd>/</kbd>
      </label>
      <a class="btn btn-primary" href="#/upload">${icon("upload")}Upload batch</a>
    </div>
  `;

  const input = el.querySelector("#topbar-search");
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      const v = input.value.trim();
      if (v) location.hash = `#/item/${encodeURIComponent(v)}`;
    }
    if (e.key === "Escape") input.blur();
  });

  refreshIcons(el);
}
