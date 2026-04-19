// Presentation helpers — keep components terse.

export function formatDate(value) {
  if (!value) return "—";
  const d = typeof value === "string" ? new Date(value) : value;
  if (isNaN(d)) return String(value);
  return d.toLocaleString(undefined, {
    year: "numeric", month: "short", day: "2-digit",
    hour: "2-digit", minute: "2-digit",
  });
}

export function formatPercent(value, digits = 1) {
  if (value == null) return "—";
  return `${(value * 100).toFixed(digits)}%`;
}

export function formatConfidence(value) {
  if (value == null) return "—";
  return value.toFixed(3);
}

export function decisionChip(decision) {
  const map = {
    approved:       { cls: "chip chip-success", label: "Approved" },
    auto_approved:  { cls: "chip chip-success", label: "Auto-approved" },
    manual_review:  { cls: "chip chip-warning", label: "Manual review" },
    flagged:        { cls: "chip chip-warning", label: "Flagged" },
    rejected:       { cls: "chip chip-danger",  label: "Rejected" },
    pending:        { cls: "chip",              label: "Pending" },
  };
  const d = map[(decision || "pending").toLowerCase()] || { cls: "chip", label: decision || "—" };
  return `<span class="${d.cls}">${d.label}</span>`;
}

export function statusChip(status) {
  const map = {
    pending:       { cls: "chip",             label: "Pending" },
    processing:    { cls: "chip chip-info",   label: "Processing" },
    completed:     { cls: "chip chip-success",label: "Completed" },
    failed:        { cls: "chip chip-danger", label: "Failed" },
    manual_review: { cls: "chip chip-warning",label: "Manual review" },
  };
  const s = map[(status || "pending").toLowerCase()] || { cls: "chip", label: status || "—" };
  return `<span class="${s.cls}">${s.label}</span>`;
}

export function escapeHtml(str) {
  if (str == null) return "";
  return String(str)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

// Tiny template tag to keep components readable; safely escapes ${...}
// substitutions. Pass raw HTML with the `raw(...)` helper.
export function html(strings, ...values) {
  let out = "";
  strings.forEach((s, i) => {
    out += s;
    if (i < values.length) {
      const v = values[i];
      if (v && typeof v === "object" && v.__raw) out += v.value;
      else if (Array.isArray(v)) out += v.map(part => (part && part.__raw) ? part.value : escapeHtml(part)).join("");
      else out += escapeHtml(v);
    }
  });
  return out;
}

export function raw(value) { return { __raw: true, value: String(value ?? "") }; }

export function icon(name, extraClass = "") {
  // Lucide replaces <i data-lucide="..."> via lucide.createIcons() on the next tick.
  return raw(`<i data-lucide="${escapeHtml(name)}" class="lucide ${escapeHtml(extraClass)}"></i>`);
}

export function refreshIcons(root = document) {
  if (window.lucide && typeof window.lucide.createIcons === "function") {
    window.lucide.createIcons({ attrs: { class: "lucide" }, nameAttr: "data-lucide" });
  }
}
