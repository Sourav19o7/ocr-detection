// Search results page - displays a list of matching items with their status.

import { api } from "../api.js";
import { toast } from "./toast.js";
import {
  html, raw, escapeHtml, refreshIcons, icon,
  decisionChip, statusChip, formatDate,
} from "../lib/format.js";

export async function mountSearchResults(el, { query } = {}) {
  if (!query) {
    el.innerHTML = `<div class="empty-state">
      <div class="icon-wrap"><i data-lucide="search"></i></div>
      <h2>No search query</h2>
      <p>Enter a tag ID or HUID in the search box above.</p>
    </div>`;
    refreshIcons(el);
    return;
  }

  el.innerHTML = `<div class="search-results-loading">
    <div class="spinner"></div>
    <p>Searching for "${escapeHtml(query)}"...</p>
  </div>`;

  try {
    const resp = await api.get(`/stage3/search/tags?q=${encodeURIComponent(query)}&limit=50`);
    const results = resp.results || [];

    if (results.length === 0) {
      el.innerHTML = `<div class="empty-state">
        <div class="icon-wrap"><i data-lucide="search-x"></i></div>
        <h2>No results found</h2>
        <p>No items match "${escapeHtml(query)}"</p>
        <p><a href="#/batches" class="btn btn-secondary">Back to batches</a></p>
      </div>`;
      refreshIcons(el);
      return;
    }

    // If exactly one result, navigate directly to it
    if (results.length === 1) {
      location.hash = `#/item/${encodeURIComponent(results[0].tag_id)}`;
      return;
    }

    render(el, query, results);
  } catch (err) {
    console.error("Search error:", err);
    el.innerHTML = `<div class="empty-state">
      <div class="icon-wrap"><i data-lucide="alert-circle"></i></div>
      <h2>Search failed</h2>
      <p>${escapeHtml(err.message || "Unknown error")}</p>
    </div>`;
    refreshIcons(el);
  }
}

function render(el, query, results) {
  el.innerHTML = html`
    <div class="search-results-page">
      <div class="search-results-header">
        <h1>Search Results</h1>
        <p class="muted">${results.length} items matching "${escapeHtml(query)}"</p>
      </div>

      <div class="search-results-list">
        ${raw(results.map(item => renderResultCard(item, query)).join(""))}
      </div>
    </div>
  `;

  refreshIcons(el);

  // Add click handlers for result cards
  el.querySelectorAll(".result-card").forEach(card => {
    card.addEventListener("click", () => {
      const tagId = card.dataset.tagId;
      location.hash = `#/item/${encodeURIComponent(tagId)}`;
    });
    card.addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        card.click();
      }
    });
  });
}

function renderResultCard(item, query) {
  const tagId = item.tag_id || "";
  const huid = item.expected_huid || "";
  const decision = item.decision || "pending";
  const huidMatch = item.huid_match;
  const processedAt = item.processed_at;

  // Highlight matching text
  const highlightedTag = highlightMatch(tagId, query);
  const highlightedHuid = huid ? highlightMatch(huid, query) : "";

  // Determine status styling
  const statusClass = getStatusClass(decision, huidMatch);

  return `
    <div class="result-card ${statusClass}" data-tag-id="${escapeHtml(tagId)}" role="button" tabindex="0">
      <div class="result-main">
        <div class="result-tag">${highlightedTag}</div>
        ${huid ? `<div class="result-huid">HUID: <span class="mono">${highlightedHuid}</span></div>` : ''}
      </div>
      <div class="result-status">
        ${decisionChip(decision)}
        ${huidMatch === false ? '<span class="huid-mismatch-chip"><i data-lucide="alert-triangle"></i> HUID Mismatch</span>' : ''}
      </div>
      <div class="result-meta">
        ${processedAt ? `<span class="result-date">${escapeHtml(formatDate(processedAt))}</span>` : '<span class="result-date muted">Not processed</span>'}
        <i data-lucide="chevron-right"></i>
      </div>
    </div>
  `;
}

function highlightMatch(text, query) {
  if (!query) return escapeHtml(text);
  const idx = text.toLowerCase().indexOf(query.toLowerCase());
  if (idx === -1) return escapeHtml(text);
  const before = text.slice(0, idx);
  const match = text.slice(idx, idx + query.length);
  const after = text.slice(idx + query.length);
  return `${escapeHtml(before)}<mark>${escapeHtml(match)}</mark>${escapeHtml(after)}`;
}

function getStatusClass(decision, huidMatch) {
  if (huidMatch === false) return "status-mismatch";
  switch (decision) {
    case "approved": return "status-approved";
    case "rejected": return "status-rejected";
    case "manual_review": return "status-review";
    default: return "status-pending";
  }
}
