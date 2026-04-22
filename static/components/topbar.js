import { html, raw, refreshIcons, icon, escapeHtml } from "../lib/format.js";
import { api } from "../api.js";

let searchDebounceTimer = null;

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
      <div class="search-wrap">
        <label class="search" aria-label="Search">
          ${icon("search")}
          <input id="topbar-search" placeholder="Search by tag ID or HUID…" autocomplete="off" />
          <kbd>/</kbd>
        </label>
        <div id="search-suggestions" class="search-suggestions hidden"></div>
      </div>
      <a class="btn btn-secondary" href="#/downloads">${icon("download")}Downloads</a>
      <a class="btn btn-primary" href="#/upload">${icon("upload")}Upload batch</a>
    </div>
  `;

  const input = el.querySelector("#topbar-search");
  const suggestions = el.querySelector("#search-suggestions");
  let selectedIndex = -1;

  // Handle input for live search
  input.addEventListener("input", () => {
    const q = input.value.trim();
    if (searchDebounceTimer) clearTimeout(searchDebounceTimer);

    if (q.length < 1) {
      hideSuggestions();
      return;
    }

    searchDebounceTimer = setTimeout(() => searchTags(q), 200);
  });

  async function searchTags(query) {
    try {
      const resp = await api.get(`/stage3/search/tags?q=${encodeURIComponent(query)}&limit=8`);
      const results = resp.results || [];
      const tags = resp.tags || [];

      if (results.length === 0) {
        suggestions.innerHTML = `<div class="suggestion-empty">No matching tags found</div>`;
        suggestions.classList.remove("hidden");
        selectedIndex = -1;
        return;
      }

      // Check if exact match exists
      const exactMatch = results.find(r => r.tag_id.toLowerCase() === query.toLowerCase());

      suggestions.innerHTML = results.map((result, i) => {
        const tag = result.tag_id;
        const huid = result.expected_huid || "";
        const isExactTag = tag.toLowerCase() === query.toLowerCase();
        const isExactHuid = huid.toLowerCase() === query.toLowerCase();
        const tagMatchesQuery = tag.toLowerCase().includes(query.toLowerCase());
        const huidMatchesQuery = huid.toLowerCase().includes(query.toLowerCase());

        // Highlight the matching field
        const highlightedTag = tagMatchesQuery ? highlightMatch(tag, query) : escapeHtml(tag);
        const highlightedHuid = huidMatchesQuery ? highlightMatch(huid, query) : escapeHtml(huid);

        return `<div class="suggestion-item${isExactTag || isExactHuid ? ' exact' : ''}" data-index="${i}" data-tag="${escapeHtml(tag)}">
          <div class="suggestion-tag">${highlightedTag}</div>
          ${huid ? `<div class="suggestion-huid">HUID: <span class="mono">${highlightedHuid}</span></div>` : ''}
        </div>`;
      }).join("");

      suggestions.classList.remove("hidden");
      selectedIndex = -1;

      // Add click handlers
      suggestions.querySelectorAll(".suggestion-item").forEach(item => {
        item.addEventListener("click", () => {
          const tag = item.dataset.tag;
          input.value = tag;
          hideSuggestions();
          location.hash = `#/item/${encodeURIComponent(tag)}`;
        });
      });

    } catch (err) {
      console.error("Search error:", err);
      hideSuggestions();
    }
  }

  function highlightMatch(tag, query) {
    const idx = tag.toLowerCase().indexOf(query.toLowerCase());
    if (idx === -1) return escapeHtml(tag);
    const before = tag.slice(0, idx);
    const match = tag.slice(idx, idx + query.length);
    const after = tag.slice(idx + query.length);
    return `${escapeHtml(before)}<mark>${escapeHtml(match)}</mark>${escapeHtml(after)}`;
  }

  function hideSuggestions() {
    suggestions.classList.add("hidden");
    suggestions.innerHTML = "";
    selectedIndex = -1;
  }

  function updateSelection() {
    const items = suggestions.querySelectorAll(".suggestion-item");
    items.forEach((item, i) => {
      item.classList.toggle("selected", i === selectedIndex);
    });
    if (selectedIndex >= 0 && items[selectedIndex]) {
      items[selectedIndex].scrollIntoView({ block: "nearest" });
    }
  }

  input.addEventListener("keydown", async (e) => {
    const items = suggestions.querySelectorAll(".suggestion-item");
    const isOpen = !suggestions.classList.contains("hidden") && items.length > 0;

    if (e.key === "ArrowDown" && isOpen) {
      e.preventDefault();
      selectedIndex = Math.min(selectedIndex + 1, items.length - 1);
      updateSelection();
      return;
    }

    if (e.key === "ArrowUp" && isOpen) {
      e.preventDefault();
      selectedIndex = Math.max(selectedIndex - 1, 0);
      updateSelection();
      return;
    }

    if (e.key === "Enter") {
      e.preventDefault();
      const v = input.value.trim();

      // If a suggestion is selected, go directly to that item
      if (isOpen && selectedIndex >= 0 && items[selectedIndex]) {
        const tag = items[selectedIndex].dataset.tag;
        input.value = tag;
        hideSuggestions();
        location.hash = `#/item/${encodeURIComponent(tag)}`;
        return;
      }

      // Otherwise, navigate to search results page to show all matches
      if (v) {
        hideSuggestions();
        location.hash = `#/search?q=${encodeURIComponent(v)}`;
      }
      return;
    }

    if (e.key === "Escape") {
      hideSuggestions();
      input.blur();
    }
  });

  // Hide suggestions when clicking outside
  document.addEventListener("click", (e) => {
    if (!el.contains(e.target)) {
      hideSuggestions();
    }
  });

  // Focus input on "/" key
  document.addEventListener("keydown", (e) => {
    if (e.key === "/" && document.activeElement !== input && !e.ctrlKey && !e.metaKey) {
      const tag = document.activeElement?.tagName?.toLowerCase();
      if (tag !== "input" && tag !== "textarea" && !document.activeElement?.isContentEditable) {
        e.preventDefault();
        input.focus();
      }
    }
  });

  refreshIcons(el);
}
