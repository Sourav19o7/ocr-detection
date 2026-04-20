// Centered modal primitive. One at a time. Esc closes; backdrop click closes.

let current = null;

export const modal = {
  open({ title, body, footer, onClose, width } = {}) {
    modal.close();

    const backdrop = document.createElement("div");
    backdrop.className = "modal-backdrop";
    backdrop.addEventListener("click", (e) => { if (e.target === backdrop) modal.close(); });

    const el = document.createElement("div");
    el.className = "modal";
    if (width) el.style.width = typeof width === "number" ? `${width}px` : width;
    el.innerHTML = `
      <div class="modal-header">
        <strong>${title || ""}</strong>
        <button class="icon-btn" aria-label="Close"><i data-lucide="x"></i></button>
      </div>
      <div class="modal-body"></div>
      ${footer ? '<div class="modal-footer"></div>' : ""}
    `;
    const bodyEl = el.querySelector(".modal-body");
    if (typeof body === "function") body(bodyEl);
    else if (body instanceof Node) bodyEl.appendChild(body);
    else if (body) bodyEl.innerHTML = body;
    if (footer) {
      const f = el.querySelector(".modal-footer");
      if (typeof footer === "function") footer(f);
      else if (footer instanceof Node) f.appendChild(footer);
      else f.innerHTML = footer;
    }
    el.querySelector(".icon-btn").addEventListener("click", () => modal.close());

    backdrop.appendChild(el);
    document.body.appendChild(backdrop);
    if (window.lucide) window.lucide.createIcons({ attrs: { class: "lucide" }, nameAttr: "data-lucide" });

    current = { backdrop, onClose };
    const focusable = el.querySelector("button, [href], input, select, textarea, [tabindex]");
    if (focusable) focusable.focus();
    return el;
  },

  close() {
    if (!current) return;
    current.backdrop.remove();
    if (typeof current.onClose === "function") { try { current.onClose(); } catch {} }
    current = null;
  },
};

window.addEventListener("keydown", (e) => { if (e.key === "Escape" && current) modal.close(); });
