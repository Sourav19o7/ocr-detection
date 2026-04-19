// Right-side drawer. One at a time. Esc closes; focus trap is minimal (first
// focusable + click-away).

let backdrop;
let current = null;

function ensureBackdrop() {
  if (backdrop) return backdrop;
  backdrop = document.createElement("div");
  backdrop.className = "drawer-backdrop";
  backdrop.addEventListener("click", () => drawer.close());
  document.body.appendChild(backdrop);
  return backdrop;
}

export const drawer = {
  open(renderFn) {
    const el = document.getElementById("drawer");
    el.innerHTML = "";
    const wrapper = document.createElement("div");
    renderFn(wrapper);
    el.appendChild(wrapper);
    el.classList.add("open");
    el.setAttribute("aria-hidden", "false");
    ensureBackdrop().classList.add("open");
    current = { el, wrapper };
    // Focus first focusable element inside
    const focusable = el.querySelector("button, [href], input, select, textarea, [tabindex]");
    if (focusable) focusable.focus();
  },

  close() {
    if (!current) return;
    current.el.classList.remove("open");
    current.el.setAttribute("aria-hidden", "true");
    if (backdrop) backdrop.classList.remove("open");
    current = null;
  },

  isOpen() { return !!current; },
};

window.addEventListener("keydown", (e) => {
  if (e.key === "Escape" && drawer.isOpen()) drawer.close();
});
