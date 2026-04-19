// Minimal toast. Call `toast.push({type, message})`.

const host = () => document.getElementById("toasts");

export const toast = {
  push({ type = "info", message = "", timeout = 3000 } = {}) {
    const el = document.createElement("div");
    el.className = `toast ${type}`;
    el.textContent = message;
    host().appendChild(el);
    const remove = () => {
      el.style.opacity = "0";
      el.style.transition = "opacity 150ms";
      setTimeout(() => el.remove(), 150);
    };
    if (timeout > 0) setTimeout(remove, timeout);
    el.addEventListener("click", remove);
    return remove;
  },
  success: (message, opts) => toast.push({ ...opts, type: "success", message }),
  error:   (message, opts) => toast.push({ ...opts, type: "error", message }),
  info:    (message, opts) => toast.push({ ...opts, type: "info", message }),
};
