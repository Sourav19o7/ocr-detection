// Thin fetch wrapper. Every helper throws {status, body} on non-2xx.

const BASE = (typeof window !== "undefined" && window.__API_BASE__) || "";

async function parse(resp) {
  const text = await resp.text();
  if (!text) return null;
  try { return JSON.parse(text); }
  catch { return text; }
}

async function request(method, path, { body, headers, signal } = {}) {
  const resp = await fetch(BASE + path, { method, headers, body, signal });
  const data = await parse(resp);
  if (!resp.ok) {
    const err = new Error((data && data.detail) || `${method} ${path} failed (${resp.status})`);
    err.status = resp.status;
    err.body = data;
    throw err;
  }
  return data;
}

export const api = {
  get:      (path, opts)        => request("GET", path, opts),
  post:     (path, payload, o)  => request("POST", path, {
                                     ...o,
                                     body: JSON.stringify(payload),
                                     headers: { "Content-Type": "application/json", ...(o?.headers || {}) },
                                   }),
  postForm: (path, formData, o) => request("POST", path, { ...o, body: formData }),
  del:      (path, opts)        => request("DELETE", path, opts),
};
