// Two-step upload wizard.
//   1. Drop an Excel/CSV, review accepted/rejected, create the batch.
//   2. Drop HUID images, auto-match by filename (<tag_id>.ext) or pick one.

import { api } from "../api.js";
import { toast } from "./toast.js";
import {
  html, raw, escapeHtml, refreshIcons, icon,
} from "../lib/format.js";

export function mountUpload(el) {
  const state = { batch: null, tagIds: [], strict: false };

  el.innerHTML = html`
    <div class="page-header"><h1>Upload batch</h1></div>

    <div class="upload-step card card-body" id="step1">
      <div class="step-head"><span class="step-num">1</span><h2>Excel / CSV</h2></div>
      <p class="muted" style="margin:0 0 12px">
        The file needs a <span class="mono">tag_id</span> column and an
        <span class="mono">expected_huid</span> column. Extra columns are ignored.
      </p>
      <label class="row" style="margin-bottom:12px; gap: 8px;">
        <input type="checkbox" id="strict-toggle" />
        <span>Strict mode — reject the whole batch if any row fails validation.</span>
      </label>
      <div class="dropzone" id="drop-sheet">
        ${raw(icon("upload-cloud").value)}
        <h3>Drop your spreadsheet here</h3>
        <p>…or click to browse. CSV, XLSX, XLS.</p>
      </div>
      <div id="sheet-result" class="hidden" style="margin-top:16px"></div>
    </div>

    <div class="upload-step card card-body hidden" id="step2">
      <div class="step-head"><span class="step-num">2</span><h2>HUID images</h2></div>
      <p class="muted" style="margin:0 0 12px">
        Files named <span class="mono">{tag_id}.jpg|png</span> auto-match. Anything else lets you pick the tag from the dropdown.
      </p>
      <div class="dropzone" id="drop-images">
        ${raw(icon("images").value)}
        <h3>Drop images here</h3>
        <p>PNG or JPG. OCR runs automatically.</p>
      </div>
      <div class="file-rows" id="image-rows"></div>
    </div>
  `;

  refreshIcons(el);

  // --- Step 1 wiring -------------------------------------------------------

  const dropSheet = el.querySelector("#drop-sheet");
  const strictEl = el.querySelector("#strict-toggle");
  strictEl.addEventListener("change", () => { state.strict = strictEl.checked; });

  wireDropzone(dropSheet, async (files) => {
    if (!files.length) return;
    await handleSheet(files[0], state, el);
  }, { multiple: false });

  dropSheet.addEventListener("click", () => promptFile("*.csv,.xlsx,.xls", false, async (files) => {
    if (files.length) await handleSheet(files[0], state, el);
  }));

  // --- Step 2 wiring -------------------------------------------------------

  const dropImg = el.querySelector("#drop-images");
  wireDropzone(dropImg, (files) => handleImages([...files], state, el), { multiple: true });
  dropImg.addEventListener("click", () => promptFile("image/*", true, (files) => handleImages([...files], state, el)));
}

// --- step 1 --------------------------------------------------------------

async function handleSheet(file, state, root) {
  const fd = new FormData();
  fd.append("file", file);
  const qs = state.strict ? "?strict=true" : "";
  const result = root.querySelector("#sheet-result");
  result.classList.remove("hidden");
  result.innerHTML = `<p class="muted">Uploading ${escapeHtml(file.name)}…</p>`;

  try {
    const resp = await api.postForm(`/stage1/upload-batch${qs}`, fd);
    state.batch = resp;
    state.tagIds = (resp.rejected ? [] : []); // populated below after batch items fetch
    const rejectedRows = (resp.rejected || []).map(r => `
      <tr>
        <td class="numeric">${r.row}</td>
        <td class="mono">${escapeHtml(r.tag_id)}</td>
        <td>${escapeHtml(r.reason)}</td>
      </tr>`).join("");

    result.innerHTML = `
      <div class="row" style="gap: 12px; margin-bottom: 12px;">
        <span class="chip chip-success">${resp.accepted} accepted</span>
        ${resp.rejected?.length ? `<span class="chip chip-danger">${resp.rejected.length} rejected</span>` : ""}
        <span class="chip">${resp.total_rows} total rows</span>
      </div>
      <div><strong>Batch #${resp.batch_id}</strong> — ${escapeHtml(resp.batch_name || "")}</div>
      ${rejectedRows ? `
        <details open style="margin-top:12px">
          <summary>Rejected rows</summary>
          <div class="table-wrap" style="margin-top:8px">
            <table class="table"><thead><tr><th>Row</th><th>Tag ID</th><th>Reason</th></tr></thead>
              <tbody>${rejectedRows}</tbody>
            </table>
          </div>
        </details>` : ""}
      <div class="row" style="margin-top: 16px; gap: 8px">
        <a class="btn btn-secondary" href="#/batch/${resp.batch_id}">View batch</a>
        <button class="btn btn-primary" id="go-images">Continue to images</button>
      </div>
    `;

    // Pre-fetch tag IDs so we can auto-match / dropdown them in step 2.
    try {
      const items = await api.get(`/stage3/batch/${resp.batch_id}/results`);
      state.tagIds = (items.results || []).map(r => r.tag_id);
    } catch { /* non-fatal */ }

    result.querySelector("#go-images").addEventListener("click", () => {
      root.querySelector("#step2").classList.remove("hidden");
      root.querySelector("#step2").scrollIntoView({ behavior: "smooth", block: "start" });
    });

    toast.success(`Batch #${resp.batch_id} created (${resp.accepted} rows)`);
  } catch (e) {
    const details = e.body?.rejected;
    result.innerHTML = `<div class="chip chip-danger">${escapeHtml(e.message || "Upload failed")}</div>`;
    if (details?.length) {
      const rows = details.map(r => `<tr><td>${r.row}</td><td class="mono">${escapeHtml(r.tag_id)}</td><td>${escapeHtml(r.reason)}</td></tr>`).join("");
      result.innerHTML += `<div class="table-wrap" style="margin-top:12px"><table class="table"><thead><tr><th>Row</th><th>Tag</th><th>Reason</th></tr></thead><tbody>${rows}</tbody></table></div>`;
    }
    toast.error(e.message || "Upload failed");
  }
}

// --- step 2 --------------------------------------------------------------

function handleImages(files, state, root) {
  if (!state.batch) {
    toast.error("Upload a batch sheet first");
    return;
  }
  const rowsEl = root.querySelector("#image-rows");

  files.forEach(file => {
    const row = document.createElement("div");
    row.className = "file-row";
    const guessed = guessTagId(file.name, state.tagIds);
    row.innerHTML = `
      <div class="truncate"><strong class="mono">${escapeHtml(file.name)}</strong></div>
      <div class="progress"><span style="width:0%"></span></div>
      <select>${optionList(state.tagIds, guessed)}</select>
      <span></span>
    `;
    rowsEl.appendChild(row);

    const bar = row.querySelector(".progress > span");
    const icon = row.querySelector(":scope > span:last-child");
    const select = row.querySelector("select");

    const xhr = new XMLHttpRequest();
    xhr.open("POST", "/stage2/upload-image");
    const fd = new FormData();
    fd.append("file", file);
    fd.append("tag_id", select.value || "");
    fd.append("image_type", "huid");
    fd.append("slot", "0");

    xhr.upload.addEventListener("progress", (e) => {
      if (!e.lengthComputable) return;
      bar.style.width = `${Math.round((e.loaded / e.total) * 100)}%`;
    });
    xhr.addEventListener("load", () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        row.classList.add("ok");
        bar.style.width = "100%";
        icon.innerHTML = '<i data-lucide="check" style="color: var(--success)"></i>';
      } else {
        row.classList.add("err");
        icon.innerHTML = '<i data-lucide="alert-circle" style="color: var(--danger)"></i>';
        let msg = xhr.statusText;
        try { msg = JSON.parse(xhr.responseText)?.detail || msg; } catch {}
        row.querySelector(".truncate").insertAdjacentHTML("beforeend", `<div class="muted" style="font-size:12px">${escapeHtml(msg)}</div>`);
      }
      refreshIcons(row);
    });
    xhr.addEventListener("error", () => {
      row.classList.add("err");
      icon.innerHTML = '<i data-lucide="alert-circle" style="color: var(--danger)"></i>';
      refreshIcons(row);
    });

    // Delay start until the dropdown has a chosen tag
    if (!select.value) {
      select.addEventListener("change", () => {
        if (!select.value) return;
        const fd2 = new FormData();
        fd2.append("file", file);
        fd2.append("tag_id", select.value);
        fd2.append("image_type", "huid");
        fd2.append("slot", "0");
        xhr.open("POST", "/stage2/upload-image");
        xhr.send(fd2);
      }, { once: true });
    } else {
      xhr.send(fd);
    }
  });
}

function guessTagId(filename, tagIds) {
  const base = filename.replace(/\.[^.]+$/, "");
  return tagIds.includes(base) ? base : "";
}

function optionList(tagIds, selected) {
  const opts = [`<option value="">Pick tag…</option>`]
    .concat(tagIds.map(t => `<option value="${escapeHtml(t)}" ${t === selected ? "selected" : ""}>${escapeHtml(t)}</option>`));
  return opts.join("");
}

// --- shared ---------------------------------------------------------------

function wireDropzone(el, onFiles, { multiple }) {
  el.addEventListener("dragover", (e) => { e.preventDefault(); el.classList.add("dragover"); });
  el.addEventListener("dragleave", () => el.classList.remove("dragover"));
  el.addEventListener("drop", (e) => {
    e.preventDefault(); el.classList.remove("dragover");
    onFiles(multiple ? e.dataTransfer.files : [e.dataTransfer.files[0]].filter(Boolean));
  });
}

function promptFile(accept, multiple, onFiles) {
  const input = document.createElement("input");
  input.type = "file"; input.accept = accept; input.multiple = !!multiple;
  input.addEventListener("change", () => onFiles(input.files));
  input.click();
}
