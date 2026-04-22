// Two-step upload wizard.
//   1. Drop an Excel/CSV, review accepted/rejected, create the batch.
//   2. Drop HUID images, auto-match by filename (<tag_id>.ext) or pick one.

import { api } from "../api.js";
import { toast } from "./toast.js";
import { openImageSourcePicker } from "./camera.js";
import {
  html, raw, escapeHtml, refreshIcons, icon,
} from "../lib/format.js";

export function mountUpload(el) {
  const state = { batch: null, tagIds: [], strict: false };

  el.innerHTML = html`
    <div class="page-header">
      <h1>Upload</h1>
      <div class="row" id="mode-switch" role="tablist" aria-label="Upload mode">
        <button class="btn btn-secondary btn-sm" role="tab" aria-selected="true" data-mode="new">New batch</button>
        <button class="btn btn-tertiary btn-sm" role="tab" aria-selected="false" data-mode="existing">Add to existing batch</button>
      </div>
    </div>

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

    <div class="upload-step card card-body hidden" id="step-existing">
      <div class="step-head"><span class="step-num">1</span><h2>Pick a batch</h2></div>
      <p class="muted" style="margin:0 0 12px">
        Choose a batch that's already been created. You can drop HUID or artifact images for any tag in it.
      </p>
      <select id="batch-picker" class="tag-input" style="max-width: 360px; font-family: Inter, sans-serif; font-size: 13px;">
        <option value="">Loading…</option>
      </select>
    </div>

    <div class="upload-step card card-body hidden" id="step2">
      <div class="step-head"><span class="step-num">2</span><h2>Images</h2></div>
      <p class="muted" style="margin:0 0 12px">
        <strong>HUID</strong> images (default) run OCR. Switch to <strong>Artifact</strong> to store reference shots without OCR.
        Files named <span class="mono">{tag_id}.jpg|png</span> auto-match.
      </p>
      <div class="row" style="margin-bottom:12px; gap: 16px;">
        <label class="row" style="gap:6px;"><input type="radio" name="img-type" value="huid" checked /> HUID</label>
        <label class="row" style="gap:6px;"><input type="radio" name="img-type" value="artifact" /> Artifact</label>
        <label class="row" style="gap:6px; margin-left:12px;" id="artifact-slot-wrap" hidden>
          Slot <select id="artifact-slot" class="tag-input" style="width:72px; font-family:Inter, sans-serif; font-size:12px;">
            <option value="1">1</option><option value="2">2</option><option value="3">3</option>
          </select>
        </label>
      </div>
      <div class="dropzone" id="drop-images">
        ${raw(icon("images").value)}
        <h3>Drop images here</h3>
        <p>PNG or JPG. One file per tag.</p>
      </div>
      <div class="upload-alt-row">
        <span class="muted">or</span>
        <button class="btn btn-secondary btn-sm" id="capture-image">
          ${raw(icon("camera").value)} Capture from camera
        </button>
      </div>
      <div class="file-rows" id="image-rows"></div>
    </div>
  `;

  refreshIcons(el);

  // --- Mode switch --------------------------------------------------------

  const step1 = el.querySelector("#step1");
  const stepExisting = el.querySelector("#step-existing");
  const step2 = el.querySelector("#step2");
  const modeButtons = el.querySelectorAll("#mode-switch [data-mode]");

  function setMode(mode) {
    modeButtons.forEach(b => {
      const active = b.dataset.mode === mode;
      b.setAttribute("aria-selected", active ? "true" : "false");
      b.classList.toggle("btn-secondary", active);
      b.classList.toggle("btn-tertiary", !active);
    });
    if (mode === "new") {
      step1.classList.remove("hidden");
      stepExisting.classList.add("hidden");
      step2.classList.add("hidden");
      state.batch = null; state.tagIds = [];
    } else {
      step1.classList.add("hidden");
      stepExisting.classList.remove("hidden");
      step2.classList.add("hidden");
      loadExistingBatches(el, state);
    }
  }
  modeButtons.forEach(b => b.addEventListener("click", () => setMode(b.dataset.mode)));

  // --- Step 1 (new batch) -------------------------------------------------

  const dropSheet = el.querySelector("#drop-sheet");
  const strictEl = el.querySelector("#strict-toggle");
  strictEl.addEventListener("change", () => { state.strict = strictEl.checked; });

  wireDropzone(dropSheet, async (files) => {
    if (!files.length) return;
    await handleSheet(files[0], state, el);
  }, { multiple: false });

  dropSheet.addEventListener("click", () => promptFile(".csv,.xlsx,.xls", false, async (files) => {
    if (files.length) await handleSheet(files[0], state, el);
  }));

  // --- Step 2 (images) ----------------------------------------------------

  const typeRadios = el.querySelectorAll('input[name="img-type"]');
  const artifactSlotWrap = el.querySelector("#artifact-slot-wrap");
  typeRadios.forEach(r => r.addEventListener("change", () => {
    const mode = el.querySelector('input[name="img-type"]:checked').value;
    artifactSlotWrap.hidden = mode !== "artifact";
  }));

  const dropImg = el.querySelector("#drop-images");
  wireDropzone(dropImg, (files) => handleImages([...files], state, el), { multiple: true });
  dropImg.addEventListener("click", () => promptFile("image/*", true, (files) => handleImages([...files], state, el)));

  // Camera capture button
  const captureBtn = el.querySelector("#capture-image");
  captureBtn.addEventListener("click", () => {
    if (!state.batch) {
      toast.error("Pick a batch (or upload a sheet) first");
      return;
    }
    openImageSourcePicker({
      accept: "image/*",
      multiple: true,
      onFiles: (files) => handleImages([...files], state, el),
      captureFilename: "captured_image.jpg"
    });
  });
}

async function loadExistingBatches(root, state) {
  const picker = root.querySelector("#batch-picker");
  try {
    const resp = await api.get("/stage1/batches");
    const batches = (resp && (resp.batches || resp)) || [];
    if (!batches.length) {
      picker.innerHTML = `<option value="">(no batches yet — create one first)</option>`;
      return;
    }
    picker.innerHTML = [`<option value="">Select a batch…</option>`]
      .concat(batches.map(b => `<option value="${b.id}">#${b.id} · ${escapeHtml(b.batch_name)} (${b.total_items || 0} items)</option>`))
      .join("");
    picker.addEventListener("change", async () => {
      const batchId = picker.value;
      if (!batchId) return;
      const detail = await api.get(`/stage3/batch/${batchId}/results`);
      state.batch = { batch_id: Number(batchId), batch_name: detail.batch_name };
      state.tagIds = (detail.results || []).map(r => r.tag_id);
      root.querySelector("#step2").classList.remove("hidden");
      root.querySelector("#step2").scrollIntoView({ behavior: "smooth", block: "start" });
      toast.info(`Batch ${detail.batch_name}: ${state.tagIds.length} tags loaded`);
    }, { once: false });
  } catch (e) {
    picker.innerHTML = `<option value="">Error loading batches</option>`;
    toast.error(e.message || "Could not load batches");
  }
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
    toast.error("Pick a batch (or upload a sheet) first");
    return;
  }
  const rowsEl = root.querySelector("#image-rows");
  ensureTagDatalist(root, state.tagIds);
  const tagSet = new Set(state.tagIds);
  const imageType = root.querySelector('input[name="img-type"]:checked')?.value || "huid";
  const artifactSlot = Number(root.querySelector("#artifact-slot")?.value || 1);

  files.forEach(file => {
    const row = document.createElement("div");
    row.className = "file-row";
    const guessed = guessTagId(file.name, state.tagIds);
    row.innerHTML = `
      <div class="truncate"><strong class="mono">${escapeHtml(file.name)}</strong></div>
      <div class="progress"><span style="width:0%"></span></div>
      <input class="tag-input" list="tag-options"
             placeholder="Type or pick tag…"
             value="${escapeHtml(guessed)}"
             autocomplete="off" spellcheck="false" />
      <span></span>
    `;
    rowsEl.appendChild(row);

    const bar = row.querySelector(".progress > span");
    const icon = row.querySelector(":scope > span:last-child");
    const input = row.querySelector("input.tag-input");

    const sendWithTag = (tag) => {
      const xhr = new XMLHttpRequest();
      const endpoint = imageType === "huid" ? "/stage2/upload-image" : "/stage2/upload-artifact";
      xhr.open("POST", endpoint);
      const fd = new FormData();
      fd.append("file", file);
      fd.append("tag_id", tag);
      if (imageType === "huid") {
        fd.append("image_type", "huid");
        fd.append("slot", "0");
      } else {
        fd.append("slot", String(artifactSlot));
      }
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
      xhr.send(fd);
    };

    const validate = () => {
      const v = input.value.trim();
      if (!v) { input.classList.remove("invalid"); return false; }
      if (!tagSet.has(v)) { input.classList.add("invalid"); return false; }
      input.classList.remove("invalid");
      return true;
    };

    input.addEventListener("input", validate);

    // If filename auto-matched, fire immediately. Otherwise wait for a valid
    // pick (via datalist, typing, or paste) and upload on change/enter.
    if (guessed) {
      sendWithTag(guessed);
    } else {
      const trigger = () => {
        if (!validate()) return;
        input.removeEventListener("change", trigger);
        input.removeEventListener("keydown", keyTrigger);
        sendWithTag(input.value.trim());
      };
      const keyTrigger = (e) => { if (e.key === "Enter") trigger(); };
      input.addEventListener("change", trigger);
      input.addEventListener("keydown", keyTrigger);
      input.focus();
    }
  });
}

function ensureTagDatalist(root, tagIds) {
  let list = document.getElementById("tag-options");
  if (!list) {
    list = document.createElement("datalist");
    list.id = "tag-options";
    root.appendChild(list);
  }
  list.innerHTML = tagIds.map(t => `<option value="${escapeHtml(t)}"></option>`).join("");
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
