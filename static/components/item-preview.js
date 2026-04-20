// Preview screen: image gallery (HUID + up to 3 artifacts) and metadata panel.
// Works both full-page and inside the drawer (compact=true hides the header).

import { api } from "../api.js";
import { toast } from "./toast.js";
import {
  html, raw, escapeHtml, refreshIcons, icon,
  decisionChip, statusChip, formatConfidence, formatDate,
  composedActualHuid,
} from "../lib/format.js";

export async function mountItemPreview(el, { tagId, compact = false } = {}) {
  el.innerHTML = loadingShell(tagId, compact);
  refreshIcons(el);

  let data;
  try {
    data = await api.get(`/stage3/item/${encodeURIComponent(tagId)}`);
  } catch (e) {
    el.innerHTML = `<div class="empty-state">
      <div class="icon-wrap"><i data-lucide="alert-circle"></i></div>
      <h2>Could not load ${escapeHtml(tagId)}</h2>
      <p>${escapeHtml((e.body && e.body.detail) || e.message || "")}</p>
    </div>`;
    refreshIcons(el);
    return;
  }

  render(el, data, { compact });
}

function loadingShell(tagId, compact) {
  return `<div class="preview">
    <div class="viewer"><div class="large empty"></div>
      <div class="gallery-strip">
        ${[0,1,2,3].map(()=>'<div class="tile"><span class="skeleton" style="width:40%"></span></div>').join("")}
      </div>
    </div>
    <div>
      ${[...Array(4)].map(()=>'<div class="card meta-card"><span class="skeleton" style="width:60%"></span></div>').join("")}
    </div>
  </div>`;
}

function render(el, data, { compact }) {
  const images = data.images || { huid: null, artifacts: [] };
  const huid = images.huid;
  const bySlot = { 1: null, 2: null, 3: null };
  (images.artifacts || []).forEach(a => { if (a.slot >= 1 && a.slot <= 3) bySlot[a.slot] = a; });

  el.innerHTML = html`
    <div class="preview">
      <div>
        <div class="viewer">
          <div id="large" class="large ${huid || images.artifacts?.length ? "" : "empty"}"
               style="${(huid && huid.url) ? `background-image:url('${escapeHtml(huid.url)}')` : ""}"></div>
          <div class="gallery-strip" id="strip">
            ${raw(tile({ type: "huid", slot: 0, image: huid, active: true, label: "HUID" }))}
            ${raw(tile({ type: "artifact", slot: 1, image: bySlot[1], label: "Artifact 1" }))}
            ${raw(tile({ type: "artifact", slot: 2, image: bySlot[2], label: "Artifact 2" }))}
            ${raw(tile({ type: "artifact", slot: 3, image: bySlot[3], label: "Artifact 3" }))}
          </div>
        </div>
      </div>

      <div>
        <div class="card meta-card">
          <h3>Identity</h3>
          <div class="meta-row"><span class="k">Tag ID</span><span class="v mono">${data.tag_id}</span></div>
          <div class="meta-row"><span class="k">Batch</span>
            <span class="v">${data.batch?.name
              ? raw(`<a href="#/batch/${encodeURIComponent(data.batch.id)}">${escapeHtml(data.batch.name)}</a>`)
              : "—"}</span>
          </div>
        </div>

        <div class="card meta-card">
          <h3>HUID</h3>
          <div class="huid-compare">
            <div class="slot"><span class="lbl">Expected</span>${escapeHtml(data.expected_huid || "—")}</div>
            <div class="equals ${data.huid_match === true ? "ok" : data.huid_match === false ? "bad" : ""}">
              ${raw(data.huid_match === true ? '<i data-lucide="check"></i>' : data.huid_match === false ? '<i data-lucide="x"></i>' : '<i data-lucide="minus"></i>')}
            </div>
            <div class="slot"><span class="lbl">Actual</span>${escapeHtml(composedActualHuid(data) || "—")}</div>
          </div>
          <div class="meta-row"><span class="k">Confidence</span><span class="v">${formatConfidence(data.confidence)}</span></div>
          ${raw(confidenceBar(data.confidence))}
        </div>

        <div class="card meta-card">
          <h3>Classification</h3>
          <div class="meta-row"><span class="k">Purity code</span><span class="v">${escapeHtml(data.purity_code || "—")}</span></div>
          <div class="meta-row"><span class="k">Karat</span><span class="v">${escapeHtml(data.karat || "—")}</span></div>
          <div class="meta-row"><span class="k">Purity %</span><span class="v">${data.purity_percentage != null ? (data.purity_percentage + "%") : "—"}</span></div>
        </div>

        <div class="card meta-card">
          <h3>Decision</h3>
          <div style="margin:8px 0">${raw(decisionChip(data.decision))}</div>
          ${raw((data.rejection_reasons || []).length
            ? `<ul class="reasons">${data.rejection_reasons.map(r => `<li>${escapeHtml(r)}</li>`).join("")}</ul>`
            : `<p class="muted" style="margin:4px 0 0">No rejection reasons.</p>`)}
          <details class="raw">
            <summary>Raw OCR text</summary>
            <pre class="raw-body">${escapeHtml(data.raw_ocr_text || "")}</pre>
          </details>
        </div>

        <div class="card meta-card">
          <h3>Timestamps</h3>
          <div class="meta-row"><span class="k">Processed</span><span class="v">${escapeHtml(formatDate(data.timestamps?.processed_at))}</span></div>
          <div class="meta-row"><span class="k">Processing time</span><span class="v">${data.timestamps?.processing_time_ms != null ? `${data.timestamps.processing_time_ms} ms` : "—"}</span></div>
          <div class="meta-row"><span class="k">Status</span><span class="v">${raw(statusChip(data.processing_status))}</span></div>
        </div>
      </div>
    </div>

    ${raw(compact ? "" : `
      <div class="action-bar" role="toolbar" aria-label="Item actions">
        <button class="btn btn-tertiary" data-action="manual_review">Request re-capture</button>
        <button class="btn btn-danger"   data-action="rejected">Reject</button>
        <button class="btn btn-primary"  data-action="approved">Approve</button>
      </div>
    `)}
  `;

  refreshIcons(el);
  wireGallery(el, { tagId: data.tag_id, huid, bySlot });
  if (!compact) wireDecisionBar(el, data);
}

function wireDecisionBar(el, data) {
  const bar = el.querySelector(".action-bar");
  if (!bar) return;
  bar.querySelectorAll("button[data-action]").forEach((btn) => {
    btn.addEventListener("click", async () => {
      const decision = btn.dataset.action;

      // Reject prompts for a reason so the rejection_reasons list is meaningful.
      // Cancel on the prompt aborts; empty string means "no specific reason".
      let rejection_reasons;
      if (decision === "rejected") {
        const reason = prompt(`Reason for rejecting ${data.tag_id}? (optional)`, "");
        if (reason === null) return;
        rejection_reasons = reason.trim()
          ? [...(data.rejection_reasons || []), reason.trim()]
          : data.rejection_reasons;
      } else if (decision === "manual_review") {
        rejection_reasons = [...(data.rejection_reasons || []), "needs_recapture"];
      }

      const original = btn.innerHTML;
      bar.querySelectorAll("button").forEach((b) => b.setAttribute("aria-disabled", "true"));
      btn.innerHTML = "Saving…";
      try {
        await api.post(`/stage3/item/${encodeURIComponent(data.tag_id)}/decision`, {
          decision,
          rejection_reasons,
        });
        toast.success(`Marked as ${decision.replace("_", " ")}`);
        // Reload so the decision chip, rejection-reasons list, and any batch
        // lists that link here all pick up the new state.
        location.reload();
      } catch (e) {
        toast.error(e.message || "Could not update decision");
        bar.querySelectorAll("button").forEach((b) => b.removeAttribute("aria-disabled"));
        btn.innerHTML = original;
      }
    });
  });
}

function tile({ type, slot, image, label, active = false }) {
  const bg = image?.url ? `style="background-image:url('${escapeHtml(image.url)}')"` : "";
  const cls = ["tile"];
  if (!image) cls.push("empty");
  if (active) cls.push("active");
  let corner = "";
  if (type === "artifact" && image) {
    corner = `<button class="remove" aria-label="Remove artifact ${slot}" data-remove="${slot}"><i data-lucide="x" style="width:12px;height:12px"></i></button>`;
  } else if (type === "huid" && image) {
    corner = `<button class="remove" aria-label="Replace HUID image" data-replace="1"><i data-lucide="refresh-cw" style="width:12px;height:12px"></i></button>`;
  }
  const inner = image
    ? `<span class="label">${escapeHtml(label)}</span>`
    : `<i data-lucide="plus"></i><span class="label">${escapeHtml(label)}</span>`;
  return `<div class="${cls.join(" ")}" data-type="${type}" data-slot="${slot}" role="button" tabindex="0" ${bg}>${inner}${corner}</div>`;
}

function wireGallery(el, { tagId, huid, bySlot }) {
  const large = el.querySelector("#large");
  const strip = el.querySelector("#strip");
  if (!large || !strip) return;

  const setLarge = (url) => {
    if (url) {
      large.style.backgroundImage = `url('${url}')`;
      large.classList.remove("empty");
    } else {
      large.style.backgroundImage = "";
      large.classList.add("empty");
    }
  };

  strip.querySelectorAll(".tile").forEach((tile) => {
    const { type, slot } = tile.dataset;
    const slotNum = Number(slot);
    const currentImg = type === "huid" ? huid : bySlot[slotNum];

    // Click to swap large viewer, or open file picker on an empty tile.
    tile.addEventListener("click", (e) => {
      if (e.target.closest("[data-remove], [data-replace]")) return;
      if (!tile.classList.contains("empty")) {
        strip.querySelectorAll(".tile").forEach((t) => t.classList.remove("active"));
        tile.classList.add("active");
        const url = currentImg?.url || tile.style.backgroundImage?.replace(/^url\(['"]?|['"]?\)$/g, "");
        setLarge(url);
      } else {
        openPicker(tagId, type, slotNum);
      }
    });

    tile.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); tile.click(); } });

    // Drag-and-drop upload onto any tile (HUID or artifact).
    tile.addEventListener("dragover", (e) => { e.preventDefault(); tile.classList.add("dragover"); });
    tile.addEventListener("dragleave", () => tile.classList.remove("dragover"));
    tile.addEventListener("drop", async (e) => {
      e.preventDefault();
      tile.classList.remove("dragover");
      const file = e.dataTransfer?.files?.[0];
      if (file) await uploadTile(tagId, type, slotNum, file);
    });

    // Remove artifact
    const remove = tile.querySelector("[data-remove]");
    if (remove) {
      remove.addEventListener("click", async (e) => {
        e.stopPropagation();
        try {
          await api.del(`/stage2/artifact/${encodeURIComponent(tagId)}/${slotNum}`);
          toast.success(`Removed artifact ${slotNum}`);
          location.reload();
        } catch (err) {
          toast.error(err.message || "Delete failed");
        }
      });
    }

    // Replace HUID (re-open picker)
    const replace = tile.querySelector("[data-replace]");
    if (replace) {
      replace.addEventListener("click", (e) => {
        e.stopPropagation();
        openPicker(tagId, "huid", 0);
      });
    }
  });

  // Arrow keys cycle through tiles when one is focused.
  strip.addEventListener("keydown", (e) => {
    if (!["ArrowLeft", "ArrowRight"].includes(e.key)) return;
    const tiles = [...strip.querySelectorAll(".tile")];
    const idx = tiles.indexOf(document.activeElement);
    if (idx < 0) return;
    const next = tiles[(idx + (e.key === "ArrowRight" ? 1 : -1) + tiles.length) % tiles.length];
    next?.focus();
    next?.click();
  });
}

function openPicker(tagId, type, slot) {
  const input = document.createElement("input");
  input.type = "file";
  input.accept = "image/*";
  input.addEventListener("change", async () => {
    const file = input.files?.[0];
    if (file) await uploadTile(tagId, type, slot, file);
  });
  input.click();
}

async function uploadTile(tagId, type, slot, file) {
  try {
    if (type === "huid") {
      const fd = new FormData();
      fd.append("tag_id", tagId);
      fd.append("image_type", "huid");
      fd.append("slot", "0");
      fd.append("file", file);
      toast.info("Running OCR…", { timeout: 2500 });
      await api.postForm("/stage2/upload-image", fd);
      toast.success("HUID uploaded");
    } else {
      const fd = new FormData();
      fd.append("tag_id", tagId);
      fd.append("slot", String(slot));
      fd.append("file", file);
      await api.postForm("/stage2/upload-artifact", fd);
      toast.success(`Artifact ${slot} uploaded`);
    }
    location.reload();
  } catch (e) {
    toast.error(e.message || "Upload failed");
  }
}

function confidenceBar(value) {
  if (value == null) return "";
  const pct = Math.max(0, Math.min(1, value)) * 100;
  const color = value >= 0.85 ? "var(--success)" : value >= 0.5 ? "var(--warning)" : "var(--danger)";
  return `<div class="progress" style="width:100%"><span style="width:${pct}%; background:${color}"></span></div>`;
}
