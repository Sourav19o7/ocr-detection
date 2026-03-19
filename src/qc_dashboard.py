"""
QC Dashboard for Jewelry Hallmarking OCR Validation.

A Streamlit-based dashboard for testing and using the QC validation endpoints.
Supports both integration flows:
1. Hallmarking Stage - Real-time validation
2. QC Dashboard - Batch validation with approval workflow
"""

import streamlit as st
import requests
from PIL import Image
import io
import json
from datetime import datetime
import sys

sys.path.insert(0, "../config")
sys.path.insert(0, "config")

try:
    from jewelry_rulesets import (
        JewelryType, ErrorCategory, ErrorSeverity,
        JEWELRY_RULESETS, ERROR_DETAILS, ConfidenceBenchmark,
        JewelryRulesetValidator
    )
    RULESETS_AVAILABLE = True
except ImportError:
    RULESETS_AVAILABLE = False

API_BASE_URL = "http://localhost:8000"

st.set_page_config(
    page_title="BAC Hallmark QC",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------
# Design system:
#   Font:         DM Sans 400-800
#   BG:           #F5F1F2  (page)  |  #FFFFFF (card/surface)  |  #F8F6F7 (input/alt)
#   Border:       #E8E2E4  (normal)  |  #F0ECED (light)
#   Text:         #2D2D2D  (primary)  |  #5A5A6B (secondary)  |  #9B9BAB (hint)
#   Accent:       #7B1F3A  (maroon)  |  #9B3A56 (maroon hover)
#   Semantic:     #16A34A (success)  |  #DC2626 (danger)  |  #D97706 (warning)  |  #2563EB (info)
#
# WCAG AA contrast (on #FFFFFF):
#   #2D2D2D = 12.6:1   #5A5A6B = 6.1:1   #9B9BAB = 3.2:1 (large text only)
#   #7B1F3A = 8.9:1    #16A34A = 4.6:1   #DC2626 = 5.6:1  #D97706 = 4.5:1
# ---------------------------------------------------------------------------
st.markdown("""
<style>
/* ── Minimal overrides — theme handles base colors ─────── */
/* The Streamlit theme (config.toml) sets:
   base=light, primaryColor=#7B1F3A, bg=#F5F1F2,
   secondaryBg=#FFFFFF, textColor=#2D2D2D.
   We only customize what the theme can't handle. */

*, *::before, *::after { font-family: 'DM Sans', sans-serif !important; }

#MainMenu, footer, header,
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stStatusWidget"] {
    display: none !important;
    visibility: hidden !important;
}

.block-container {
    padding: 1.25rem 2rem 2.5rem !important;
    max-width: 1360px;
}

/* ── Sidebar ────────────────────────────────────────────── */
section[data-testid="stSidebar"] > div {
    padding-top: 1.25rem !important;
}

/* Collapsed sidebar expand button */
[data-testid="collapsedControl"] {
    border-right: 1px solid #E8E2E4 !important;
}

/* ── Tabs ───────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    gap: 2px;
    background: #FFFFFF;
    border-radius: 10px;
    padding: 4px;
    border: 1px solid #E8E2E4;
}

.stTabs [data-baseweb="tab"] {
    height: 38px;
    background: transparent !important;
    border-radius: 7px;
    font-weight: 600;
    font-size: 12.5px;
    padding: 0 18px;
    border: none;
    white-space: nowrap;
}

.stTabs [data-baseweb="tab"]:hover {
    background: rgba(123,31,58,0.05) !important;
}

.stTabs [aria-selected="true"] {
    background: #7B1F3A !important;
    color: #FFFFFF !important;
    font-weight: 700;
}

/* Force white text inside selected tab */
.stTabs [aria-selected="true"] * {
    color: #FFFFFF !important;
}

.stTabs [data-baseweb="tab-highlight"],
.stTabs [data-baseweb="tab-border"] { display: none !important; }

/* ── Buttons ────────────────────────────────────────────── */
/* Force white text on ALL children inside buttons */
.stButton > button { border-radius: 7px; padding: 10px 24px; font-weight: 600; font-size: 13px; }
.stButton > button * { color: #FFFFFF !important; }
.stButton > button:hover { background: #9B3A56 !important; }
.stButton > button:active { background: #5A1529 !important; }
.stButton > button:focus-visible { outline: 2px solid #7B1F3A !important; outline-offset: 2px !important; }

/* Sidebar buttons are secondary style */
[data-testid="stSidebar"] .stButton > button {
    background: #F8F6F7 !important;
    color: #7B1F3A !important;
    border: 1px solid #E8E2E4 !important;
}
[data-testid="stSidebar"] .stButton > button * { color: #7B1F3A !important; }

/* ── Form controls ──────────────────────────────────────── */
.stSelectbox [data-baseweb="select"] > div,
.stTextInput > div > div > input,
.stNumberInput > div > div > input {
    background: #F8F6F7 !important;
    border: 1px solid #E8E2E4 !important;
    border-radius: 7px !important;
}

.stSelectbox [data-baseweb="select"] > div:focus-within,
.stTextInput > div > div > input:focus {
    border-color: #7B1F3A !important;
    box-shadow: 0 0 0 3px rgba(123,31,58,0.1) !important;
}

.stTextArea textarea {
    background: #F8F6F7 !important;
    border: 1px solid #E8E2E4 !important;
    border-radius: 7px !important;
}

.stTextArea textarea:focus {
    border-color: #7B1F3A !important;
    box-shadow: 0 0 0 3px rgba(123,31,58,0.1) !important;
}

/* ── Dropdown popup ────────────────────────────────────── */
[data-baseweb="popover"],
[data-baseweb="menu"],
ul[role="listbox"] {
    background: #FFFFFF !important;
    border: 1px solid #E8E2E4 !important;
    border-radius: 8px !important;
    box-shadow: 0 4px 20px rgba(0,0,0,0.08) !important;
}

[data-baseweb="menu"] li,
ul[role="listbox"] li {
    background: #FFFFFF !important;
}

[data-baseweb="menu"] li:hover,
ul[role="listbox"] li:hover {
    background: rgba(123,31,58,0.05) !important;
}

[data-baseweb="menu"] li[aria-selected="true"],
ul[role="listbox"] li[aria-selected="true"] {
    background: rgba(123,31,58,0.08) !important;
}

/* ── File uploader ──────────────────────────────────────── */
[data-testid="stFileUploader"] section {
    background: #F8F6F7 !important;
    border: 2px dashed #E8E2E4 !important;
    border-radius: 10px !important;
}

[data-testid="stFileUploader"] section:hover {
    border-color: #7B1F3A !important;
}

/* Browse button white text */
[data-testid="stFileUploader"] button * { color: #FFFFFF !important; }

/* Delete button reset */
[data-testid="stFileUploader"] [data-testid="baseButton-secondary"] {
    background: transparent !important;
    border: none !important;
    min-width: auto !important;
}
[data-testid="stFileUploader"] [data-testid="baseButton-secondary"] * {
    color: #5A5A6B !important;
}

/* ── Image ──────────────────────────────────────────────── */
[data-testid="stImage"] {
    border-radius: 8px;
    overflow: hidden;
    border: 1px solid #E8E2E4;
}

/* ── Metrics ────────────────────────────────────────────── */
[data-testid="stMetric"] {
    background: #F8F6F7 !important;
    border: 1px solid #E8E2E4 !important;
    border-radius: 8px !important;
    padding: 14px 16px !important;
}

[data-testid="stMetricValue"],
[data-testid="stMetricValue"] * {
    color: #7B1F3A !important;
    font-weight: 800 !important;
}

/* ── Expander ───────────────────────────────────────────── */
[data-testid="stExpander"] {
    border: 1px solid #E8E2E4 !important;
    border-radius: 8px !important;
    overflow: hidden;
}

[data-testid="stExpander"] summary {
    background: #F8F6F7 !important;
}

/* ── Spinner ────────────────────────────────────────────── */
.stSpinner > div > div { border-top-color: #7B1F3A !important; }

/* ── Scrollbar ──────────────────────────────────────────── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #F5F1F2; }
::-webkit-scrollbar-thumb { background: #D8D2D4; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #7B1F3A; }

/* ══════════════════════════════════════════════════════════
   CUSTOM COMPONENT CLASSES
   ══════════════════════════════════════════════════════════ */

/* Pastel semantic palette */

/* Top bar */
.qc-topbar {
    background: #7B1F3A;
    padding: 0 24px;
    height: 50px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    border-radius: 10px;
    margin-bottom: 20px;
    box-shadow: 0 2px 12px rgba(123,31,58,0.15);
    color: #FFFFFF;
}

.qc-topbar * { color: #FFFFFF !important; }

.qc-topbar-left {
    display: flex;
    align-items: center;
    gap: 12px;
}

.qc-topbar-logo {
    width: 32px; height: 32px;
    background: rgba(255,255,255,0.15);
    border-radius: 7px;
    display: flex; align-items: center; justify-content: center;
    font-size: 11px; font-weight: 800; letter-spacing: 0.3px;
}

.qc-topbar-name  { font-size: 14px; font-weight: 700; }
.qc-topbar-sub   { font-size: 10px; opacity: 0.75; letter-spacing: 0.6px; text-transform: uppercase; }
.qc-topbar-right { font-size: 11px; opacity: 0.85; letter-spacing: 0.5px; }

/* Section title */
.sec-title {
    font-size: 11px !important;
    font-weight: 700 !important;
    color: #7B1F3A !important;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    border-bottom: 2px solid #7B1F3A;
    display: inline-block;
    padding-bottom: 6px;
    margin-bottom: 4px;
}

.sec-desc {
    font-size: 13px !important;
    color: #6B6B7B !important;
    margin-bottom: 20px;
}

/* Card title */
.card-label {
    font-size: 10px !important;
    font-weight: 700 !important;
    color: #7B1F3A !important;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin-bottom: 12px;
    padding-bottom: 8px;
    border-bottom: 1px solid #F0ECED;
}

/* Decision badge — pastel */
.dec-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 6px 18px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.6px;
}

.dec-approved { background: #E8F5E9; color: #2E7D32 !important; }
.dec-rejected { background: #FDECEA; color: #C62828 !important; }
.dec-review   { background: #FFF3E0; color: #E65100 !important; }

/* Confidence gauge */
.gauge { margin: 14px 0; }
.gauge-head { display: flex; justify-content: space-between; margin-bottom: 6px; }
.gauge-lbl  { font-size: 12px; font-weight: 600; color: #6B6B7B !important; }
.gauge-val  { font-size: 13px; font-weight: 700; }
.gauge-track { height: 8px; background: #F0ECED; border-radius: 4px; overflow: hidden; }
.gauge-fill  { height: 100%; border-radius: 4px; transition: width 0.4s ease; }

/* Error card */
.ecard {
    background: #FFFFFF;
    border: 1px solid #E8E2E4;
    border-radius: 8px;
    padding: 14px 16px;
    margin: 8px 0;
}
.ecard-critical { border-left: 4px solid #EF5350; }
.ecard-major    { border-left: 4px solid #FFA726; }
.ecard-minor    { border-left: 4px solid #FFCC80; }

.ecard-sev {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 10px;
    font-size: 9px;
    font-weight: 700;
    letter-spacing: 0.5px;
    text-transform: uppercase;
    margin-right: 8px;
}

.ecard-name { font-weight: 700; font-size: 13px; color: #2D2D2D !important; }
.ecard-msg  { font-size: 12px; color: #6B6B7B !important; margin: 6px 0 0; line-height: 1.5; }
.ecard-sug  { font-size: 11px; color: #2E7D32 !important; font-weight: 600; margin: 6px 0 0; }

/* Checklist row */
.ck-row {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 7px 0;
    font-size: 13px;
    font-weight: 600;
    color: #2D2D2D !important;
}

.ck-icon {
    width: 22px; height: 22px;
    border-radius: 50%;
    display: inline-flex;
    align-items: center; justify-content: center;
    font-size: 12px; font-weight: 700;
    flex-shrink: 0;
}

.ck-pass { background: #E8F5E9; color: #2E7D32 !important; }
.ck-fail { background: #FDECEA; color: #C62828 !important; }

/* Score pill */
.sc-pill {
    background: #F8F6F7;
    border: 1px solid #E8E2E4;
    border-radius: 8px;
    padding: 8px 14px;
    margin: 8px 0;
    display: flex;
    align-items: center;
    gap: 10px;
}

.sc-pill-label { font-weight: 700; font-size: 12px; }
.sc-pill-range { font-size: 11px; color: #9B9BAB !important; }

/* Position card */
.pos {
    border-radius: 8px;
    padding: 10px 14px;
    margin: 6px 0;
}

.pos-ok {
    background: rgba(22,163,74,0.04);
    border: 1px solid rgba(22,163,74,0.18);
}

.pos-ok-pref {
    background: rgba(22,163,74,0.04);
    border: 2px solid rgba(22,163,74,0.3);
}

.pos-no {
    background: rgba(220,38,38,0.03);
    border: 1px solid rgba(220,38,38,0.15);
}

.pos-name { font-weight: 700; font-size: 12px; }
.pos-desc { font-size: 11px; color: #5A5A6B !important; margin-top: 3px; }
.pos-why  { font-size: 11px; color: #DC2626 !important; font-weight: 600; margin-top: 3px; }

/* Rule badge */
.rbadge {
    display: inline-block;
    background: rgba(123,31,58,0.06);
    color: #7B1F3A !important;
    padding: 5px 14px;
    border-radius: 14px;
    font-size: 11px;
    font-weight: 600;
    margin: 3px 2px;
    line-height: 1.4;
}

/* Threshold card */
.tcard {
    background: #FFFFFF;
    border: 1px solid #E8E2E4;
    border-radius: 10px;
    padding: 22px 16px;
    text-align: center;
}
.tcard-val   { font-size: 1.75rem; font-weight: 800; line-height: 1; }
.tcard-label { font-weight: 700; font-size: 12px; text-transform: uppercase; letter-spacing: 0.5px; margin-top: 10px; color: #2D2D2D !important; }
.tcard-hint  { font-size: 11px; color: #9B9BAB !important; margin-top: 4px; }

/* Data row */
.drow {
    display: flex;
    align-items: center;
    padding: 11px 16px;
    background: #FFFFFF;
    border: 1px solid #E8E2E4;
    border-radius: 8px;
    margin: 5px 0;
}
.drow-key { flex: 0 0 130px; font-weight: 700; font-size: 13px; color: #7B1F3A !important; }
.drow-val { flex: 1; font-size: 12px; color: #5A5A6B !important; }

/* Error ref */
.eref {
    background: #FFFFFF;
    padding: 12px 16px;
    border-radius: 8px;
    border: 1px solid #E8E2E4;
    margin: 5px 0;
}

.eref code {
    background: rgba(123,31,58,0.06);
    color: #7B1F3A !important;
    padding: 2px 6px;
    border-radius: 4px;
    font-size: 11px;
    font-weight: 600;
}

.eref-desc { color: #2D2D2D !important; font-size: 12px; margin: 6px 0 0; }
.eref-sug  { color: #16A34A !important; font-size: 11px; font-weight: 600; margin: 4px 0 0; }

/* Sidebar custom elements */
.sb-brand {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 0 0 12px 0;
}

.sb-icon {
    width: 34px; height: 34px;
    background: rgba(123,31,58,0.07);
    border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    color: #7B1F3A !important;
    font-weight: 800;
    font-size: 14px;
}

.sb-name { font-size: 14px; font-weight: 800; color: #7B1F3A !important; }
.sb-sub  { font-size: 9px; color: #9B9BAB !important; text-transform: uppercase; letter-spacing: 1.2px; font-weight: 600; }

.sb-feat {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 12px;
    color: #5A5A6B !important;
    padding: 4px 0;
}

.sb-feat-icon { color: #16A34A !important; font-size: 13px; font-weight: 700; }

/* Empty state */
.empty-state {
    text-align: center;
    padding: 48px 24px;
}

.empty-icon {
    font-size: 40px;
    opacity: 0.18;
    margin-bottom: 8px;
}

.empty-title {
    font-size: 14px;
    font-weight: 600;
    color: #5A5A6B !important;
    margin-bottom: 4px;
}

.empty-hint {
    font-size: 12px;
    color: #9B9BAB !important;
}

/* Guidelines block */
.guide-heading {
    font-weight: 700;
    font-size: 13px;
    color: #2D2D2D !important;
    margin-bottom: 6px;
}

.guide-item {
    display: flex;
    align-items: flex-start;
    gap: 8px;
    padding: 4px 0;
    font-size: 12px;
    color: #5A5A6B !important;
    line-height: 1.5;
}

.guide-bullet {
    color: #9B9BAB !important;
    flex-shrink: 0;
    margin-top: 1px;
}

/* Component threshold card */
.cthresh {
    background: #FFFFFF;
    padding: 12px 16px;
    border-radius: 8px;
    border: 1px solid #E8E2E4;
    margin: 5px 0;
}

.cthresh-name { color: #7B1F3A !important; font-weight: 700; font-size: 13px; }
.cthresh-min  { color: #2D2D2D !important; font-weight: 700; font-size: 13px; }
.cthresh-desc { color: #9B9BAB !important; font-size: 11px; margin-top: 4px; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def check_api_health():
    try:
        return requests.get(f"{API_BASE_URL}/health", timeout=5).status_code == 200
    except Exception:
        return False


def conf_color(s):
    if s >= 0.85:
        return "#16A34A"
    if s >= 0.50:
        return "#D97706"
    return "#DC2626"


def html_gauge(score, label="Confidence"):
    c = conf_color(score)
    p = score * 100
    return f"""
    <div class="gauge">
        <div class="gauge-head">
            <span class="gauge-lbl">{label}</span>
            <span class="gauge-val" style="color:{c};">{p:.1f}%</span>
        </div>
        <div class="gauge-track"><div class="gauge-fill" style="width:{p}%;background:{c};"></div></div>
    </div>"""


def html_decision(decision):
    m = {"approved": ("dec-approved", "&#10003;"), "rejected": ("dec-rejected", "&#10007;"), "manual_review": ("dec-review", "&#9679;")}
    cls, icon = m.get(decision, ("dec-review", "&#9679;"))
    return f'<span class="dec-badge {cls}">{icon} {decision.replace("_"," ").upper()}</span>'


def html_check_row(label, passed):
    cls = "ck-pass" if passed else "ck-fail"
    icon = "&#10003;" if passed else "&#10007;"
    return f'<div class="ck-row"><span class="ck-icon {cls}">{icon}</span>{label}</div>'


def html_error_card(title, message, suggestion=None, severity="major"):
    sev = {"critical": ("#DC2626", "CRITICAL"), "major": ("#D97706", "MAJOR"), "minor": ("#C8A44E", "MINOR")}
    c, lbl = sev.get(severity, sev["major"])
    sug = f'<div class="ecard-sug">Suggestion: {suggestion}</div>' if suggestion else ""
    return f"""
    <div class="ecard ecard-{severity}">
        <div><span class="ecard-sev" style="background:{c}14;color:{c};">{lbl}</span><span class="ecard-name">{title}</span></div>
        <div class="ecard-msg">{message}</div>
        {sug}
    </div>"""


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div class="sb-brand">
            <div class="sb-icon">B</div>
            <div><div class="sb-name">QC Dashboard</div><div class="sb-sub">Hallmark Validation</div></div>
        </div>""", unsafe_allow_html=True)

        st.divider()

        online = check_api_health()
        dot = "background:#16A34A" if online else "background:#DC2626"
        txt = "Online" if online else "Offline"
        st.markdown(f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:4px;"><span style="width:8px;height:8px;border-radius:50%;{dot};display:inline-block;"></span><span style="font-size:12px;font-weight:600;color:#2D2D2D !important;">API: {txt}</span></div>', unsafe_allow_html=True)

        st.markdown(f"[API Documentation]({API_BASE_URL}/docs)")
        st.divider()

        st.markdown('<div style="font-size:9px;font-weight:700;text-transform:uppercase;letter-spacing:1px;color:#9B9BAB !important;margin-bottom:4px;">Features</div>', unsafe_allow_html=True)
        for f in ["Hallmark validation", "Error categorization", "Jewelry rulesets", "Confidence benchmarks", "QC override"]:
            st.markdown(f'<div class="sb-feat"><span class="sb-feat-icon">&#10003;</span><span>{f}</span></div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Tab: Validate
# ---------------------------------------------------------------------------
def render_validate_tab():
    st.markdown('<div class="sec-title">Hallmark Validation</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-desc">Upload an image to validate hallmark against BIS standards with detailed error analysis.</div>', unsafe_allow_html=True)

    col_form, col_result = st.columns([1, 1], gap="large")

    # ── Left column: form ──
    with col_form:
        jewelry_type = st.selectbox("Jewelry Type", ["ring", "bangle", "chain", "necklace", "pendant", "earring", "mangalsutra", "other"], help="Select jewelry type for position validation")

        uploaded_file = st.file_uploader("Upload hallmark image", type=["png", "jpg", "jpeg", "bmp", "webp"], key="hallmark_uploader")

        job_id = st.text_input("Job ID", value=f"HM-{datetime.now().strftime('%Y%m%d%H%M%S')}", help="Unique job identifier")

        c1, c2 = st.columns(2)
        with c1:
            expected_purity = st.selectbox("Expected Purity", ["", "916", "750", "585", "375", "875", "958", "999"], help="For cross-validation")
        with c2:
            marking_position = st.selectbox("Marking Position", ["12_oclock", "3_oclock", "9_oclock", "6_oclock", "unknown"], help="Position of hallmark on ring")

        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)

            if st.button("Validate Hallmark", use_container_width=True, key="validate_btn"):
                with st.spinner("Analyzing..."):
                    try:
                        uploaded_file.seek(0)
                        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                        response = requests.post(f"{API_BASE_URL}/qc/validate/v2", files=files)

                        if response.status_code == 200:
                            result = response.json()
                            if jewelry_type == "ring" and marking_position == "6_oclock" and "data" in result:
                                if result["data"].get("rejection_info") is None:
                                    result["data"]["rejection_info"] = {"reasons": [], "message": ""}
                                result["data"]["rejection_info"]["reasons"].append("marking_at_6_oclock")
                                result["data"]["rejection_info"]["message"] += "; Ring hallmark at 6 o'clock position (bottom inside) is NOT acceptable"
                                result["data"]["decision"] = "rejected"
                            st.session_state.validation_result = result
                            st.session_state.jewelry_type = jewelry_type
                            st.session_state.marking_position = marking_position
                        else:
                            st.error(f"Error: {response.text}")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")

    # ── Right column: results ──
    with col_result:
        st.markdown('<div class="card-label">Validation Result</div>', unsafe_allow_html=True)

        if "validation_result" not in st.session_state or not st.session_state.validation_result:
            st.markdown('<div class="empty-state"><div class="empty-icon">&#9678;</div><div class="empty-title">No results yet</div><div class="empty-hint">Upload an image and click Validate Hallmark</div></div>', unsafe_allow_html=True)
            return

        result = st.session_state.validation_result
        data = result.get("data", {})
        decision = data.get("decision", "unknown")
        confidence = data.get("confidence", 0)

        # Decision + gauge
        st.markdown(html_decision(decision), unsafe_allow_html=True)
        st.markdown(html_gauge(confidence, "Overall Confidence"), unsafe_allow_html=True)

        # Score pill
        for lbl, lo, hi, c in [("EXCELLENT",.95,1,"#16A34A"),("GOOD",.85,.95,"#16A34A"),("ACCEPTABLE",.70,.85,"#D97706"),("POOR",.50,.70,"#D97706"),("UNACCEPTABLE",0,.50,"#DC2626")]:
            if lo <= confidence <= hi:
                st.markdown(f'<div class="sc-pill"><span class="sc-pill-label" style="color:{c} !important;">Score: {lbl}</span><span class="sc-pill-range">{lo*100:.0f}% &ndash; {hi*100:.0f}%</span></div>', unsafe_allow_html=True)
                break

        # Hallmark data
        hd = data.get("hallmark_data", {})
        if hd:
            st.markdown('<div class="card-label" style="margin-top:16px;">Detected Hallmark</div>', unsafe_allow_html=True)
            m1, m2 = st.columns(2)
            with m1:
                st.metric("Purity Code", hd.get("purity_code") or "--")
                st.metric("Karat", hd.get("karat") or "--")
            with m2:
                pp = hd.get("purity_percentage")
                st.metric("Purity %", f"{pp}%" if pp else "--")
                st.metric("HUID", hd.get("huid") or "--")

        # Checklist
        st.markdown('<div class="card-label" style="margin-top:16px;">Validation Checklist</div>', unsafe_allow_html=True)
        vs = data.get("validation_status", {})
        checks_html = ""
        checks_html += html_check_row("Purity Valid", vs.get("purity_valid", False))
        checks_html += html_check_row("HUID Valid", vs.get("huid_valid", False))
        checks_html += html_check_row("BIS Certified", data.get("bis_certified", False))
        st.markdown(checks_html, unsafe_allow_html=True)

        # Errors
        rejection_info = data.get("rejection_info")
        if rejection_info or decision != "approved":
            st.markdown('<div class="card-label" style="margin-top:16px;">Error Analysis</div>', unsafe_allow_html=True)
            reasons = rejection_info.get("reasons", []) if rejection_info else []
            errors_html = ""

            for r in reasons:
                if r.startswith("image_"):
                    errors_html += html_error_card(r.replace("_", " ").title(), "Image quality issue detected", "Retake image with better lighting and focus", "major")

            esug = {"missing_huid": ("HUID not detected", "Ensure HUID is clearly engraved and visible", "critical"), "missing_purity_mark": ("Purity mark not found", "Ensure purity code is visible", "critical"), "invalid_purity_code": ("Invalid purity code", "Valid: 375, 585, 750, 875, 916, 958, 999", "critical"), "invalid_huid_format": ("Invalid HUID format", "Must be 6 alphanumeric characters with at least one letter", "critical")}
            for r in reasons:
                if r in esug:
                    i = esug[r]
                    errors_html += html_error_card(i[0], f"Error: {r}", i[1], i[2])

            if any("6_oclock" in r.lower() or "position" in r.lower() for r in reasons) or (st.session_state.get("jewelry_type") == "ring" and st.session_state.get("marking_position") == "6_oclock"):
                errors_html += html_error_card("Incorrect Marking Position", "Ring hallmark at 6 o'clock position (bottom inside) is NOT acceptable.", "Use 12, 3, or 9 o'clock position inside the ring", "critical")

            if confidence < 0.50:
                errors_html += html_error_card("Low Confidence Score", f"OCR confidence ({confidence*100:.1f}%) is below 50%", "Retake image with better conditions", "major")
            elif confidence < 0.85 and decision == "manual_review":
                errors_html += html_error_card("Manual Review Needed", f"Confidence ({confidence*100:.1f}%) requires human verification", "QC personnel should verify the hallmark", "minor")

            if errors_html:
                st.markdown(errors_html, unsafe_allow_html=True)

        with st.expander("View Raw Response"):
            st.json(result)


# ---------------------------------------------------------------------------
# Tab: Jewelry Rules
# ---------------------------------------------------------------------------
def render_jewelry_rules_tab():
    st.markdown('<div class="sec-title">Jewelry-Specific Rulesets</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-desc">Marking rules for different jewelry types.</div>', unsafe_allow_html=True)

    if not RULESETS_AVAILABLE:
        st.warning("Jewelry rulesets module not loaded. Using default rules.")

        default_rulesets = [
            {"type": "ring", "display_name": "Ring", "description": "Finger rings including engagement, wedding, and fashion rings",
             "acceptable_positions": [{"name": "12 o'clock", "description": "Inside at top (preferred)", "preferred": True}, {"name": "3 o'clock", "description": "Inside at right side", "preferred": False}, {"name": "9 o'clock", "description": "Inside at left side", "preferred": False}],
             "forbidden_positions": [{"name": "6 o'clock", "description": "Inside at bottom", "reason": "Wears off quickly due to surface contact"}, {"name": "Outside", "description": "Outer visible surface", "reason": "Affects aesthetics"}],
             "special_rules": ["Hallmark must be inside the band", "Marking should not interfere with stone settings", "6 o'clock position (bottom inside) is NOT acceptable"]},
            {"type": "bangle", "display_name": "Bangle", "description": "Rigid circular bracelets",
             "acceptable_positions": [{"name": "Inside near opening", "description": "Inner surface near clasp", "preferred": True}, {"name": "Inside flat area", "description": "Any flat inner surface", "preferred": False}],
             "forbidden_positions": [{"name": "Outside decorative", "description": "Decorative outer surface", "reason": "Must not interfere with design"}],
             "special_rules": ["Marking should be on inner surface", "Should not be on carved/embossed areas"]},
            {"type": "chain", "display_name": "Chain", "description": "Neck chains and rope chains",
             "acceptable_positions": [{"name": "Clasp area", "description": "On or near the clasp", "preferred": True}, {"name": "Tag attached", "description": "On attached hallmark tag", "preferred": True}],
             "forbidden_positions": [{"name": "Middle links", "description": "On middle chain links", "reason": "Would weaken chain structure"}],
             "special_rules": ["Marking on clasp or attached tag", "Must not weaken any chain link"]},
        ]
    else:
        validator = JewelryRulesetValidator()
        default_rulesets = validator.get_all_rulesets_summary()

    selected_type = st.selectbox("Select Jewelry Type", options=[r["type"] for r in default_rulesets], format_func=lambda x: next((r["display_name"] for r in default_rulesets if r["type"] == x), x))

    ruleset = next((r for r in default_rulesets if r["type"] == selected_type), None)
    if not ruleset:
        return

    st.markdown(f'<div style="background:#FFFFFF;border:1px solid #E8E2E4;border-left:3px solid #7B1F3A;border-radius:10px;padding:16px 20px;margin:8px 0;"><div style="font-size:16px;font-weight:800;color:#7B1F3A !important;">{ruleset["display_name"]}</div><div style="font-size:13px;color:#5A5A6B !important;margin-top:2px;">{ruleset["description"]}</div></div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2, gap="medium")

    with c1:
        st.markdown('<div class="card-label">Acceptable Positions</div>', unsafe_allow_html=True)
        html = ""
        for pos in ruleset.get("acceptable_positions", []):
            cls = "pos-ok-pref" if pos.get("preferred") else "pos-ok"
            pref = '<span style="color:#16A34A !important;font-size:10px;font-weight:700;margin-left:6px;">PREFERRED</span>' if pos.get("preferred") else ""
            html += f'<div class="pos {cls}"><div class="pos-name" style="color:#16A34A !important;">{pos["name"]}{pref}</div><div class="pos-desc">{pos["description"]}</div></div>'
        st.markdown(html, unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="card-label">Forbidden Positions</div>', unsafe_allow_html=True)
        html = ""
        for pos in ruleset.get("forbidden_positions", []):
            html += f'<div class="pos pos-no"><div class="pos-name" style="color:#DC2626 !important;">{pos["name"]}</div><div class="pos-desc">{pos["description"]}</div><div class="pos-why">{pos.get("reason", "Not allowed")}</div></div>'
        st.markdown(html, unsafe_allow_html=True)

    st.markdown('<div class="card-label" style="margin-top:16px;">Special Rules</div>', unsafe_allow_html=True)
    html = ""
    for rule in ruleset.get("special_rules", []):
        html += f'<span class="rbadge">{rule}</span>'
    st.markdown(html, unsafe_allow_html=True)

    if ruleset.get("common_issues"):
        st.markdown('<div class="card-label" style="margin-top:16px;">Common Issues</div>', unsafe_allow_html=True)
        for issue in ruleset.get("common_issues", []):
            st.markdown(f"- {issue.replace('_', ' ').title()}")

    st.divider()
    with st.expander("Add Custom Rule"):
        st.text_input("Jewelry Type Name", key="nr_type")
        st.text_area("Rule Description", key="nr_desc")
        st.selectbox("Position Type", ["Acceptable", "Forbidden"], key="nr_postype")
        st.text_input("Position Name", key="nr_posname")
        st.text_area("Position Description", key="nr_posdesc")
        if st.button("Add Rule"):
            st.success("Rule added (demo only - not persisted)")


# ---------------------------------------------------------------------------
# Tab: Error Categories
# ---------------------------------------------------------------------------
def render_error_categories_tab():
    st.markdown('<div class="sec-title">Error Categories Reference</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-desc">Complete reference of all error categories used in QC validation.</div>', unsafe_allow_html=True)

    categories = {
        "Image Quality": [
            ("image_blurred", "Image is blurred or out of focus", "MAJOR", "Retake with steady camera and proper focus"),
            ("image_overexposed", "Image too bright / washed out", "MAJOR", "Reduce lighting or adjust exposure"),
            ("image_underexposed", "Image too dark", "MAJOR", "Increase lighting"),
            ("image_reflection", "Specular reflection on metal", "MINOR", "Use diffused lighting or change angle"),
            ("image_low_resolution", "Resolution too low for OCR", "MAJOR", "Use higher resolution or move closer"),
        ],
        "OCR / Detection": [
            ("low_confidence", "OCR confidence below threshold", "MAJOR", "Retake image with better conditions"),
            ("partial_detection", "Only partial hallmark detected", "MAJOR", "Ensure full hallmark is visible"),
            ("ocr_misread", "Characters incorrectly read", "MAJOR", "Verify manually or retake"),
            ("no_detection", "No text detected", "CRITICAL", "Check if hallmark exists and is visible"),
        ],
        "Hallmark Content": [
            ("invalid_purity_code", "Purity code not BIS standard", "CRITICAL", "Valid: 375, 585, 750, 875, 916, 958, 999"),
            ("invalid_huid_format", "HUID format incorrect", "CRITICAL", "Must be 6 alphanumeric with 1+ letter"),
            ("missing_huid", "HUID not present", "CRITICAL", "HUID mandatory since April 2023"),
            ("missing_purity_mark", "Purity mark not found", "CRITICAL", "Purity code required on all items"),
            ("missing_bis_logo", "BIS logo not detected", "MAJOR", "BIS certification mark required"),
        ],
        "Position / Placement": [
            ("marking_at_6_oclock", "Ring marked at bottom (6 o'clock)", "CRITICAL", "Use 12, 3, or 9 o'clock position"),
            ("wrong_position", "Marking in unacceptable area", "MAJOR", "Refer to jewelry-specific rules"),
            ("marking_on_decorative", "Marking on design area", "MAJOR", "Place on non-decorative surface"),
        ],
        "Engraving Quality": [
            ("shallow_engraving", "Engraving too shallow", "MAJOR", "Must be deep enough for longevity"),
            ("uneven_engraving", "Inconsistent engraving depth", "MINOR", "Check engraving equipment"),
            ("smudged_marking", "Marking is smudged / unclear", "MAJOR", "Requires re-engraving"),
        ],
    }

    for cat, errors in categories.items():
        st.markdown(f'<div class="card-label" style="margin-top:16px;">{cat}</div>', unsafe_allow_html=True)
        html = ""
        for code, desc, sev, sug in errors:
            sc = {"CRITICAL": "#DC2626", "MAJOR": "#D97706", "MINOR": "#C8A44E"}.get(sev, "#5A5A6B")
            html += f"""<div class="eref" style="border-left:3px solid {sc};">
                <div style="display:flex;justify-content:space-between;align-items:center;">
                    <code>{code}</code>
                    <span class="ecard-sev" style="background:{sc}14;color:{sc};">{sev}</span>
                </div>
                <div class="eref-desc">{desc}</div>
                <div class="eref-sug">Suggestion: {sug}</div>
            </div>"""
        st.markdown(html, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Tab: Benchmarks
# ---------------------------------------------------------------------------
def render_benchmarks_tab():
    st.markdown('<div class="sec-title">Confidence Score Benchmarks</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-desc">Understanding confidence scores and decision thresholds.</div>', unsafe_allow_html=True)

    st.markdown('<div class="card-label">Decision Thresholds</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3, gap="medium")
    with c1:
        st.markdown('<div class="tcard" style="border-top:3px solid #16A34A;"><div class="tcard-val" style="color:#16A34A !important;">&ge; 85%</div><div class="tcard-label">Auto-Approve</div><div class="tcard-hint">No human review needed</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="tcard" style="border-top:3px solid #D97706;"><div class="tcard-val" style="color:#D97706 !important;">50 &ndash; 85%</div><div class="tcard-label">Manual Review</div><div class="tcard-hint">QC personnel verification</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="tcard" style="border-top:3px solid #DC2626;"><div class="tcard-val" style="color:#DC2626 !important;">&lt; 50%</div><div class="tcard-label">Auto-Reject</div><div class="tcard-hint">Requires re-capture</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="card-label" style="margin-top:24px;">Score Interpretation</div>', unsafe_allow_html=True)
    html = ""
    for lbl, lo, hi, c, desc in [("Excellent",95,100,"#16A34A","Hallmark clearly readable, all components detected with high accuracy"),("Good",85,95,"#16A34A","Reliable detection, suitable for auto-approval"),("Acceptable",70,85,"#D97706","Readable but some uncertainty, review recommended"),("Poor",50,70,"#D97706","Significant uncertainty, manual verification required"),("Unacceptable",0,50,"#DC2626","Cannot reliably read hallmark, retake image")]:
        html += f'<div class="drow" style="border-left:3px solid {c};"><div class="drow-key" style="color:{c} !important;"><div style="font-size:13px;font-weight:700;">{lbl}</div><div style="font-size:10px;color:#9B9BAB !important;">{lo}% &ndash; {hi}%</div></div><div class="drow-val">{desc}</div></div>'
    st.markdown(html, unsafe_allow_html=True)

    st.markdown('<div class="card-label" style="margin-top:24px;">Component Thresholds</div>', unsafe_allow_html=True)
    html = ""
    for name, thresh, desc in [("Purity Mark",80,"916, 750, etc."),("HUID",80,"6-char alphanumeric"),("BIS Logo",70,"Triangle mark")]:
        html += f'<div class="cthresh"><div style="display:flex;justify-content:space-between;align-items:center;"><span class="cthresh-name">{name}</span><span class="cthresh-min">Min: {thresh}%</span></div><div class="cthresh-desc">{desc}</div></div>'
    st.markdown(html, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Tab: HUID
# ---------------------------------------------------------------------------
def render_huid_tab():
    st.markdown('<div class="sec-title">HUID Format Validator</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-desc">Validate Hallmark Unique Identification codes.</div>', unsafe_allow_html=True)

    c1, c2 = st.columns([1, 1], gap="large")

    with c1:
        huid_input = st.text_input("Enter HUID", placeholder="e.g., AB1234", max_chars=10)
        if st.button("Validate HUID", use_container_width=True):
            if huid_input:
                try:
                    response = requests.post(f"{API_BASE_URL}/validate/huid", data={"huid": huid_input})
                    if response.status_code == 200:
                        st.session_state.huid_result = response.json()
                except Exception as e:
                    st.error(f"Error: {str(e)}")

        st.markdown('<div class="card-label" style="margin-top:24px;">Requirements</div>', unsafe_allow_html=True)
        st.markdown(f"""
        {html_check_row("Exactly 6 characters in length", True)}
        {html_check_row("Characters: A-Z, 0-9 only", True)}
        {html_check_row("Must contain at least one letter", True)}
        {html_check_row("Mandatory since April 2023", True)}
        """, unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="card-label">Result</div>', unsafe_allow_html=True)
        if "huid_result" in st.session_state and st.session_state.huid_result:
            result = st.session_state.huid_result
            if result.get("valid"):
                st.success(f"Valid HUID: {result.get('cleaned')}")
            else:
                st.error("Invalid HUID")
                for error in result.get("errors", []):
                    if error:
                        st.warning(error)
        else:
            st.markdown('<div class="empty-state"><div class="empty-icon">&#9678;</div><div class="empty-title">Enter a HUID code</div><div class="empty-hint">Type a code and click Validate</div></div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Tab: BIS Rules
# ---------------------------------------------------------------------------
def render_bis_rules_tab():
    st.markdown('<div class="sec-title">BIS Compliance Rules</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-desc">Bureau of Indian Standards purity grades reference.</div>', unsafe_allow_html=True)

    try:
        response = requests.get(f"{API_BASE_URL}/qc/rules")
        if response.status_code == 200:
            rules = response.json()
            c1, c2 = st.columns(2, gap="large")

            with c1:
                st.markdown('<div class="card-label">Gold Purity Grades</div>', unsafe_allow_html=True)
                html = ""
                for code, info in rules.get("bis_standards", {}).get("gold_grades", {}).items():
                    html += f'<div class="drow"><div class="drow-key">{code}</div><div class="drow-val">{info.get("karat")} &mdash; {info.get("purity")}%</div></div>'
                st.markdown(html, unsafe_allow_html=True)

            with c2:
                st.markdown('<div class="card-label">Silver Purity Grades</div>', unsafe_allow_html=True)
                html = ""
                for code, info in rules.get("bis_standards", {}).get("silver_grades", {}).items():
                    html += f'<div class="drow"><div class="drow-key">{code}</div><div class="drow-val">{info.get("grade")} &mdash; {info.get("purity")}%</div></div>'
                st.markdown(html, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Failed to load rules: {str(e)}")


# ---------------------------------------------------------------------------
# Tab: Override
# ---------------------------------------------------------------------------
def render_override_tab():
    st.markdown('<div class="sec-title">QC Override</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-desc">Manually override automated QC decisions when human judgement is needed.</div>', unsafe_allow_html=True)

    c1, c2 = st.columns([1, 1], gap="large")

    with c1:
        job_id = st.text_input("Job ID", placeholder="HM-123456789", key="ov_job")
        override_decision = st.selectbox("Override Decision", ["approved", "rejected"], key="ov_dec")
        override_reason = st.text_area("Reason for Override", key="ov_reason")
        operator_id = st.text_input("Operator ID", placeholder="QC-001", key="ov_op")
        notes = st.text_area("Additional Notes", key="ov_notes")

        if st.button("Apply Override", use_container_width=True):
            if job_id and override_reason and operator_id:
                try:
                    response = requests.post(f"{API_BASE_URL}/qc/override", json={"job_id": job_id, "override_decision": override_decision, "override_reason": override_reason, "operator_id": operator_id, "notes": notes})
                    if response.status_code == 200:
                        st.success("Override applied successfully")
                        st.json(response.json())
                except Exception as e:
                    st.error(f"Error: {str(e)}")
            else:
                st.warning("Please fill in Job ID, Reason, and Operator ID")

    with c2:
        st.markdown('<div class="card-label">Override Guidelines</div>', unsafe_allow_html=True)
        st.markdown("""
        <div style="margin-bottom:18px;">
            <div class="guide-heading">When to Override</div>
            <div class="guide-item"><span class="guide-bullet">&#8226;</span>AI misread a clear engraving</div>
            <div class="guide-item"><span class="guide-bullet">&#8226;</span>Unusual font not recognized by OCR</div>
            <div class="guide-item"><span class="guide-bullet">&#8226;</span>Image quality issue but hallmark is readable</div>
        </div>
        <div>
            <div class="guide-heading">Required Information</div>
        """ + html_check_row("Valid Job ID", True) + html_check_row("Clear reason for override", True) + html_check_row("Operator ID for audit trail", True) + """
        </div>
        """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    render_sidebar()

    st.markdown("""
    <div class="qc-topbar">
        <div class="qc-topbar-left">
            <div class="qc-topbar-logo">BAC</div>
            <div><div class="qc-topbar-name">Hallmark QC Dashboard</div><div class="qc-topbar-sub">Jewelry Quality Control System</div></div>
        </div>
        <div class="qc-topbar-right">Bombay Assay Company</div>
    </div>""", unsafe_allow_html=True)

    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["Validate", "Jewelry Rules", "Error Categories", "Benchmarks", "HUID", "BIS Rules", "Override"])

    with tab1:
        render_validate_tab()
    with tab2:
        render_jewelry_rules_tab()
    with tab3:
        render_error_categories_tab()
    with tab4:
        render_benchmarks_tab()
    with tab5:
        render_huid_tab()
    with tab6:
        render_bis_rules_tab()
    with tab7:
        render_override_tab()


if __name__ == "__main__":
    main()
