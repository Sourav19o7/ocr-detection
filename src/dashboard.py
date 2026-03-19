"""
Streamlit Dashboard for OCR Text Detection.
Professional light theme with DM Sans font matching BAC design system.
Supports V1 (standard) and V2 (hallmark-specific) OCR.
"""

import streamlit as st
from PIL import Image
from ocr_model import OCREngine
from ocr_model_v2 import OCREngineV2, HallmarkType
from history import add_to_history, get_history, clear_history


# Page configuration
st.set_page_config(
    page_title="BAC OCR Studio",
    page_icon="",
    layout="wide",
)

# Custom CSS - Professional light theme with maroon accent using DM Sans font
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&display=swap');

    /* Theme:
       Maroon:       #7B1F3A
       Maroon Light: #9B3A56
       Gold:         #C8A44E
       Page BG:      #F5F1F2
       Card BG:      #FFFFFF
       Alt BG:       #F8F6F7
       Border:       #E8E2E4
       Border Light: #F0ECED
       Text:         #2D2D2D
       Text Muted:   #6B6B7B
       Text Dim:     #9B9BAB
       Success:      #16A34A
       Danger:       #DC2626
       Warning:      #D97706
       Info:         #2563EB
    */

    * {
        font-family: 'DM Sans', sans-serif !important;
    }

    .stApp {
        background: #F5F1F2;
        min-height: 100vh;
    }

    #MainMenu, footer, header {visibility: hidden;}

    .block-container {
        padding: 0 2rem 2rem 2rem;
        max-width: 1400px;
    }

    /* Top Bar */
    .top-bar {
        background: #7B1F3A;
        margin: -1rem -2rem 0 -2rem;
        padding: 0 24px;
        height: 46px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        color: #FFFFFF;
        margin-bottom: 0;
    }

    .top-bar-brand {
        display: flex;
        align-items: center;
        gap: 10px;
    }

    .top-bar-icon {
        width: 28px;
        height: 28px;
        background: rgba(255,255,255,0.15);
        border-radius: 6px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 12px;
        font-weight: 700;
    }

    .top-bar-title {
        font-size: 13px;
        font-weight: 700;
        letter-spacing: 0.5px;
    }

    .top-bar-sub {
        font-size: 10px;
        font-weight: 500;
        opacity: 0.7;
        letter-spacing: 0.8px;
        text-transform: uppercase;
    }

    /* Page Header */
    .page-header {
        background: #FFFFFF;
        margin: 0 -2rem;
        padding: 14px 24px;
        border-bottom: 1px solid #E8E2E4;
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 20px;
    }

    .page-title {
        font-size: 17px;
        font-weight: 800;
        color: #2D2D2D;
        letter-spacing: -0.3px;
        margin: 0;
    }

    .page-subtitle {
        font-size: 11px;
        color: #9B9BAB;
        margin-top: 2px;
    }

    /* Cards */
    .premium-card {
        background: #FFFFFF;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 2px 20px rgba(0, 0, 0, 0.06);
        border: 1px solid #E8E2E4;
    }

    /* File Uploader */
    [data-testid="stFileUploader"] {
        background: transparent !important;
        border: none !important;
        padding: 0 !important;
    }

    [data-testid="stFileUploader"] section {
        background: #F8F6F7 !important;
        border: 2px dashed #E8E2E4 !important;
        border-radius: 10px !important;
        padding: 2rem !important;
        transition: all 0.2s ease !important;
    }

    [data-testid="stFileUploader"] section:hover {
        border-color: #7B1F3A !important;
        background: rgba(123, 31, 58, 0.04) !important;
    }

    [data-testid="stFileUploader"] > div > div:last-child {
        display: none !important;
    }

    [data-testid="stFileUploader"] p,
    [data-testid="stFileUploader"] span,
    [data-testid="stFileUploader"] small {
        color: #6B6B7B !important;
        font-family: 'DM Sans', sans-serif !important;
    }

    [data-testid="stFileUploader"] small {
        display: none !important;
    }

    [data-testid="stFileUploader"] svg {
        color: #7B1F3A !important;
        width: 36px !important;
        height: 36px !important;
    }

    [data-testid="stFileUploader"] button {
        background: #7B1F3A !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 6px !important;
        padding: 0.5rem 1.25rem !important;
        font-weight: 600 !important;
        font-family: 'DM Sans', sans-serif !important;
        font-size: 12px !important;
        transition: all 0.2s ease !important;
    }

    [data-testid="stFileUploader"] button:hover {
        background: #9B3A56 !important;
        transform: none !important;
        box-shadow: none !important;
    }

    [data-testid="stFileUploader"] button[kind="icon"],
    [data-testid="stFileUploader"] [data-testid="baseButton-secondary"] {
        background: transparent !important;
        padding: 0.3rem !important;
        width: 28px !important;
        height: 28px !important;
        min-width: 28px !important;
        border-radius: 6px !important;
        box-shadow: none !important;
    }

    [data-testid="stFileUploader"] button[kind="icon"] svg,
    [data-testid="stFileUploader"] [data-testid="baseButton-secondary"] svg {
        width: 14px !important;
        height: 14px !important;
        color: #6B6B7B !important;
    }

    [data-testid="stFileUploader"] button[kind="icon"]:hover,
    [data-testid="stFileUploader"] [data-testid="baseButton-secondary"]:hover {
        background: #F0ECED !important;
        transform: none !important;
    }

    /* Image container */
    [data-testid="stImage"] {
        border-radius: 8px;
        overflow: hidden;
        border: 1px solid #E8E2E4;
    }

    /* Buttons - must also target inner span/p/div */
    .stButton > button,
    .stButton > button span,
    .stButton > button p,
    .stButton > button div {
        background: #7B1F3A !important;
        color: #FFFFFF !important;
        border: none !important;
    }

    .stButton > button {
        border-radius: 6px;
        padding: 10px 20px;
        font-weight: 600;
        font-size: 12px;
        letter-spacing: 0.3px;
        width: 100%;
        transition: all 0.2s ease;
        font-family: 'DM Sans', sans-serif !important;
    }

    .stButton > button:hover,
    .stButton > button:hover span {
        background: #9B3A56 !important;
        color: #FFFFFF !important;
    }

    .stButton > button:active,
    .stButton > button:active span {
        background: #5A1529 !important;
    }

    /* Dropdown popup */
    [data-baseweb="popover"],
    [data-baseweb="menu"],
    ul[role="listbox"] {
        background: #FFFFFF !important;
        border: 1px solid #E8E2E4 !important;
        border-radius: 8px !important;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1) !important;
    }

    [data-baseweb="menu"] li,
    ul[role="listbox"] li {
        color: #2D2D2D !important;
        background: #FFFFFF !important;
    }

    [data-baseweb="menu"] li:hover,
    ul[role="listbox"] li:hover {
        background: rgba(123,31,58,0.06) !important;
        color: #7B1F3A !important;
    }

    /* Tab inner text - inherit from parent */
    .stTabs [data-baseweb="tab"] span,
    .stTabs [data-baseweb="tab"] div {
        color: inherit !important;
    }

    /* Hide toolbar artifacts */
    [data-testid="stToolbar"],
    [data-testid="stDecoration"],
    [data-testid="stStatusWidget"] {
        display: none !important;
    }

    /* Sidebar collapse/expand buttons */
    [data-testid="collapsedControl"] {
        background: #FFFFFF !important;
        border: 1px solid #E8E2E4 !important;
    }

    [data-testid="collapsedControl"] button {
        background: transparent !important;
        color: #7B1F3A !important;
    }

    /* File uploader buttons */
    [data-testid="stFileUploader"] button,
    [data-testid="stFileUploader"] button span {
        background: #7B1F3A !important;
        color: #FFFFFF !important;
    }

    [data-testid="stFileUploader"] button:hover,
    [data-testid="stFileUploader"] button:hover span {
        background: #9B3A56 !important;
        color: #FFFFFF !important;
    }

    /* Progress steps */
    .step-container {
        display: flex;
        justify-content: center;
        align-items: center;
        gap: 1rem;
        margin: 16px 0;
        padding: 12px 16px;
        background: #FFFFFF;
        border-radius: 10px;
        border: 1px solid #E8E2E4;
    }

    .step {
        display: flex;
        align-items: center;
        gap: 8px;
        color: #9B9BAB;
        font-size: 12px;
        font-weight: 600;
        padding: 6px 14px;
        border-radius: 6px;
        transition: all 0.2s ease;
    }

    .step.active {
        color: #7B1F3A;
        background: rgba(123, 31, 58, 0.08);
    }

    .step.completed {
        color: #16A34A;
    }

    .step-line {
        width: 32px;
        height: 2px;
        background: #E8E2E4;
        border-radius: 1px;
    }

    .step.completed + .step-line,
    .step-line.completed {
        background: #16A34A;
    }

    .step-number {
        width: 24px;
        height: 24px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 11px;
        font-weight: 700;
        background: #F8F6F7;
        border: 2px solid #E8E2E4;
        color: #9B9BAB;
    }

    .step.active .step-number {
        background: #7B1F3A;
        border-color: #7B1F3A;
        color: #FFFFFF;
    }

    .step.completed .step-number {
        background: #FFFFFF;
        border-color: #16A34A;
        color: #16A34A;
    }

    /* Output section */
    .output-container {
        background: #FFFFFF;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 2px 20px rgba(0, 0, 0, 0.06);
        border: 1px solid #E8E2E4;
    }

    .output-label {
        color: #7B1F3A;
        font-weight: 700;
        font-size: 10px;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        margin-bottom: 10px;
    }

    /* Text area */
    .stTextArea textarea {
        border: 1px solid #E8E2E4;
        border-radius: 8px;
        font-family: 'DM Sans', sans-serif !important;
        font-size: 12px;
        color: #2D2D2D;
        background: #F8F6F7;
        padding: 12px 14px;
        line-height: 1.6;
    }

    .stTextArea textarea:focus {
        border-color: #7B1F3A;
        box-shadow: 0 0 0 3px rgba(123, 31, 58, 0.1);
    }

    /* Info message */
    .stAlert {
        border-radius: 8px;
        border: none;
        background: #F8F6F7;
    }

    /* Processing status */
    .processing-status {
        text-align: center;
        padding: 20px;
        color: #2D2D2D;
        background: #FFFFFF;
        border-radius: 10px;
        border: 1px solid #E8E2E4;
    }

    .processing-step {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 8px;
        margin: 8px 0;
        font-size: 12px;
        font-weight: 500;
    }

    .processing-step.done { color: #16A34A; }
    .processing-step.active { color: #7B1F3A; font-weight: 600; }
    .processing-step.pending { color: #9B9BAB; }

    /* Upload prompt */
    .upload-prompt {
        text-align: center;
        padding: 2.5rem 1rem;
        color: #9B9BAB;
    }

    .upload-prompt p {
        margin: 4px 0;
        font-weight: 500;
    }

    /* Confidence display */
    .confidence-item {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 10px 14px;
        margin: 4px 0;
        background: #F8F6F7;
        border-radius: 8px;
        font-size: 13px;
        border: 1px solid #F0ECED;
        transition: all 0.2s ease;
    }

    .confidence-item:hover {
        border-color: #E8E2E4;
    }

    .confidence-text {
        color: #2D2D2D;
        flex: 1;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
        margin-right: 10px;
        font-weight: 500;
    }

    .confidence-badges {
        display: flex;
        align-items: center;
        gap: 6px;
    }

    .confidence-score {
        font-weight: 700;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 10px;
        letter-spacing: 0.4px;
    }

    .approval-badge {
        font-weight: 700;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 10px;
        letter-spacing: 0.4px;
        text-transform: uppercase;
    }

    .approved {
        background: rgba(22, 163, 74, 0.1);
        color: #16A34A;
    }

    .not-approved {
        background: rgba(220, 38, 38, 0.1);
        color: #DC2626;
    }

    .confidence-high {
        background: rgba(22, 163, 74, 0.1);
        color: #16A34A;
    }
    .confidence-medium {
        background: rgba(217, 119, 6, 0.1);
        color: #D97706;
    }
    .confidence-low {
        background: rgba(220, 38, 38, 0.1);
        color: #DC2626;
    }

    .avg-confidence {
        text-align: center;
        padding: 18px 20px;
        background: #FFFFFF;
        border-radius: 10px;
        margin: 14px 0;
        border: 1px solid #E8E2E4;
    }

    .avg-score {
        font-size: 2.5rem;
        font-weight: 800;
        color: #7B1F3A;
        letter-spacing: -0.5px;
    }

    .avg-label {
        font-size: 10px;
        color: #9B9BAB;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 700;
        margin-top: 2px;
    }

    /* History sidebar */
    .history-title {
        color: #2D2D2D;
        font-size: 10px;
        font-weight: 700;
        margin-bottom: 14px;
        padding-bottom: 10px;
        border-bottom: 1px solid #E8E2E4;
        text-transform: uppercase;
        letter-spacing: 0.8px;
    }

    .history-item {
        background: #F8F6F7;
        border-radius: 8px;
        padding: 10px 12px;
        margin-bottom: 8px;
        cursor: pointer;
        transition: all 0.2s ease;
        border: 1px solid #F0ECED;
    }

    .history-item:hover {
        border-color: #E8E2E4;
    }

    .history-time {
        font-size: 10px;
        color: #7B1F3A;
        margin-bottom: 4px;
        font-weight: 600;
        letter-spacing: 0.3px;
    }

    .history-preview {
        font-size: 12px;
        color: #6B6B7B;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
        font-weight: 400;
    }

    .history-confidence {
        font-size: 10px;
        color: #2D2D2D;
        margin-top: 4px;
        font-weight: 600;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: #FFFFFF;
        border-right: 1px solid #E8E2E4;
    }

    [data-testid="stSidebar"] [data-testid="stImage"] {
        border-radius: 8px;
        border: 1px solid #E8E2E4;
    }

    [data-testid="stSidebar"] .stButton > button,
    [data-testid="stSidebar"] .stButton > button span {
        background: #F8F6F7 !important;
        color: #7B1F3A !important;
        border: 1px solid #E8E2E4 !important;
        font-size: 11px;
        padding: 8px 14px;
    }

    [data-testid="stSidebar"] .stButton > button:hover,
    [data-testid="stSidebar"] .stButton > button:hover span {
        background: rgba(123, 31, 58, 0.06) !important;
        color: #7B1F3A !important;
    }

    [data-testid="stSidebar"] hr {
        border-color: #E8E2E4;
        margin: 8px 0;
    }

    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 6px;
        height: 6px;
    }

    ::-webkit-scrollbar-track {
        background: #F5F1F2;
        border-radius: 3px;
    }

    ::-webkit-scrollbar-thumb {
        background: #E8E2E4;
        border-radius: 3px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: #7B1F3A;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background: #FFFFFF;
        border-radius: 10px;
        padding: 4px;
        border: 1px solid #E8E2E4;
    }

    .stTabs [data-baseweb="tab"] {
        height: 40px;
        background: transparent;
        border-radius: 6px;
        color: #6B6B7B;
        font-weight: 600;
        font-size: 12px;
        padding: 0 20px;
        border: none;
        font-family: 'DM Sans', sans-serif !important;
    }

    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(123, 31, 58, 0.06);
        color: #7B1F3A;
    }

    .stTabs [aria-selected="true"] {
        background: #7B1F3A !important;
        color: #FFFFFF !important;
        font-weight: 700;
    }

    .stTabs [data-baseweb="tab-highlight"] {
        display: none;
    }

    .stTabs [data-baseweb="tab-border"] {
        display: none;
    }

    /* Hallmark info card */
    .hallmark-card {
        background: #FFFFFF;
        border-radius: 10px;
        padding: 16px 18px;
        margin: 10px 0;
        border: 1px solid #E8E2E4;
        border-left: 3px solid #7B1F3A;
    }

    .hallmark-title {
        color: #7B1F3A;
        font-size: 10px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        margin-bottom: 10px;
    }

    .hallmark-item {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 8px 0;
        border-bottom: 1px solid #F0ECED;
        font-size: 12px;
    }

    .hallmark-item:last-child {
        border-bottom: none;
    }

    .hallmark-label {
        color: #6B6B7B;
        font-weight: 500;
    }

    .hallmark-value {
        color: #7B1F3A;
        font-weight: 700;
    }

    .bis-badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 10px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.4px;
    }

    .bis-certified {
        background: rgba(22, 163, 74, 0.1);
        color: #16A34A;
    }

    .bis-not-certified {
        background: rgba(217, 119, 6, 0.1);
        color: #D97706;
    }

    /* Type badges */
    .type-badge {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 10px;
        font-size: 10px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.4px;
        margin-left: 6px;
    }

    .type-purity {
        background: rgba(123, 31, 58, 0.1);
        color: #7B1F3A;
    }

    .type-huid {
        background: rgba(22, 163, 74, 0.1);
        color: #16A34A;
    }

    .type-check {
        background: rgba(37, 99, 235, 0.1);
        color: #2563EB;
    }

    .type-unknown {
        background: #F8F6F7;
        color: #9B9BAB;
    }

    /* Check info card */
    .check-card {
        background: #FFFFFF;
        border-radius: 10px;
        padding: 16px 18px;
        margin: 10px 0;
        border: 1px solid #E8E2E4;
        border-left: 3px solid #2563EB;
    }

    .check-title {
        color: #2563EB;
        font-size: 10px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        margin-bottom: 10px;
    }

    .check-item {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 8px 0;
        border-bottom: 1px solid #F0ECED;
        font-size: 12px;
    }

    .check-item:last-child {
        border-bottom: none;
    }

    .check-label {
        color: #6B6B7B;
        font-weight: 500;
    }

    .check-value {
        color: #2563EB;
        font-weight: 700;
        font-size: 12px;
        font-family: 'DM Sans', monospace;
    }

    .check-badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 10px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.4px;
    }

    .check-valid {
        background: rgba(22, 163, 74, 0.1);
        color: #16A34A;
    }

    .check-invalid {
        background: rgba(217, 119, 6, 0.1);
        color: #D97706;
    }

    .validated-badge {
        background: rgba(22, 163, 74, 0.1);
        color: #16A34A;
        padding: 2px 8px;
        border-radius: 10px;
        font-size: 10px;
        font-weight: 700;
        margin-left: 6px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_ocr_engine():
    """Load OCR engine V1 (cached for performance)."""
    return OCREngine()


@st.cache_resource
def load_ocr_engine_v2():
    """Load OCR engine V2 with hallmark detection (cached for performance)."""
    return OCREngineV2(enable_preprocessing=True)


def render_steps(current_step):
    """Render progress steps."""
    steps = ["Upload", "Process", "Result"]
    html = '<div class="step-container">'
    for i, step in enumerate(steps):
        if i < current_step:
            status = "completed"
        elif i == current_step:
            status = "active"
        else:
            status = ""

        check = "&#10003;" if i < current_step else str(i + 1)
        html += f'<div class="step {status}"><span class="step-number">{check}</span>{step}</div>'
        if i < len(steps) - 1:
            line_status = "completed" if i < current_step else ""
            html += f'<div class="step-line {line_status}"></div>'

    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)


def get_confidence_class(score: float) -> str:
    """Get CSS class based on confidence score."""
    if score >= 0.9:
        return "confidence-high"
    elif score >= 0.7:
        return "confidence-medium"
    return "confidence-low"


def render_confidence_results(results: list):
    """Render OCR results with confidence scores (V1)."""
    if not results:
        st.info("No text detected in this image.")
        return

    avg_conf = sum(r.confidence for r in results) / len(results)

    st.markdown(f"""
    <div class="avg-confidence">
        <div class="avg-score">{avg_conf * 100:.1f}%</div>
        <div class="avg-label">Confidence Score</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<p class="output-label">Detected Text</p>', unsafe_allow_html=True)

    for r in results:
        conf_class = get_confidence_class(r.confidence)
        is_approved = r.confidence >= 0.75
        approval_class = "approved" if is_approved else "not-approved"
        approval_text = "Approved" if is_approved else "Not Approved"
        st.markdown(f"""
        <div class="confidence-item">
            <span class="confidence-text">{r.text}</span>
            <div class="confidence-badges">
                <span class="confidence-score {conf_class}">{r.confidence * 100:.0f}%</span>
                <span class="approval-badge {approval_class}">{approval_text}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)


def render_check_info(check_info):
    """Render check/cheque information card."""
    if not check_info:
        return

    check_class = "check-valid" if check_info.is_valid_check else "check-invalid"
    check_text = "Valid Check" if check_info.is_valid_check else "Incomplete"

    items_html = ""

    if check_info.bank_name:
        items_html += f"""
        <div class="check-item">
            <span class="check-label">Bank Name</span>
            <span class="check-value">{check_info.bank_name}</span>
        </div>
        """

    if check_info.ifsc_code:
        items_html += f"""
        <div class="check-item">
            <span class="check-label">IFSC Code</span>
            <span class="check-value">{check_info.ifsc_code}</span>
        </div>
        """

    if check_info.micr_code:
        items_html += f"""
        <div class="check-item">
            <span class="check-label">MICR Code</span>
            <span class="check-value">{check_info.micr_code}</span>
        </div>
        """

    if check_info.account_number:
        items_html += f"""
        <div class="check-item">
            <span class="check-label">Account Number</span>
            <span class="check-value">{check_info.account_number}</span>
        </div>
        """

    if check_info.check_number:
        items_html += f"""
        <div class="check-item">
            <span class="check-label">Check Number</span>
            <span class="check-value">{check_info.check_number}</span>
        </div>
        """

    if items_html:
        st.markdown(f"""
        <div class="check-card">
            <div class="check-title">
                Check Information
                <span class="check-badge {check_class}">{check_text}</span>
            </div>
            {items_html}
        </div>
        """, unsafe_allow_html=True)


def render_hallmark_info(hallmark_info):
    """Render hallmark-specific information card."""
    bis_class = "bis-certified" if hallmark_info.bis_certified else "bis-not-certified"
    bis_text = "BIS Certified" if hallmark_info.bis_certified else "Not Certified"

    items_html = ""

    if hallmark_info.purity_code:
        items_html += f"""
        <div class="hallmark-item">
            <span class="hallmark-label">Purity Code</span>
            <span class="hallmark-value">{hallmark_info.purity_code}</span>
        </div>
        """

    if hallmark_info.karat:
        items_html += f"""
        <div class="hallmark-item">
            <span class="hallmark-label">Karat</span>
            <span class="hallmark-value">{hallmark_info.karat}</span>
        </div>
        """

    if hallmark_info.purity_percentage:
        items_html += f"""
        <div class="hallmark-item">
            <span class="hallmark-label">Purity</span>
            <span class="hallmark-value">{hallmark_info.purity_percentage}%</span>
        </div>
        """

    if hallmark_info.huid:
        items_html += f"""
        <div class="hallmark-item">
            <span class="hallmark-label">HUID</span>
            <span class="hallmark-value">{hallmark_info.huid}</span>
        </div>
        """

    if items_html:
        st.markdown(f"""
        <div class="hallmark-card">
            <div class="hallmark-title">
                Hallmark Information
                <span class="bis-badge {bis_class}">{bis_text}</span>
            </div>
            {items_html}
        </div>
        """, unsafe_allow_html=True)
    elif not hallmark_info.check_info:
        st.markdown(f"""
        <div class="hallmark-card">
            <div class="hallmark-title">
                Hallmark Information
                <span class="bis-badge {bis_class}">{bis_text}</span>
            </div>
            <div class="hallmark-item">
                <span class="hallmark-label" style="color: #9B9BAB;">No hallmark data detected</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    if hallmark_info.check_info:
        render_check_info(hallmark_info.check_info)


def render_confidence_results_v2(results: list, hallmark_info):
    """Render OCR results with confidence scores and hallmark info (V2)."""
    if not results:
        st.info("No text detected in this image.")
        return

    avg_conf = sum(r.confidence for r in results) / len(results)

    st.markdown(f"""
    <div class="avg-confidence">
        <div class="avg-score">{avg_conf * 100:.1f}%</div>
        <div class="avg-label">Confidence Score</div>
    </div>
    """, unsafe_allow_html=True)

    render_hallmark_info(hallmark_info)

    st.markdown('<p class="output-label">Detected Text</p>', unsafe_allow_html=True)

    for r in results:
        conf_class = get_confidence_class(r.confidence)
        is_approved = r.confidence >= 0.75
        approval_class = "approved" if is_approved else "not-approved"
        approval_text = "Approved" if is_approved else "Not Approved"

        type_class = "type-unknown"
        type_text = ""
        if r.hallmark_type == HallmarkType.PURITY_MARK:
            type_class = "type-purity"
            type_text = "Purity"
        elif r.hallmark_type == HallmarkType.HUID:
            type_class = "type-huid"
            type_text = "HUID"
        elif r.hallmark_type == HallmarkType.CHECK:
            type_class = "type-check"
            type_text = "Check"

        type_badge = f'<span class="type-badge {type_class}">{type_text}</span>' if type_text else ""
        validated_badge = '<span class="validated-badge">Validated</span>' if r.validated else ""

        st.markdown(f"""
        <div class="confidence-item">
            <span class="confidence-text">{r.text}{type_badge}{validated_badge}</span>
            <div class="confidence-badges">
                <span class="confidence-score {conf_class}">{r.confidence * 100:.0f}%</span>
                <span class="approval-badge {approval_class}">{approval_text}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)


def render_history_sidebar():
    """Render history in sidebar."""
    with st.sidebar:
        st.markdown('<p class="history-title">Recent Scans</p>', unsafe_allow_html=True)

        history = get_history()

        if not history:
            st.markdown("""
            <p style="color: #9B9BAB; font-size: 12px; text-align: center; font-weight: 500;">
                No history yet
            </p>
            """, unsafe_allow_html=True)
            return

        if st.button("Clear All", key="clear_history"):
            clear_history()
            st.rerun()

        for entry in history:
            with st.container():
                col1, col2 = st.columns([1, 2])
                with col1:
                    try:
                        thumb = Image.open(entry["thumbnail_path"])
                        st.image(thumb, use_container_width=True)
                    except Exception:
                        st.markdown("--")

                with col2:
                    preview = entry["text"][:40] + "..." if len(entry["text"]) > 40 else entry["text"]
                    st.markdown(f"""
                    <div class="history-time">{entry["timestamp"]}</div>
                    <div class="history-preview">{preview}</div>
                    <div class="history-confidence">{entry["avg_confidence"] * 100:.0f}% confidence</div>
                    """, unsafe_allow_html=True)

                st.markdown("---")


def render_v1_tab():
    """Render the V1 (Standard OCR) tab content."""
    if "ocr_results_v1" not in st.session_state:
        st.session_state.ocr_results_v1 = None
    if "processed_v1" not in st.session_state:
        st.session_state.processed_v1 = False
    if "current_image_v1" not in st.session_state:
        st.session_state.current_image_v1 = None
    if "processing_v1" not in st.session_state:
        st.session_state.processing_v1 = False

    col_main, col_results = st.columns([1, 1], gap="large")

    with col_main:
        uploaded_file = st.file_uploader(
            "Drop an image here or click to upload",
            type=["png", "jpg", "jpeg", "bmp", "webp"],
            help="Supported: PNG, JPG, JPEG, BMP, WEBP",
            label_visibility="collapsed",
            key="uploader_v1"
        )

        if uploaded_file:
            if st.session_state.processed_v1 and st.session_state.ocr_results_v1 is not None:
                current_step = 2
            else:
                current_step = 1

            render_steps(current_step)

            image = Image.open(uploaded_file)
            st.session_state.current_image_v1 = image.copy()
            st.image(image, use_container_width=True)

            st.markdown("<div style='height: 0.5rem'></div>", unsafe_allow_html=True)

            if st.button("Extract Text", use_container_width=True, key="extract_v1"):
                st.session_state.processing_v1 = True
                st.rerun()
        else:
            st.session_state.ocr_results_v1 = None
            st.session_state.processed_v1 = False
            st.session_state.current_image_v1 = None
            render_steps(0)

            st.markdown("""
            <div class="upload-prompt">
                <p style="font-size: 14px; font-weight: 600; color: #6B6B7B;">Drop your image here</p>
                <p style="font-size: 11px; color: #9B9BAB;">PNG, JPG, JPEG, BMP, or WEBP</p>
            </div>
            """, unsafe_allow_html=True)

    with col_results:
        if st.session_state.processing_v1 and st.session_state.current_image_v1 is not None:
            status_placeholder = st.empty()

            with status_placeholder.container():
                st.markdown("""
                <div class="processing-status">
                    <div class="processing-step done">&#10003; Image loaded</div>
                    <div class="processing-step active">&#9679; Analyzing...</div>
                    <div class="processing-step pending">&#9675; Extracting text</div>
                </div>
                """, unsafe_allow_html=True)

            ocr_engine = load_ocr_engine()

            status_placeholder.markdown("""
            <div class="processing-status">
                <div class="processing-step done">&#10003; Image loaded</div>
                <div class="processing-step done">&#10003; Analysis complete</div>
                <div class="processing-step active">&#9679; Extracting...</div>
            </div>
            """, unsafe_allow_html=True)

            results = ocr_engine.extract_text_with_confidence(st.session_state.current_image_v1)
            st.session_state.ocr_results_v1 = results
            st.session_state.processed_v1 = True
            st.session_state.processing_v1 = False

            if results:
                avg_conf = sum(r.confidence for r in results) / len(results)
                text = "\n".join([r.text for r in results])
                add_to_history(st.session_state.current_image_v1, text, avg_conf)

            status_placeholder.empty()
            st.rerun()

        elif st.session_state.processed_v1 and st.session_state.ocr_results_v1 is not None:
            st.markdown('<div class="output-container">', unsafe_allow_html=True)
            render_confidence_results(st.session_state.ocr_results_v1)

            if st.session_state.ocr_results_v1:
                combined_text = "\n".join([r.text for r in st.session_state.ocr_results_v1])
                st.markdown('<p class="output-label" style="margin-top: 14px;">Full Text</p>', unsafe_allow_html=True)
                st.text_area("Full text", combined_text, height=150, label_visibility="collapsed", key="text_v1")

            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="
                background: #FFFFFF;
                border: 2px dashed #E8E2E4;
                border-radius: 10px;
                padding: 3rem 2rem;
                text-align: center;
                color: #9B9BAB;
            ">
                <p style="font-size: 13px; margin-bottom: 4px; font-weight: 600;">Results will appear here</p>
                <p style="font-size: 11px;">Upload an image and click Extract</p>
            </div>
            """, unsafe_allow_html=True)


def render_v2_tab():
    """Render the V2 (Hallmark OCR) tab content."""
    if "ocr_results_v2" not in st.session_state:
        st.session_state.ocr_results_v2 = None
    if "hallmark_info_v2" not in st.session_state:
        st.session_state.hallmark_info_v2 = None
    if "processed_v2" not in st.session_state:
        st.session_state.processed_v2 = False
    if "current_image_v2" not in st.session_state:
        st.session_state.current_image_v2 = None
    if "processing_v2" not in st.session_state:
        st.session_state.processing_v2 = False

    col_main, col_results = st.columns([1, 1], gap="large")

    with col_main:
        uploaded_file = st.file_uploader(
            "Drop an image here or click to upload",
            type=["png", "jpg", "jpeg", "bmp", "webp"],
            help="Supported: PNG, JPG, JPEG, BMP, WEBP",
            label_visibility="collapsed",
            key="uploader_v2"
        )

        if uploaded_file:
            if st.session_state.processed_v2 and st.session_state.ocr_results_v2 is not None:
                current_step = 2
            else:
                current_step = 1

            render_steps(current_step)

            image = Image.open(uploaded_file)
            st.session_state.current_image_v2 = image.copy()
            st.image(image, use_container_width=True)

            st.markdown("<div style='height: 0.5rem'></div>", unsafe_allow_html=True)

            if st.button("Extract Hallmark", use_container_width=True, key="extract_v2"):
                st.session_state.processing_v2 = True
                st.rerun()
        else:
            st.session_state.ocr_results_v2 = None
            st.session_state.hallmark_info_v2 = None
            st.session_state.processed_v2 = False
            st.session_state.current_image_v2 = None
            render_steps(0)

            st.markdown("""
            <div class="upload-prompt">
                <p style="font-size: 14px; font-weight: 600; color: #6B6B7B;">Drop your hallmark image here</p>
                <p style="font-size: 11px; color: #9B9BAB;">PNG, JPG, JPEG, BMP, or WEBP</p>
            </div>
            """, unsafe_allow_html=True)

    with col_results:
        if st.session_state.processing_v2 and st.session_state.current_image_v2 is not None:
            status_placeholder = st.empty()

            with status_placeholder.container():
                st.markdown("""
                <div class="processing-status">
                    <div class="processing-step done">&#10003; Image loaded</div>
                    <div class="processing-step active">&#9679; Preprocessing...</div>
                    <div class="processing-step pending">&#9675; Detecting hallmarks</div>
                    <div class="processing-step pending">&#9675; Validating</div>
                </div>
                """, unsafe_allow_html=True)

            ocr_engine_v2 = load_ocr_engine_v2()

            status_placeholder.markdown("""
            <div class="processing-status">
                <div class="processing-step done">&#10003; Image loaded</div>
                <div class="processing-step done">&#10003; Preprocessed</div>
                <div class="processing-step active">&#9679; Detecting...</div>
                <div class="processing-step pending">&#9675; Validating</div>
            </div>
            """, unsafe_allow_html=True)

            hallmark_info = ocr_engine_v2.extract_with_hallmark_info(st.session_state.current_image_v2)
            st.session_state.ocr_results_v2 = hallmark_info.all_results
            st.session_state.hallmark_info_v2 = hallmark_info
            st.session_state.processed_v2 = True
            st.session_state.processing_v2 = False

            if hallmark_info.all_results:
                text = "\n".join([r.text for r in hallmark_info.all_results])
                add_to_history(st.session_state.current_image_v2, text, hallmark_info.overall_confidence)

            status_placeholder.empty()
            st.rerun()

        elif st.session_state.processed_v2 and st.session_state.ocr_results_v2 is not None:
            st.markdown('<div class="output-container">', unsafe_allow_html=True)
            render_confidence_results_v2(st.session_state.ocr_results_v2, st.session_state.hallmark_info_v2)

            if st.session_state.ocr_results_v2:
                combined_text = "\n".join([r.text for r in st.session_state.ocr_results_v2])
                st.markdown('<p class="output-label" style="margin-top: 14px;">Full Text</p>', unsafe_allow_html=True)
                st.text_area("Full text", combined_text, height=150, label_visibility="collapsed", key="text_v2")

            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="
                background: #FFFFFF;
                border: 2px dashed #E8E2E4;
                border-radius: 10px;
                padding: 3rem 2rem;
                text-align: center;
                color: #9B9BAB;
            ">
                <p style="font-size: 13px; margin-bottom: 4px; font-weight: 600;">Results will appear here</p>
                <p style="font-size: 11px;">Upload a hallmark image and click Extract</p>
            </div>
            """, unsafe_allow_html=True)


def main():
    render_history_sidebar()

    # Top bar
    st.markdown("""
    <div class="top-bar">
        <div class="top-bar-brand">
            <div class="top-bar-icon">B</div>
            <div>
                <div class="top-bar-title">BAC OCR Studio</div>
                <div class="top-bar-sub">Optical Character Recognition</div>
            </div>
        </div>
        <div style="font-size: 11px; font-weight: 500; opacity: 0.8; letter-spacing: 0.5px;">Text Extraction</div>
    </div>
    """, unsafe_allow_html=True)

    # Page header
    st.markdown("""
    <div class="page-header">
        <div>
            <h1 class="page-title">OCR Studio</h1>
            <div class="page-subtitle">Upload images to extract and analyze text</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Tabs for V1 and V2
    tab1, tab2 = st.tabs(["Standard OCR", "Hallmark OCR V2"])

    with tab1:
        render_v1_tab()

    with tab2:
        render_v2_tab()


if __name__ == "__main__":
    main()
