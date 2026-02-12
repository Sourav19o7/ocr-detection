"""
Streamlit Dashboard for OCR Text Detection.
Premium minimalist design with dark theme using Lexend font.
Supports V1 (standard) and V2 (hallmark-specific) OCR.
"""

import streamlit as st
from PIL import Image
from ocr_model import OCREngine
from ocr_model_v2 import OCREngineV2, HallmarkType
from history import add_to_history, get_history, clear_history


# Page configuration
st.set_page_config(
    page_title="OCR Studio",
    page_icon="",
    layout="wide",
)

# Custom CSS - Premium dark theme with gold accent using Lexend font
st.markdown("""
<style>
    /* Import Lexend font from Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Lexend:wght@300;400;500;600;700&display=swap');

    /* Color palette:
       Black: #000000
       Navy: #14213d
       Gold/Orange: #fca311
       Light Gray: #e5e5e5
       White: #ffffff
    */

    * {
        font-family: 'Lexend', sans-serif !important;
    }

    .stApp {
        background: linear-gradient(160deg, #000000 0%, #14213d 100%);
        min-height: 100vh;
    }

    /* Hide default Streamlit elements */
    #MainMenu, footer, header {visibility: hidden;}

    /* Main container */
    .block-container {
        padding: 3rem 2rem;
        max-width: 1400px;
    }

    /* Premium Header */
    .main-header {
        color: #ffffff;
        font-size: 2.8rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 0.5rem;
        letter-spacing: -1px;
    }

    .sub-header {
        color: #fca311;
        font-size: 1rem;
        text-align: center;
        margin-bottom: 3rem;
        font-weight: 300;
        letter-spacing: 3px;
        text-transform: uppercase;
    }

    /* Premium Card Style */
    .premium-card {
        background: linear-gradient(145deg, #1a2744 0%, #14213d 100%);
        border-radius: 24px;
        padding: 2rem;
        box-shadow:
            0 4px 6px rgba(0, 0, 0, 0.2),
            0 20px 40px rgba(0, 0, 0, 0.4);
        border: 1px solid rgba(252, 163, 17, 0.2);
    }

    /* Upload area - Clean minimal style */
    [data-testid="stFileUploader"] {
        background: transparent !important;
        border: none !important;
        padding: 0 !important;
    }

    /* Dropzone styling */
    [data-testid="stFileUploader"] section {
        background: rgba(20, 33, 61, 0.6) !important;
        border: 2px dashed rgba(252, 163, 17, 0.5) !important;
        border-radius: 16px !important;
        padding: 2.5rem 2rem !important;
        transition: all 0.3s ease !important;
    }

    [data-testid="stFileUploader"] section:hover {
        border-color: #fca311 !important;
        background: rgba(20, 33, 61, 0.8) !important;
    }

    /* Hide the uploaded file info row - we show image separately */
    [data-testid="stFileUploader"] > div > div:last-child {
        display: none !important;
    }

    /* Text styling */
    [data-testid="stFileUploader"] p,
    [data-testid="stFileUploader"] span,
    [data-testid="stFileUploader"] small {
        color: #e5e5e5 !important;
        font-family: 'Lexend', sans-serif !important;
    }

    /* Hide the file size limit text */
    [data-testid="stFileUploader"] small {
        display: none !important;
    }

    /* Icon styling */
    [data-testid="stFileUploader"] svg {
        color: #e5e5e5 !important;
        width: 40px !important;
        height: 40px !important;
    }

    /* Browse files button */
    [data-testid="stFileUploader"] button {
        background: linear-gradient(135deg, #fca311 0%, #e09000 100%) !important;
        color: #000000 !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.6rem 1.5rem !important;
        font-weight: 600 !important;
        font-family: 'Lexend', sans-serif !important;
        transition: all 0.3s ease !important;
    }

    [data-testid="stFileUploader"] button:hover {
        background: linear-gradient(135deg, #ffb733 0%, #fca311 100%) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 12px rgba(252, 163, 17, 0.4) !important;
    }

    /* Delete/X button for uploaded file */
    [data-testid="stFileUploader"] button[kind="icon"],
    [data-testid="stFileUploader"] button:has(svg[data-testid="stBaseButton-headerNoPadding"]),
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
        color: #14213d !important;
    }

    [data-testid="stFileUploader"] button[kind="icon"]:hover,
    [data-testid="stFileUploader"] [data-testid="baseButton-secondary"]:hover {
        background: rgba(20, 33, 61, 0.3) !important;
        transform: none !important;
    }

    /* Image container */
    [data-testid="stImage"] {
        border-radius: 16px;
        overflow: hidden;
        box-shadow:
            0 4px 6px rgba(0, 0, 0, 0.15),
            0 12px 28px rgba(0, 0, 0, 0.25);
    }

    /* Premium Button */
    .stButton > button {
        background: linear-gradient(135deg, #fca311 0%, #e09000 100%);
        color: #000000;
        border: none;
        border-radius: 14px;
        padding: 1rem 2.5rem;
        font-weight: 600;
        font-size: 0.95rem;
        letter-spacing: 0.5px;
        width: 100%;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow:
            0 4px 14px rgba(252, 163, 17, 0.4),
            inset 0 1px 0 rgba(255, 255, 255, 0.3);
        font-family: 'Lexend', sans-serif !important;
    }

    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow:
            0 8px 24px rgba(252, 163, 17, 0.5),
            inset 0 1px 0 rgba(255, 255, 255, 0.4);
        background: linear-gradient(135deg, #ffb733 0%, #fca311 100%);
    }

    .stButton > button:active {
        transform: translateY(-1px);
    }

    /* Progress steps */
    .step-container {
        display: flex;
        justify-content: center;
        align-items: center;
        gap: 1rem;
        margin: 2rem 0;
        padding: 1rem;
        background: rgba(255, 255, 255, 0.03);
        border-radius: 16px;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.05);
    }

    .step {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        color: rgba(255, 255, 255, 0.3);
        font-size: 0.85rem;
        font-weight: 500;
        padding: 0.5rem 1rem;
        border-radius: 10px;
        transition: all 0.3s ease;
    }

    .step.active {
        color: #ffffff;
        background: rgba(252, 163, 17, 0.15);
    }

    .step.completed {
        color: #fca311;
    }

    .step-line {
        width: 40px;
        height: 2px;
        background: rgba(255, 255, 255, 0.1);
        border-radius: 1px;
    }

    .step.completed + .step-line,
    .step-line.completed {
        background: #fca311;
    }

    .step-number {
        width: 28px;
        height: 28px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.75rem;
        font-weight: 600;
        background: rgba(255, 255, 255, 0.05);
        border: 2px solid rgba(255, 255, 255, 0.15);
    }

    .step.active .step-number {
        background: #fca311;
        border-color: #fca311;
        color: #000000;
        box-shadow: 0 0 20px rgba(252, 163, 17, 0.4);
    }

    .step.completed .step-number {
        background: #14213d;
        border-color: #fca311;
        color: #fca311;
    }

    /* Output section */
    .output-container {
        background: linear-gradient(145deg, #1a2744 0%, #14213d 100%);
        border-radius: 24px;
        padding: 2rem;
        box-shadow:
            0 4px 6px rgba(0, 0, 0, 0.2),
            0 20px 40px rgba(0, 0, 0, 0.4);
        border: 1px solid rgba(252, 163, 17, 0.2);
    }

    .output-label {
        color: #fca311;
        font-weight: 600;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-bottom: 1rem;
    }

    /* Text area styling */
    .stTextArea textarea {
        border: 1px solid rgba(252, 163, 17, 0.3);
        border-radius: 14px;
        font-family: 'Lexend', sans-serif !important;
        font-size: 0.9rem;
        color: #e5e5e5;
        background: #0d1521;
        padding: 1rem;
    }

    .stTextArea textarea:focus {
        border-color: #fca311;
        box-shadow: 0 0 0 4px rgba(252, 163, 17, 0.15);
    }

    /* Info message */
    .stAlert {
        border-radius: 14px;
        border: none;
        background: #1a2744;
    }

    /* Processing status */
    .processing-status {
        text-align: center;
        padding: 2rem;
        color: #ffffff;
        background: rgba(255, 255, 255, 0.03);
        border-radius: 16px;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.05);
    }

    .processing-step {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 0.75rem;
        margin: 0.75rem 0;
        font-size: 0.9rem;
        font-weight: 400;
    }

    .processing-step.done { color: #fca311; }
    .processing-step.active { color: #ffffff; }
    .processing-step.pending { color: rgba(255, 255, 255, 0.2); }

    /* Upload prompt text */
    .upload-prompt {
        text-align: center;
        padding: 3rem 1rem;
        color: rgba(255, 255, 255, 0.5);
    }

    .upload-prompt p {
        margin: 0.5rem 0;
        font-weight: 300;
    }

    /* Confidence display */
    .confidence-item {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.875rem 1rem;
        margin: 0.5rem 0;
        background: #0d1521;
        border-radius: 12px;
        font-size: 0.9rem;
        border: 1px solid rgba(252, 163, 17, 0.2);
        transition: all 0.2s ease;
    }

    .confidence-item:hover {
        box-shadow: 0 4px 12px rgba(252, 163, 17, 0.15);
        transform: translateX(4px);
        border-color: #fca311;
    }

    .confidence-text {
        color: #e5e5e5;
        flex: 1;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
        margin-right: 1rem;
        font-weight: 400;
    }

    .confidence-badges {
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    .confidence-score {
        font-weight: 600;
        padding: 0.375rem 0.75rem;
        border-radius: 8px;
        font-size: 0.75rem;
        letter-spacing: 0.5px;
    }

    .approval-badge {
        font-weight: 600;
        padding: 0.375rem 0.75rem;
        border-radius: 8px;
        font-size: 0.7rem;
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }

    .approved {
        background: linear-gradient(135deg, #1a4d2e 0%, #2d6a4f 100%);
        color: #95d5b2;
    }

    .not-approved {
        background: linear-gradient(135deg, #4d1a1a 0%, #6a2d2d 100%);
        color: #f5a5a5;
    }

    .confidence-high {
        background: linear-gradient(135deg, #1a4d2e 0%, #2d6a4f 100%);
        color: #95d5b2;
    }
    .confidence-medium {
        background: linear-gradient(135deg, #5c4d1a 0%, #7d6608 100%);
        color: #fca311;
    }
    .confidence-low {
        background: linear-gradient(135deg, #4d1a1a 0%, #6a2d2d 100%);
        color: #f5a5a5;
    }

    .avg-confidence {
        text-align: center;
        padding: 1.5rem;
        background: linear-gradient(135deg, #14213d 0%, #000000 100%);
        border-radius: 16px;
        margin: 1.5rem 0;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.3);
        border: 1px solid rgba(252, 163, 17, 0.2);
    }

    .avg-score {
        font-size: 3rem;
        font-weight: 700;
        color: #fca311;
        letter-spacing: -2px;
    }

    .avg-label {
        font-size: 0.75rem;
        color: #e5e5e5;
        text-transform: uppercase;
        letter-spacing: 2px;
        font-weight: 500;
        margin-top: 0.25rem;
    }

    /* History sidebar */
    .history-title {
        color: #ffffff;
        font-size: 0.85rem;
        font-weight: 600;
        margin-bottom: 1.5rem;
        padding-bottom: 0.75rem;
        border-bottom: 1px solid rgba(252, 163, 17, 0.2);
        text-transform: uppercase;
        letter-spacing: 2px;
    }

    .history-item {
        background: rgba(255, 255, 255, 0.03);
        border-radius: 14px;
        padding: 1rem;
        margin-bottom: 1rem;
        cursor: pointer;
        transition: all 0.3s ease;
        border: 1px solid rgba(255, 255, 255, 0.03);
    }

    .history-item:hover {
        background: rgba(255, 255, 255, 0.06);
        border-color: rgba(252, 163, 17, 0.3);
        transform: translateX(4px);
    }

    .history-time {
        font-size: 0.7rem;
        color: #fca311;
        margin-bottom: 0.375rem;
        font-weight: 500;
        letter-spacing: 0.5px;
    }

    .history-preview {
        font-size: 0.8rem;
        color: rgba(255, 255, 255, 0.7);
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
        font-weight: 300;
    }

    .history-confidence {
        font-size: 0.7rem;
        color: #e5e5e5;
        margin-top: 0.375rem;
        font-weight: 600;
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #000000 0%, #14213d 100%);
        border-right: 1px solid rgba(252, 163, 17, 0.1);
    }

    [data-testid="stSidebar"] [data-testid="stImage"] {
        border-radius: 10px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
    }

    [data-testid="stSidebar"] .stButton > button {
        background: rgba(252, 163, 17, 0.1);
        color: #fca311;
        border: 1px solid rgba(252, 163, 17, 0.3);
        font-size: 0.8rem;
        padding: 0.625rem 1rem;
    }

    [data-testid="stSidebar"] .stButton > button:hover {
        background: rgba(252, 163, 17, 0.2);
        color: #ffffff;
    }

    /* Divider styling */
    [data-testid="stSidebar"] hr {
        border-color: rgba(252, 163, 17, 0.1);
        margin: 0.75rem 0;
    }

    /* Badge/Tag styling */
    .premium-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        background: linear-gradient(135deg, #fca311 0%, #e09000 100%);
        color: #000000;
        border-radius: 20px;
        font-size: 0.65rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* Scrollbar styling */
    ::-webkit-scrollbar {
        width: 6px;
        height: 6px;
    }

    ::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.03);
        border-radius: 3px;
    }

    ::-webkit-scrollbar-thumb {
        background: rgba(252, 163, 17, 0.3);
        border-radius: 3px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: rgba(252, 163, 17, 0.5);
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(255, 255, 255, 0.03);
        border-radius: 16px;
        padding: 8px;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }

    .stTabs [data-baseweb="tab"] {
        height: 45px;
        background: transparent;
        border-radius: 12px;
        color: rgba(255, 255, 255, 0.5);
        font-weight: 500;
        font-size: 0.9rem;
        padding: 0 24px;
        border: none;
        font-family: 'Lexend', sans-serif !important;
    }

    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(252, 163, 17, 0.1);
        color: #ffffff;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #fca311 0%, #e09000 100%) !important;
        color: #000000 !important;
        font-weight: 600;
    }

    .stTabs [data-baseweb="tab-highlight"] {
        display: none;
    }

    .stTabs [data-baseweb="tab-border"] {
        display: none;
    }

    /* Hallmark info card */
    .hallmark-card {
        background: linear-gradient(145deg, #1a4d2e 0%, #14213d 100%);
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid rgba(149, 213, 178, 0.3);
    }

    .hallmark-title {
        color: #95d5b2;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-bottom: 1rem;
    }

    .hallmark-item {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.5rem 0;
        border-bottom: 1px solid rgba(255, 255, 255, 0.05);
    }

    .hallmark-item:last-child {
        border-bottom: none;
    }

    .hallmark-label {
        color: rgba(255, 255, 255, 0.6);
        font-size: 0.85rem;
    }

    .hallmark-value {
        color: #fca311;
        font-weight: 600;
        font-size: 0.9rem;
    }

    .bis-badge {
        display: inline-block;
        padding: 0.375rem 0.875rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    .bis-certified {
        background: linear-gradient(135deg, #1a4d2e 0%, #2d6a4f 100%);
        color: #95d5b2;
    }

    .bis-not-certified {
        background: linear-gradient(135deg, #4d3a1a 0%, #6a4d08 100%);
        color: #fca311;
    }

    /* Type badge */
    .type-badge {
        display: inline-block;
        padding: 0.25rem 0.5rem;
        border-radius: 6px;
        font-size: 0.65rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-left: 0.5rem;
    }

    .type-purity {
        background: rgba(252, 163, 17, 0.2);
        color: #fca311;
    }

    .type-huid {
        background: rgba(149, 213, 178, 0.2);
        color: #95d5b2;
    }

    .type-check {
        background: rgba(100, 149, 237, 0.2);
        color: #6495ed;
    }

    .type-unknown {
        background: rgba(255, 255, 255, 0.1);
        color: rgba(255, 255, 255, 0.5);
    }

    /* Check info card */
    .check-card {
        background: linear-gradient(145deg, #1a3d5c 0%, #14213d 100%);
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid rgba(100, 149, 237, 0.3);
    }

    .check-title {
        color: #6495ed;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-bottom: 1rem;
    }

    .check-item {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.5rem 0;
        border-bottom: 1px solid rgba(255, 255, 255, 0.05);
    }

    .check-item:last-child {
        border-bottom: none;
    }

    .check-label {
        color: rgba(255, 255, 255, 0.6);
        font-size: 0.85rem;
    }

    .check-value {
        color: #6495ed;
        font-weight: 600;
        font-size: 0.9rem;
        font-family: 'Courier New', monospace;
    }

    .check-badge {
        display: inline-block;
        padding: 0.375rem 0.875rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    .check-valid {
        background: linear-gradient(135deg, #1a4d2e 0%, #2d6a4f 100%);
        color: #95d5b2;
    }

    .check-invalid {
        background: linear-gradient(135deg, #4d3a1a 0%, #6a4d08 100%);
        color: #fca311;
    }

    .validated-badge {
        background: linear-gradient(135deg, #1a4d2e 0%, #2d6a4f 100%);
        color: #95d5b2;
        padding: 0.25rem 0.5rem;
        border-radius: 6px;
        font-size: 0.65rem;
        font-weight: 600;
        margin-left: 0.5rem;
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
    """Render premium progress steps."""
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

    # Calculate average confidence
    avg_conf = sum(r.confidence for r in results) / len(results)

    # Display average confidence
    st.markdown(f"""
    <div class="avg-confidence">
        <div class="avg-score">{avg_conf * 100:.1f}%</div>
        <div class="avg-label">Confidence Score</div>
    </div>
    """, unsafe_allow_html=True)

    # Display individual results
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

    # Only show hallmark card if there's hallmark data
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
        # Only show "no data" if there's also no check info
        st.markdown(f"""
        <div class="hallmark-card">
            <div class="hallmark-title">
                Hallmark Information
                <span class="bis-badge {bis_class}">{bis_text}</span>
            </div>
            <div class="hallmark-item">
                <span class="hallmark-label" style="color: rgba(255,255,255,0.4);">No hallmark data detected</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Render check info if available
    if hallmark_info.check_info:
        render_check_info(hallmark_info.check_info)


def render_confidence_results_v2(results: list, hallmark_info):
    """Render OCR results with confidence scores and hallmark info (V2)."""
    if not results:
        st.info("No text detected in this image.")
        return

    # Calculate average confidence
    avg_conf = sum(r.confidence for r in results) / len(results)

    # Display average confidence
    st.markdown(f"""
    <div class="avg-confidence">
        <div class="avg-score">{avg_conf * 100:.1f}%</div>
        <div class="avg-label">Confidence Score</div>
    </div>
    """, unsafe_allow_html=True)

    # Display hallmark info card
    render_hallmark_info(hallmark_info)

    # Display individual results
    st.markdown('<p class="output-label">Detected Text</p>', unsafe_allow_html=True)

    for r in results:
        conf_class = get_confidence_class(r.confidence)
        is_approved = r.confidence >= 0.75
        approval_class = "approved" if is_approved else "not-approved"
        approval_text = "Approved" if is_approved else "Not Approved"

        # Type badge
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
            <p style="color: rgba(255,255,255,0.3); font-size: 0.85rem; text-align: center; font-weight: 300;">
                No history yet
            </p>
            """, unsafe_allow_html=True)
            return

        # Clear history button
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
                        st.markdown("—")

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
    # Initialize V1 session state
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

            st.markdown("<div style='height: 1rem'></div>", unsafe_allow_html=True)

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
                <p style="font-size: 1.2rem; font-weight: 400;">Drop your image here</p>
                <p style="font-size: 0.85rem;">PNG, JPG, JPEG, BMP, or WEBP</p>
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
                st.markdown('<p class="output-label" style="margin-top: 1.5rem;">Full Text</p>', unsafe_allow_html=True)
                st.text_area("Full text", combined_text, height=150, label_visibility="collapsed", key="text_v1")

            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="
                background: rgba(255, 255, 255, 0.02);
                border: 2px dashed rgba(252, 163, 17, 0.15);
                border-radius: 24px;
                padding: 4rem 2rem;
                text-align: center;
                color: rgba(255, 255, 255, 0.25);
            ">
                <p style="font-size: 1rem; margin-bottom: 0.5rem;">Results will appear here</p>
                <p style="font-size: 0.8rem;">Upload an image and click Extract</p>
            </div>
            """, unsafe_allow_html=True)


def render_v2_tab():
    """Render the V2 (Hallmark OCR) tab content."""
    # Initialize V2 session state
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

            st.markdown("<div style='height: 1rem'></div>", unsafe_allow_html=True)

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
                <p style="font-size: 1.2rem; font-weight: 400;">Drop your hallmark image here</p>
                <p style="font-size: 0.85rem;">PNG, JPG, JPEG, BMP, or WEBP</p>
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
                st.markdown('<p class="output-label" style="margin-top: 1.5rem;">Full Text</p>', unsafe_allow_html=True)
                st.text_area("Full text", combined_text, height=150, label_visibility="collapsed", key="text_v2")

            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="
                background: rgba(255, 255, 255, 0.02);
                border: 2px dashed rgba(252, 163, 17, 0.15);
                border-radius: 24px;
                padding: 4rem 2rem;
                text-align: center;
                color: rgba(255, 255, 255, 0.25);
            ">
                <p style="font-size: 1rem; margin-bottom: 0.5rem;">Results will appear here</p>
                <p style="font-size: 0.8rem;">Upload a hallmark image and click Extract</p>
            </div>
            """, unsafe_allow_html=True)


def main():
    # Render history sidebar
    render_history_sidebar()

    # Header
    st.markdown('<h1 class="main-header">OCR Studio</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Premium Text Extraction</p>', unsafe_allow_html=True)

    # Tabs for V1 and V2
    tab1, tab2 = st.tabs(["Standard OCR", "Hallmark OCR V2"])

    with tab1:
        render_v1_tab()

    with tab2:
        render_v2_tab()


if __name__ == "__main__":
    main()
