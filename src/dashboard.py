"""
Streamlit Dashboard for OCR Text Detection.
Premium minimalist design with teal and beige using Lexend font.
"""

import streamlit as st
from PIL import Image
from ocr_model import OCREngine
from history import add_to_history, get_history, clear_history


# Page configuration
st.set_page_config(
    page_title="OCR Studio",
    page_icon="",
    layout="wide",
)

# Custom CSS - Premium teal & beige theme with Lexend font
st.markdown("""
<style>
    /* Import Lexend font from Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Lexend:wght@300;400;500;600;700&display=swap');

    /* Color palette:
       Primary: Deep Teal #115E59
       Secondary: Soft Beige #F5F0E8
       Accent: Warm Sand #D4A574
       Card: Cream #FAF8F5
       Text: Charcoal #1C1917
    */

    * {
        font-family: 'Lexend', sans-serif !important;
    }

    .stApp {
        background: linear-gradient(160deg, #115E59 0%, #0D4F4A 50%, #134E4A 100%);
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
        color: #F5F0E8;
        font-size: 2.8rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 0.5rem;
        letter-spacing: -1px;
    }

    .sub-header {
        color: #99CCC7;
        font-size: 1rem;
        text-align: center;
        margin-bottom: 3rem;
        font-weight: 300;
        letter-spacing: 2px;
        text-transform: uppercase;
    }

    /* Premium Card Style */
    .premium-card {
        background: linear-gradient(145deg, #FAF8F5 0%, #F5F0E8 100%);
        border-radius: 24px;
        padding: 2rem;
        box-shadow:
            0 4px 6px rgba(0, 0, 0, 0.05),
            0 20px 40px rgba(0, 0, 0, 0.1),
            inset 0 1px 0 rgba(255, 255, 255, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }

    /* Upload area */
    [data-testid="stFileUploader"] {
        background: linear-gradient(145deg, #FAF8F5 0%, #F5F0E8 100%);
        border: 2px dashed #D4A574;
        border-radius: 20px;
        padding: 2rem;
        transition: all 0.3s ease;
    }

    [data-testid="stFileUploader"]:hover {
        border-color: #115E59;
        background: #FFFFFF;
        box-shadow: 0 8px 32px rgba(212, 165, 116, 0.2);
    }

    [data-testid="stFileUploader"] label,
    [data-testid="stFileUploader"] p,
    [data-testid="stFileUploader"] span {
        color: #1C1917 !important;
        font-family: 'Lexend', sans-serif !important;
    }

    /* Image container */
    [data-testid="stImage"] {
        border-radius: 16px;
        overflow: hidden;
        box-shadow:
            0 4px 6px rgba(0, 0, 0, 0.07),
            0 12px 28px rgba(0, 0, 0, 0.12);
    }

    /* Premium Button */
    .stButton > button {
        background: linear-gradient(135deg, #D4A574 0%, #C49A6C 100%);
        color: #1C1917;
        border: none;
        border-radius: 14px;
        padding: 1rem 2.5rem;
        font-weight: 600;
        font-size: 0.95rem;
        letter-spacing: 0.5px;
        width: 100%;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow:
            0 4px 14px rgba(212, 165, 116, 0.4),
            inset 0 1px 0 rgba(255, 255, 255, 0.3);
        font-family: 'Lexend', sans-serif !important;
    }

    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow:
            0 8px 24px rgba(212, 165, 116, 0.5),
            inset 0 1px 0 rgba(255, 255, 255, 0.4);
        background: linear-gradient(135deg, #E0B485 0%, #D4A574 100%);
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
        background: rgba(255, 255, 255, 0.05);
        border-radius: 16px;
        backdrop-filter: blur(10px);
    }

    .step {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        color: rgba(245, 240, 232, 0.4);
        font-size: 0.85rem;
        font-weight: 500;
        padding: 0.5rem 1rem;
        border-radius: 10px;
        transition: all 0.3s ease;
    }

    .step.active {
        color: #F5F0E8;
        background: rgba(212, 165, 116, 0.2);
    }

    .step.completed {
        color: #99CCC7;
    }

    .step-line {
        width: 40px;
        height: 2px;
        background: rgba(255, 255, 255, 0.15);
        border-radius: 1px;
    }

    .step.completed + .step-line,
    .step-line.completed {
        background: #99CCC7;
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
        background: rgba(255, 255, 255, 0.1);
        border: 2px solid rgba(255, 255, 255, 0.2);
    }

    .step.active .step-number {
        background: #D4A574;
        border-color: #D4A574;
        color: #1C1917;
        box-shadow: 0 0 20px rgba(212, 165, 116, 0.4);
    }

    .step.completed .step-number {
        background: #115E59;
        border-color: #99CCC7;
        color: #F5F0E8;
    }

    /* Output section */
    .output-container {
        background: linear-gradient(145deg, #FAF8F5 0%, #F5F0E8 100%);
        border-radius: 24px;
        padding: 2rem;
        box-shadow:
            0 4px 6px rgba(0, 0, 0, 0.05),
            0 20px 40px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.3);
    }

    .output-label {
        color: #115E59;
        font-weight: 600;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-bottom: 1rem;
    }

    /* Text area styling */
    .stTextArea textarea {
        border: 1px solid #E8E0D5;
        border-radius: 14px;
        font-family: 'Lexend', sans-serif !important;
        font-size: 0.9rem;
        color: #1C1917;
        background: #FFFFFF;
        padding: 1rem;
    }

    .stTextArea textarea:focus {
        border-color: #D4A574;
        box-shadow: 0 0 0 4px rgba(212, 165, 116, 0.15);
    }

    /* Info message */
    .stAlert {
        border-radius: 14px;
        border: none;
        background: #FAF8F5;
    }

    /* Processing status */
    .processing-status {
        text-align: center;
        padding: 2rem;
        color: #F5F0E8;
        background: rgba(255, 255, 255, 0.05);
        border-radius: 16px;
        backdrop-filter: blur(10px);
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

    .processing-step.done { color: #99CCC7; }
    .processing-step.active { color: #D4A574; }
    .processing-step.pending { color: rgba(245, 240, 232, 0.3); }

    /* Upload prompt text */
    .upload-prompt {
        text-align: center;
        padding: 3rem 1rem;
        color: rgba(245, 240, 232, 0.6);
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
        background: #FFFFFF;
        border-radius: 12px;
        font-size: 0.9rem;
        border: 1px solid #E8E0D5;
        transition: all 0.2s ease;
    }

    .confidence-item:hover {
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
        transform: translateX(4px);
    }

    .confidence-text {
        color: #1C1917;
        flex: 1;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
        margin-right: 1rem;
        font-weight: 400;
    }

    .confidence-score {
        font-weight: 600;
        padding: 0.375rem 0.75rem;
        border-radius: 8px;
        font-size: 0.75rem;
        letter-spacing: 0.5px;
    }

    .confidence-high {
        background: linear-gradient(135deg, #D1E7DD 0%, #BADBCC 100%);
        color: #0F5132;
    }
    .confidence-medium {
        background: linear-gradient(135deg, #FFF3CD 0%, #FFE69C 100%);
        color: #664D03;
    }
    .confidence-low {
        background: linear-gradient(135deg, #F8D7DA 0%, #F5C2C7 100%);
        color: #842029;
    }

    .avg-confidence {
        text-align: center;
        padding: 1.5rem;
        background: linear-gradient(135deg, #115E59 0%, #0D4F4A 100%);
        border-radius: 16px;
        margin: 1.5rem 0;
        box-shadow: 0 8px 24px rgba(17, 94, 89, 0.3);
    }

    .avg-score {
        font-size: 3rem;
        font-weight: 700;
        color: #F5F0E8;
        letter-spacing: -2px;
    }

    .avg-label {
        font-size: 0.75rem;
        color: #99CCC7;
        text-transform: uppercase;
        letter-spacing: 2px;
        font-weight: 500;
        margin-top: 0.25rem;
    }

    /* History sidebar */
    .history-title {
        color: #F5F0E8;
        font-size: 0.85rem;
        font-weight: 600;
        margin-bottom: 1.5rem;
        padding-bottom: 0.75rem;
        border-bottom: 1px solid rgba(245, 240, 232, 0.15);
        text-transform: uppercase;
        letter-spacing: 2px;
    }

    .history-item {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 14px;
        padding: 1rem;
        margin-bottom: 1rem;
        cursor: pointer;
        transition: all 0.3s ease;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }

    .history-item:hover {
        background: rgba(255, 255, 255, 0.1);
        border-color: rgba(212, 165, 116, 0.3);
        transform: translateX(4px);
    }

    .history-time {
        font-size: 0.7rem;
        color: #99CCC7;
        margin-bottom: 0.375rem;
        font-weight: 500;
        letter-spacing: 0.5px;
    }

    .history-preview {
        font-size: 0.8rem;
        color: rgba(245, 240, 232, 0.8);
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
        font-weight: 300;
    }

    .history-confidence {
        font-size: 0.7rem;
        color: #D4A574;
        margin-top: 0.375rem;
        font-weight: 600;
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0D4F4A 0%, #115E59 50%, #134E4A 100%);
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }

    [data-testid="stSidebar"] [data-testid="stImage"] {
        border-radius: 10px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
    }

    [data-testid="stSidebar"] .stButton > button {
        background: rgba(212, 165, 116, 0.2);
        color: #D4A574;
        border: 1px solid rgba(212, 165, 116, 0.3);
        font-size: 0.8rem;
        padding: 0.625rem 1rem;
    }

    [data-testid="stSidebar"] .stButton > button:hover {
        background: rgba(212, 165, 116, 0.3);
        color: #F5F0E8;
    }

    /* Divider styling */
    [data-testid="stSidebar"] hr {
        border-color: rgba(245, 240, 232, 0.1);
        margin: 0.75rem 0;
    }

    /* Badge/Tag styling */
    .premium-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        background: linear-gradient(135deg, #D4A574 0%, #C49A6C 100%);
        color: #1C1917;
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
        background: rgba(255, 255, 255, 0.05);
        border-radius: 3px;
    }

    ::-webkit-scrollbar-thumb {
        background: rgba(212, 165, 116, 0.3);
        border-radius: 3px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: rgba(212, 165, 116, 0.5);
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_ocr_engine():
    """Load OCR engine (cached for performance)."""
    return OCREngine()


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
    """Render OCR results with confidence scores."""
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
        st.markdown(f"""
        <div class="confidence-item">
            <span class="confidence-text">{r.text}</span>
            <span class="confidence-score {conf_class}">{r.confidence * 100:.0f}%</span>
        </div>
        """, unsafe_allow_html=True)


def render_history_sidebar():
    """Render history in sidebar."""
    with st.sidebar:
        st.markdown('<p class="history-title">Recent Scans</p>', unsafe_allow_html=True)

        history = get_history()

        if not history:
            st.markdown("""
            <p style="color: rgba(245,240,232,0.4); font-size: 0.85rem; text-align: center; font-weight: 300;">
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


def main():
    # Render history sidebar
    render_history_sidebar()

    # Header
    st.markdown('<h1 class="main-header">OCR Studio</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Premium Text Extraction</p>', unsafe_allow_html=True)

    # Initialize session state
    if "ocr_results" not in st.session_state:
        st.session_state.ocr_results = None
    if "processed" not in st.session_state:
        st.session_state.processed = False
    if "current_image" not in st.session_state:
        st.session_state.current_image = None

    # Main content area
    col_main, col_results = st.columns([1, 1], gap="large")

    with col_main:
        # File uploader
        uploaded_file = st.file_uploader(
            "Drop an image here or click to upload",
            type=["png", "jpg", "jpeg", "bmp", "webp"],
            help="Supported: PNG, JPG, JPEG, BMP, WEBP",
            label_visibility="collapsed"
        )

        if uploaded_file:
            # Determine current step
            if st.session_state.processed and st.session_state.ocr_results is not None:
                current_step = 2
            else:
                current_step = 1

            # Show progress steps
            render_steps(current_step)

            # Display uploaded image
            image = Image.open(uploaded_file)
            st.session_state.current_image = image.copy()
            st.image(image, use_container_width=True)

            st.markdown("<div style='height: 1rem'></div>", unsafe_allow_html=True)

            # Extract button
            if st.button("Extract Text", use_container_width=True):
                # Processing display
                status_placeholder = st.empty()

                with status_placeholder.container():
                    st.markdown("""
                    <div class="processing-status">
                        <div class="processing-step done">&#10003; Image loaded</div>
                        <div class="processing-step active">&#9679; Analyzing...</div>
                        <div class="processing-step pending">&#9675; Extracting text</div>
                    </div>
                    """, unsafe_allow_html=True)

                # Load model and process
                ocr_engine = load_ocr_engine()

                status_placeholder.markdown("""
                <div class="processing-status">
                    <div class="processing-step done">&#10003; Image loaded</div>
                    <div class="processing-step done">&#10003; Analysis complete</div>
                    <div class="processing-step active">&#9679; Extracting...</div>
                </div>
                """, unsafe_allow_html=True)

                # Extract text with confidence
                results = ocr_engine.extract_text_with_confidence(image)
                st.session_state.ocr_results = results
                st.session_state.processed = True

                # Save to history
                if results:
                    avg_conf = sum(r.confidence for r in results) / len(results)
                    text = "\n".join([r.text for r in results])
                    add_to_history(image, text, avg_conf)

                status_placeholder.empty()
                st.rerun()

        else:
            # Reset state when no file
            st.session_state.ocr_results = None
            st.session_state.processed = False
            st.session_state.current_image = None
            render_steps(0)

            # Upload prompt
            st.markdown("""
            <div class="upload-prompt">
                <p style="font-size: 1.2rem; font-weight: 400;">Drop your image here</p>
                <p style="font-size: 0.85rem;">PNG, JPG, JPEG, BMP, or WEBP</p>
            </div>
            """, unsafe_allow_html=True)

    with col_results:
        if st.session_state.processed and st.session_state.ocr_results is not None:
            st.markdown('<div class="output-container">', unsafe_allow_html=True)
            render_confidence_results(st.session_state.ocr_results)

            # Combined text area
            if st.session_state.ocr_results:
                combined_text = "\n".join([r.text for r in st.session_state.ocr_results])
                st.markdown('<p class="output-label" style="margin-top: 1.5rem;">Full Text</p>', unsafe_allow_html=True)
                st.text_area("Full text", combined_text, height=150, label_visibility="collapsed")

            st.markdown('</div>', unsafe_allow_html=True)
        else:
            # Placeholder for results
            st.markdown("""
            <div style="
                background: rgba(255, 255, 255, 0.03);
                border: 2px dashed rgba(245, 240, 232, 0.1);
                border-radius: 24px;
                padding: 4rem 2rem;
                text-align: center;
                color: rgba(245, 240, 232, 0.3);
            ">
                <p style="font-size: 1rem; margin-bottom: 0.5rem;">Results will appear here</p>
                <p style="font-size: 0.8rem;">Upload an image and click Extract</p>
            </div>
            """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
