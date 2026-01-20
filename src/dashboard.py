"""
Streamlit Dashboard for OCR Text Detection.
Minimalistic design with teal background and warm accent colors.
"""

import streamlit as st
from PIL import Image
from ocr_model import OCREngine
from history import add_to_history, get_history, clear_history


# Page configuration
st.set_page_config(
    page_title="OCR Text Detector",
    page_icon="",
    layout="wide",
)

# Custom CSS - Teal background theme with warm accent
st.markdown("""
<style>
    /* Color palette:
       Background: Teal #0F766E
       Card: White #FFFFFF
       Accent: Warm Peach #FB923C
       Text Light: #F0FDFA
       Text Dark: #134E4A
    */

    .stApp {
        background: linear-gradient(135deg, #0F766E 0%, #115E59 100%);
        min-height: 100vh;
    }

    /* Hide default Streamlit elements */
    #MainMenu, footer, header {visibility: hidden;}

    /* Main container */
    .block-container {
        padding-top: 2rem;
        max-width: 1200px;
    }

    /* Header */
    .main-header {
        color: #FFFFFF;
        font-size: 2.2rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 0.25rem;
        letter-spacing: -0.5px;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }

    .sub-header {
        color: #99F6E4;
        font-size: 0.95rem;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 400;
    }

    /* Upload area */
    [data-testid="stFileUploader"] {
        background: rgba(255, 255, 255, 0.95);
        border: 2px dashed #14B8A6;
        border-radius: 16px;
        padding: 1.5rem;
    }

    [data-testid="stFileUploader"]:hover {
        border-color: #FB923C;
        background: #FFFFFF;
    }

    /* Image container */
    [data-testid="stImage"] {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.2);
    }

    /* Button */
    .stButton > button {
        background: linear-gradient(135deg, #FB923C 0%, #F97316 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.875rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        width: 100%;
        transition: all 0.2s ease;
        box-shadow: 0 4px 14px rgba(249, 115, 22, 0.4);
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(249, 115, 22, 0.5);
        background: linear-gradient(135deg, #FDBA74 0%, #FB923C 100%);
    }

    .stButton > button:active {
        transform: translateY(0);
    }

    /* Progress steps */
    .step-container {
        display: flex;
        justify-content: center;
        gap: 2rem;
        margin: 1.5rem 0;
    }

    .step {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        color: rgba(255, 255, 255, 0.5);
        font-size: 0.85rem;
    }

    .step.active { color: #FFFFFF; }
    .step.completed { color: #99F6E4; }

    .step-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: rgba(255, 255, 255, 0.3);
    }

    .step.active .step-dot {
        background: #FB923C;
        box-shadow: 0 0 0 4px rgba(251, 146, 60, 0.3);
    }

    .step.completed .step-dot { background: #99F6E4; }

    /* Output section */
    .output-container {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 16px;
        padding: 1.5rem;
        margin-top: 1rem;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.15);
    }

    .output-label {
        color: #0F766E;
        font-weight: 600;
        font-size: 0.875rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 0.75rem;
    }

    /* Text area styling */
    .stTextArea textarea {
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        font-family: 'SF Mono', 'Fira Code', monospace;
        font-size: 0.9rem;
        color: #134E4A;
        background: #F0FDFA;
    }

    .stTextArea textarea:focus {
        border-color: #0F766E;
        box-shadow: 0 0 0 3px rgba(15, 118, 110, 0.2);
    }

    /* Info message */
    .stAlert {
        border-radius: 12px;
        border: none;
        background: rgba(255, 255, 255, 0.9);
    }

    /* Processing status */
    .processing-status {
        text-align: center;
        padding: 1.5rem;
        color: #FFFFFF;
        background: rgba(255, 255, 255, 0.1);
        border-radius: 12px;
    }

    .processing-step {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 0.75rem;
        margin: 0.5rem 0;
        font-size: 0.9rem;
    }

    .processing-step.done { color: #99F6E4; }
    .processing-step.active { color: #FB923C; }
    .processing-step.pending { color: rgba(255, 255, 255, 0.4); }

    /* Upload prompt text */
    .upload-prompt {
        text-align: center;
        padding: 2rem 1rem;
        color: rgba(255, 255, 255, 0.7);
    }

    /* Confidence display */
    .confidence-item {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.5rem 0.75rem;
        margin: 0.25rem 0;
        background: #F0FDFA;
        border-radius: 8px;
        font-size: 0.85rem;
    }

    .confidence-text {
        color: #134E4A;
        flex: 1;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
        margin-right: 1rem;
    }

    .confidence-score {
        font-weight: 600;
        padding: 0.25rem 0.5rem;
        border-radius: 6px;
        font-size: 0.75rem;
    }

    .confidence-high { background: #D1FAE5; color: #065F46; }
    .confidence-medium { background: #FEF3C7; color: #92400E; }
    .confidence-low { background: #FEE2E2; color: #991B1B; }

    .avg-confidence {
        text-align: center;
        padding: 1rem;
        background: rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        margin: 1rem 0;
    }

    .avg-score {
        font-size: 2rem;
        font-weight: 700;
        color: #FFFFFF;
    }

    .avg-label {
        font-size: 0.8rem;
        color: #99F6E4;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* History sidebar */
    .history-title {
        color: #FFFFFF;
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid rgba(255, 255, 255, 0.2);
    }

    .history-item {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 0.75rem;
        margin-bottom: 0.75rem;
        cursor: pointer;
        transition: all 0.2s ease;
    }

    .history-item:hover {
        background: rgba(255, 255, 255, 0.2);
    }

    .history-time {
        font-size: 0.7rem;
        color: #99F6E4;
        margin-bottom: 0.25rem;
    }

    .history-preview {
        font-size: 0.75rem;
        color: rgba(255, 255, 255, 0.8);
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
    }

    .history-confidence {
        font-size: 0.7rem;
        color: #FB923C;
        margin-top: 0.25rem;
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0D5652 0%, #0F766E 100%);
    }

    [data-testid="stSidebar"] [data-testid="stImage"] {
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_ocr_engine():
    """Load OCR engine (cached for performance)."""
    return OCREngine()


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
        html += f'<div class="step {status}"><span class="step-dot"></span>{step}</div>'
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
        <div class="avg-label">Average Confidence</div>
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
        st.markdown('<p class="history-title">History</p>', unsafe_allow_html=True)

        history = get_history()

        if not history:
            st.markdown("""
            <p style="color: rgba(255,255,255,0.5); font-size: 0.85rem; text-align: center;">
                No history yet
            </p>
            """, unsafe_allow_html=True)
            return

        # Clear history button
        if st.button("Clear History", key="clear_history"):
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
                        st.markdown("No preview")

                with col2:
                    st.markdown(f"""
                    <div class="history-time">{entry["timestamp"]}</div>
                    <div class="history-preview">{entry["text"][:50]}...</div>
                    <div class="history-confidence">Conf: {entry["avg_confidence"] * 100:.0f}%</div>
                    """, unsafe_allow_html=True)

                st.markdown("---")


def main():
    # Render history sidebar
    render_history_sidebar()

    # Header
    st.markdown('<h1 class="main-header">OCR Text Detector</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Extract text from images using AI</p>', unsafe_allow_html=True)

    # Initialize session state
    if "ocr_results" not in st.session_state:
        st.session_state.ocr_results = None
    if "processed" not in st.session_state:
        st.session_state.processed = False
    if "current_image" not in st.session_state:
        st.session_state.current_image = None

    # Main content area
    col_main, col_results = st.columns([1, 1])

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

            # Extract button
            if st.button("Extract Text", use_container_width=True):
                # Processing display
                status_placeholder = st.empty()

                with status_placeholder.container():
                    st.markdown("""
                    <div class="processing-status">
                        <div class="processing-step done">&#10003; Image loaded</div>
                        <div class="processing-step active">&#9679; Detecting text...</div>
                        <div class="processing-step pending">&#9675; Recognizing</div>
                    </div>
                    """, unsafe_allow_html=True)

                # Load model and process
                ocr_engine = load_ocr_engine()

                status_placeholder.markdown("""
                <div class="processing-status">
                    <div class="processing-step done">&#10003; Image loaded</div>
                    <div class="processing-step done">&#10003; Text detected</div>
                    <div class="processing-step active">&#9679; Recognizing...</div>
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
                <p style="font-size: 1.1rem;">Upload an image to get started</p>
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
                st.markdown('<p class="output-label" style="margin-top: 1rem;">Full Text</p>', unsafe_allow_html=True)
                st.text_area("Full text", combined_text, height=150, label_visibility="collapsed")

            st.markdown('</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
