"""
QC Dashboard for Jewelry Hallmarking OCR Validation.

A Streamlit-based dashboard for testing and using the QC validation endpoints.
Supports both integration flows:
1. Hallmarking Stage - Real-time validation
2. QC Dashboard - Batch validation with approval workflow

Features:
- Hallmark validation with error categorization
- HUID format validation
- BIS compliance rules display
- Jewelry-specific rulesets management
- QC override workflow
"""

import streamlit as st
import requests
from PIL import Image
import io
import json
from datetime import datetime
import sys

# Add config path
sys.path.insert(0, "../config")
sys.path.insert(0, "config")

# Try to import jewelry rulesets
try:
    from jewelry_rulesets import (
        JewelryType, ErrorCategory, ErrorSeverity,
        JEWELRY_RULESETS, ERROR_DETAILS, ConfidenceBenchmark,
        JewelryRulesetValidator
    )
    RULESETS_AVAILABLE = True
except ImportError:
    RULESETS_AVAILABLE = False

# Configuration
API_BASE_URL = "http://localhost:8000"

# Page configuration
st.set_page_config(
    page_title="Hallmark QC Dashboard",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - Professional light theme matching BAC design system
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&display=swap');

    /* Theme:
       Maroon:       #7B1F3A
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
    }

    /* Header styling */
    .top-bar {
        background: #7B1F3A;
        margin: -1rem -2rem 0 -2rem;
        padding: 0 24px;
        height: 46px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        color: #FFFFFF;
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

    .page-header {
        background: #FFFFFF;
        margin: 0 -2rem;
        padding: 14px 24px;
        border-bottom: 1px solid #E8E2E4;
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

    /* Card styling */
    .result-card {
        background: #FFFFFF;
        border-radius: 10px;
        padding: 16px 18px;
        margin: 10px 0;
        border: 1px solid #E8E2E4;
        box-shadow: 0 2px 20px rgba(0, 0, 0, 0.06);
    }

    .approved-card {
        border-left: 3px solid #16A34A;
    }

    .rejected-card {
        border-left: 3px solid #DC2626;
    }

    .review-card {
        border-left: 3px solid #D97706;
    }

    /* Error cards */
    .error-critical {
        background: #FFFFFF;
        border-left: 3px solid #DC2626;
        padding: 12px 16px;
        margin: 6px 0;
        border-radius: 8px;
        border-top: 1px solid #F0ECED;
        border-right: 1px solid #F0ECED;
        border-bottom: 1px solid #F0ECED;
    }

    .error-major {
        background: #FFFFFF;
        border-left: 3px solid #D97706;
        padding: 12px 16px;
        margin: 6px 0;
        border-radius: 8px;
        border-top: 1px solid #F0ECED;
        border-right: 1px solid #F0ECED;
        border-bottom: 1px solid #F0ECED;
    }

    .error-minor {
        background: #FFFFFF;
        border-left: 3px solid #C8A44E;
        padding: 12px 16px;
        margin: 6px 0;
        border-radius: 8px;
        border-top: 1px solid #F0ECED;
        border-right: 1px solid #F0ECED;
        border-bottom: 1px solid #F0ECED;
    }

    /* Decision badges */
    .decision-badge {
        display: inline-block;
        padding: 4px 14px;
        border-radius: 12px;
        font-size: 10px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    .badge-approved {
        background: rgba(22, 163, 74, 0.1);
        color: #16A34A;
    }

    .badge-rejected {
        background: rgba(220, 38, 38, 0.1);
        color: #DC2626;
    }

    .badge-review {
        background: rgba(217, 119, 6, 0.1);
        color: #D97706;
    }

    /* Metric cards */
    .metric-card {
        background: #FFFFFF;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
        border: 1px solid #E8E2E4;
    }

    .metric-value {
        font-size: 2rem;
        font-weight: 800;
        color: #7B1F3A;
    }

    .metric-label {
        font-size: 10px;
        color: #9B9BAB;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        font-weight: 700;
    }

    /* Ruleset cards */
    .ruleset-card {
        background: #FFFFFF;
        border-radius: 10px;
        padding: 16px 18px;
        margin: 10px 0;
        border: 1px solid #E8E2E4;
        border-left: 3px solid #7B1F3A;
    }

    .ruleset-header {
        color: #7B1F3A;
        font-size: 14px;
        font-weight: 700;
        margin-bottom: 8px;
    }

    .position-acceptable {
        background: rgba(22, 163, 74, 0.06);
        border: 1px solid rgba(22, 163, 74, 0.2);
        padding: 8px 14px;
        border-radius: 8px;
        margin: 4px 0;
    }

    .position-forbidden {
        background: rgba(220, 38, 38, 0.06);
        border: 1px solid rgba(220, 38, 38, 0.2);
        padding: 8px 14px;
        border-radius: 8px;
        margin: 4px 0;
    }

    .position-preferred {
        border: 2px solid #16A34A !important;
    }

    /* Confidence gauge */
    .confidence-gauge {
        width: 100%;
        height: 8px;
        background: #F0ECED;
        border-radius: 4px;
        overflow: hidden;
        margin: 6px 0;
    }

    .confidence-fill {
        height: 100%;
        border-radius: 4px;
        transition: width 0.5s ease;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: #FFFFFF;
        border-right: 1px solid #E8E2E4;
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
        font-family: 'DM Sans', sans-serif !important;
    }

    .stTabs [aria-selected="true"] {
        background: #7B1F3A !important;
        color: #FFFFFF !important;
        font-weight: 700;
    }

    .stTabs [data-baseweb="tab-highlight"],
    .stTabs [data-baseweb="tab-border"] {
        display: none;
    }

    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(123, 31, 58, 0.06);
        color: #7B1F3A;
    }

    /* Buttons */
    .stButton > button {
        background: #7B1F3A;
        color: #FFFFFF;
        border: none;
        border-radius: 6px;
        padding: 10px 20px;
        font-weight: 600;
        font-size: 12px;
        font-family: 'DM Sans', sans-serif !important;
    }

    .stButton > button:hover {
        background: #9B3A56;
        transform: none;
    }

    /* Status indicators */
    .status-dot {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        margin-right: 8px;
    }

    .status-online { background: #16A34A; }
    .status-offline { background: #DC2626; }

    /* Rule badges */
    .rule-badge {
        display: inline-block;
        background: rgba(123, 31, 58, 0.08);
        color: #7B1F3A;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 11px;
        font-weight: 600;
        margin: 3px;
    }

    /* File uploader */
    [data-testid="stFileUploader"] section {
        background: #F8F6F7 !important;
        border: 2px dashed #E8E2E4 !important;
        border-radius: 10px !important;
    }

    [data-testid="stFileUploader"] section:hover {
        border-color: #7B1F3A !important;
    }

    [data-testid="stFileUploader"] button {
        background: #7B1F3A !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 6px !important;
        font-weight: 600 !important;
        font-family: 'DM Sans', sans-serif !important;
        font-size: 12px !important;
    }

    [data-testid="stFileUploader"] button:hover {
        background: #9B3A56 !important;
    }

    [data-testid="stFileUploader"] svg {
        color: #7B1F3A !important;
    }

    /* Image container */
    [data-testid="stImage"] {
        border-radius: 8px;
        overflow: hidden;
        border: 1px solid #E8E2E4;
    }

    /* Scrollbar */
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: #F5F1F2; border-radius: 3px; }
    ::-webkit-scrollbar-thumb { background: #E8E2E4; border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: #7B1F3A; }

    /* Select boxes and inputs */
    .stSelectbox > div > div {
        background: #F8F6F7;
        border-color: #E8E2E4;
        border-radius: 6px;
    }

    .stTextInput > div > div > input {
        background: #F8F6F7;
        border-color: #E8E2E4;
        border-radius: 6px;
    }

    .stTextArea textarea {
        background: #F8F6F7;
        border-color: #E8E2E4;
        border-radius: 6px;
        font-family: 'DM Sans', sans-serif !important;
    }

    #MainMenu, footer, header {visibility: hidden;}

    .block-container {
        padding: 0 2rem 2rem 2rem;
        max-width: 1400px;
    }

    /* Section label */
    .section-label {
        color: #7B1F3A;
        font-size: 10px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)


def check_api_health():
    """Check if API is running."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False


def get_confidence_color(score):
    """Get color based on confidence score."""
    if score >= 0.85:
        return "#16A34A"
    elif score >= 0.70:
        return "#D97706"
    elif score >= 0.50:
        return "#D97706"
    else:
        return "#DC2626"


def render_decision_badge(decision):
    """Render a styled decision badge."""
    badge_class = {
        "approved": "badge-approved",
        "rejected": "badge-rejected",
        "manual_review": "badge-review"
    }.get(decision, "badge-review")

    display_text = decision.replace("_", " ").upper()

    st.markdown(f"""
        <span class="decision-badge {badge_class}">{display_text}</span>
    """, unsafe_allow_html=True)


def render_confidence_gauge(score, label="Confidence"):
    """Render a visual confidence gauge."""
    color = get_confidence_color(score)
    percentage = score * 100

    st.markdown(f"""
        <div style="margin: 10px 0;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                <span style="color: #6B6B7B; font-size: 11px; font-weight: 600;">{label}</span>
                <span style="color: {color}; font-weight: 700; font-size: 12px;">{percentage:.1f}%</span>
            </div>
            <div class="confidence-gauge">
                <div class="confidence-fill" style="width: {percentage}%; background: {color};"></div>
            </div>
        </div>
    """, unsafe_allow_html=True)


def render_error_card(error_type, message, suggestion=None, severity="major"):
    """Render an error card with styling based on severity."""
    severity_class = f"error-{severity}"
    severity_label = {
        "critical": "CRITICAL",
        "major": "MAJOR",
        "minor": "MINOR"
    }.get(severity, "MAJOR")

    severity_color = {
        "critical": "#DC2626",
        "major": "#D97706",
        "minor": "#C8A44E"
    }.get(severity, "#D97706")

    suggestion_html = f"<p style='color: #16A34A; font-size: 11px; margin-top: 6px; font-weight: 500;'>Suggestion: {suggestion}</p>" if suggestion else ""

    st.markdown(f"""
        <div class="{severity_class}">
            <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 4px;">
                <span style="background: {severity_color}18; color: {severity_color}; padding: 2px 8px; border-radius: 10px; font-size: 9px; font-weight: 700; letter-spacing: 0.5px;">{severity_label}</span>
                <span style="color: #2D2D2D; font-weight: 700; font-size: 12px;">{error_type}</span>
            </div>
            <p style="color: #6B6B7B; margin: 0; font-size: 12px;">{message}</p>
            {suggestion_html}
        </div>
    """, unsafe_allow_html=True)


def render_hallmark_tab():
    """Render the Hallmark Validation tab with error categorization."""
    st.markdown('<p class="section-label">Hallmark Validation</p>', unsafe_allow_html=True)
    st.markdown("Upload an image to validate hallmark against BIS standards with detailed error analysis.")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("#### Upload Image")

        # Jewelry type selection
        jewelry_type = st.selectbox(
            "Jewelry Type",
            ["ring", "bangle", "chain", "necklace", "pendant", "earring", "mangalsutra", "other"],
            help="Select jewelry type for position validation"
        )

        uploaded_file = st.file_uploader(
            "Choose a hallmark image",
            type=["png", "jpg", "jpeg", "bmp", "webp"],
            key="hallmark_uploader"
        )

        job_id = st.text_input(
            "Job ID",
            value=f"HM-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            help="Unique job identifier"
        )

        col_a, col_b = st.columns(2)
        with col_a:
            expected_purity = st.selectbox(
                "Expected Purity",
                ["", "916", "750", "585", "375", "875", "958", "999"],
                help="For cross-validation"
            )
        with col_b:
            marking_position = st.selectbox(
                "Marking Position (Rings)",
                ["12_oclock", "3_oclock", "9_oclock", "6_oclock", "unknown"],
                help="Position of hallmark on ring"
            )

        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)

            if st.button("Validate Hallmark", use_container_width=True, key="validate_btn"):
                with st.spinner("Analyzing image and validating hallmark..."):
                    try:
                        uploaded_file.seek(0)
                        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}

                        response = requests.post(
                            f"{API_BASE_URL}/qc/validate/v2",
                            files=files
                        )

                        if response.status_code == 200:
                            result = response.json()

                            # Add jewelry-specific validation
                            if jewelry_type == "ring" and marking_position == "6_oclock":
                                if "data" in result:
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

    with col2:
        st.markdown("#### Validation Result")

        if "validation_result" in st.session_state and st.session_state.validation_result:
            result = st.session_state.validation_result
            data = result.get("data", {})

            # Decision badge
            decision = data.get("decision", "unknown")
            render_decision_badge(decision)

            # Confidence gauges
            st.markdown("##### Confidence Analysis")
            confidence = data.get("confidence", 0)
            render_confidence_gauge(confidence, "Overall Confidence")

            # Confidence benchmark info
            benchmark_info = {
                "excellent": (0.95, 1.0, "#16A34A"),
                "good": (0.85, 0.95, "#16A34A"),
                "acceptable": (0.70, 0.85, "#D97706"),
                "poor": (0.50, 0.70, "#D97706"),
                "unacceptable": (0.0, 0.50, "#DC2626"),
            }

            for label, (low, high, color) in benchmark_info.items():
                if low <= confidence <= high:
                    st.markdown(f"""
                        <div style="background: #F8F6F7; padding: 8px 14px; border-radius: 8px; margin: 6px 0; border: 1px solid #E8E2E4;">
                            <span style="color: {color}; font-weight: 700; font-size: 12px;">Score: {label.upper()}</span>
                            <span style="color: #9B9BAB; font-size: 11px; margin-left: 10px;">
                                ({low*100:.0f}% - {high*100:.0f}%)
                            </span>
                        </div>
                    """, unsafe_allow_html=True)
                    break

            # Hallmark data
            hallmark_data = data.get("hallmark_data", {})
            if hallmark_data:
                st.markdown("##### Detected Hallmark")
                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("Purity Code", hallmark_data.get("purity_code") or "--")
                    st.metric("Karat", hallmark_data.get("karat") or "--")
                with col_b:
                    purity_pct = hallmark_data.get("purity_percentage")
                    st.metric("Purity %", f"{purity_pct}%" if purity_pct else "--")
                    st.metric("HUID", hallmark_data.get("huid") or "--")

            # Validation status
            status = data.get("validation_status", {})
            st.markdown("##### Validation Checklist")
            checks = [
                ("Purity Valid", status.get("purity_valid", False)),
                ("HUID Valid", status.get("huid_valid", False)),
                ("BIS Certified", data.get("bis_certified", False)),
            ]
            for check_name, check_status in checks:
                if check_status:
                    color = "#16A34A"
                    icon = "&#10003;"
                else:
                    color = "#DC2626"
                    icon = "&#10007;"
                st.markdown(f"<span style='color: {color}; font-weight: 600; font-size: 12px;'>{icon} {check_name}</span>", unsafe_allow_html=True)

            # Error analysis section
            rejection_info = data.get("rejection_info")
            if rejection_info or decision != "approved":
                st.markdown("##### Error Analysis")

                reasons = rejection_info.get("reasons", []) if rejection_info else []

                # Image quality errors
                image_errors = [r for r in reasons if r.startswith("image_")]
                if image_errors:
                    st.markdown("###### Image Quality Issues")
                    for error in image_errors:
                        render_error_card(
                            error.replace("_", " ").title(),
                            "Image quality issue detected",
                            "Retake image with better lighting and focus",
                            "major"
                        )

                # Content errors
                content_errors = [r for r in reasons if r in ["missing_huid", "missing_purity_mark", "invalid_purity_code", "invalid_huid_format"]]
                if content_errors:
                    st.markdown("###### Hallmark Content Issues")
                    error_suggestions = {
                        "missing_huid": ("HUID not detected", "Ensure HUID is clearly engraved and visible", "critical"),
                        "missing_purity_mark": ("Purity mark not found", "Ensure purity code (916, 750, etc.) is visible", "critical"),
                        "invalid_purity_code": ("Invalid purity code", "Valid codes: 375, 585, 750, 875, 916, 958, 999", "critical"),
                        "invalid_huid_format": ("Invalid HUID format", "HUID must be 6 alphanumeric characters with at least one letter", "critical"),
                    }
                    for error in content_errors:
                        info = error_suggestions.get(error, (error, None, "major"))
                        render_error_card(info[0], f"Error: {error}", info[1], info[2])

                # Position errors (ring-specific)
                position_errors = [r for r in reasons if "position" in r.lower() or "6_oclock" in r.lower()]
                if position_errors or (st.session_state.get("jewelry_type") == "ring" and st.session_state.get("marking_position") == "6_oclock"):
                    st.markdown("###### Position Issues")
                    render_error_card(
                        "Incorrect Marking Position",
                        "Ring hallmark at 6 o'clock position (bottom inside) is NOT acceptable. This position causes the marking to wear off quickly due to contact with surfaces.",
                        "Hallmark should be at 12 o'clock (top), 3 o'clock (right), or 9 o'clock (left) position inside the ring",
                        "critical"
                    )

                # Confidence issues
                if confidence < 0.50:
                    st.markdown("###### Confidence Issues")
                    render_error_card(
                        "Low Confidence Score",
                        f"OCR confidence ({confidence*100:.1f}%) is below the acceptable threshold (50%)",
                        "Retake image with better lighting, focus, and positioning",
                        "major"
                    )
                elif confidence < 0.85 and decision == "manual_review":
                    st.markdown("###### Review Required")
                    render_error_card(
                        "Manual Review Needed",
                        f"Confidence ({confidence*100:.1f}%) is between 50-85%. Human verification required.",
                        "QC personnel should manually verify the hallmark",
                        "minor"
                    )

            # Raw JSON expander
            with st.expander("View Raw Response"):
                st.json(result)
        else:
            st.info("Upload an image and click 'Validate Hallmark' to see results.")


def render_jewelry_rulesets_tab():
    """Render the Jewelry Rulesets management tab."""
    st.markdown('<p class="section-label">Jewelry-Specific Rulesets</p>', unsafe_allow_html=True)
    st.markdown("View and understand marking rules for different jewelry types.")

    if not RULESETS_AVAILABLE:
        st.warning("Jewelry rulesets module not loaded. Using default rules.")

        default_rulesets = [
            {
                "type": "ring",
                "display_name": "Ring",
                "description": "Finger rings including engagement, wedding, and fashion rings",
                "acceptable_positions": [
                    {"name": "12 o'clock", "description": "Inside at top (preferred)", "preferred": True},
                    {"name": "3 o'clock", "description": "Inside at right side", "preferred": False},
                    {"name": "9 o'clock", "description": "Inside at left side", "preferred": False},
                ],
                "forbidden_positions": [
                    {"name": "6 o'clock", "description": "Inside at bottom", "reason": "Wears off quickly due to surface contact"},
                    {"name": "Outside", "description": "Outer visible surface", "reason": "Affects aesthetics"},
                ],
                "special_rules": [
                    "Hallmark must be inside the band",
                    "Marking should not interfere with stone settings",
                    "6 o'clock position (bottom inside) is NOT acceptable",
                ],
            },
            {
                "type": "bangle",
                "display_name": "Bangle",
                "description": "Rigid circular bracelets",
                "acceptable_positions": [
                    {"name": "Inside near opening", "description": "Inner surface near clasp", "preferred": True},
                    {"name": "Inside flat area", "description": "Any flat inner surface", "preferred": False},
                ],
                "forbidden_positions": [
                    {"name": "Outside decorative", "description": "Decorative outer surface", "reason": "Must not interfere with design"},
                ],
                "special_rules": [
                    "Marking should be on inner surface",
                    "Should not be on carved/embossed areas",
                ],
            },
            {
                "type": "chain",
                "display_name": "Chain",
                "description": "Neck chains and rope chains",
                "acceptable_positions": [
                    {"name": "Clasp area", "description": "On or near the clasp", "preferred": True},
                    {"name": "Tag attached", "description": "On attached hallmark tag", "preferred": True},
                ],
                "forbidden_positions": [
                    {"name": "Middle links", "description": "On middle chain links", "reason": "Would weaken chain structure"},
                ],
                "special_rules": [
                    "Marking on clasp or attached tag",
                    "Must not weaken any chain link",
                ],
            },
        ]
    else:
        validator = JewelryRulesetValidator()
        default_rulesets = validator.get_all_rulesets_summary()

    # Jewelry type selector
    selected_type = st.selectbox(
        "Select Jewelry Type",
        options=[r["type"] for r in default_rulesets],
        format_func=lambda x: next((r["display_name"] for r in default_rulesets if r["type"] == x), x)
    )

    # Find selected ruleset
    ruleset = next((r for r in default_rulesets if r["type"] == selected_type), None)

    if ruleset:
        st.markdown(f"""
            <div class="ruleset-card">
                <div class="ruleset-header">{ruleset['display_name']}</div>
                <p style="color: #6B6B7B; font-size: 12px;">{ruleset['description']}</p>
            </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Acceptable Positions")
            for pos in ruleset.get("acceptable_positions", []):
                preferred_class = "position-preferred" if pos.get("preferred") else ""
                preferred_badge = ' <span style="color: #16A34A; font-size: 10px; font-weight: 700;">PREFERRED</span>' if pos.get("preferred") else ""
                st.markdown(f"""
                    <div class="position-acceptable {preferred_class}">
                        <strong style="color: #16A34A; font-size: 12px;">{pos['name']}</strong>{preferred_badge}
                        <p style="color: #6B6B7B; margin: 4px 0 0 0; font-size: 11px;">
                            {pos['description']}
                        </p>
                    </div>
                """, unsafe_allow_html=True)

        with col2:
            st.markdown("#### Forbidden Positions")
            for pos in ruleset.get("forbidden_positions", []):
                st.markdown(f"""
                    <div class="position-forbidden">
                        <strong style="color: #DC2626; font-size: 12px;">{pos['name']}</strong>
                        <p style="color: #6B6B7B; margin: 4px 0 0 0; font-size: 11px;">
                            {pos['description']}
                        </p>
                        <p style="color: #DC2626; margin: 4px 0 0 0; font-size: 11px; font-weight: 600;">
                            {pos.get('reason', 'Not allowed')}
                        </p>
                    </div>
                """, unsafe_allow_html=True)

        st.markdown("#### Special Rules")
        for rule in ruleset.get("special_rules", []):
            st.markdown(f'<span class="rule-badge">{rule}</span>', unsafe_allow_html=True)

        if ruleset.get("common_issues"):
            st.markdown("#### Common Issues")
            for issue in ruleset.get("common_issues", []):
                st.markdown(f"- {issue.replace('_', ' ').title()}")

    # Add/Edit Rules section
    st.markdown("---")
    st.markdown("### Add Custom Rule")

    with st.expander("Add New Rule for Jewelry Type"):
        new_jewelry_type = st.text_input("Jewelry Type Name")
        new_rule_description = st.text_area("Rule Description")
        new_rule_position = st.selectbox("Position Type", ["Acceptable", "Forbidden"])
        new_position_name = st.text_input("Position Name")
        new_position_desc = st.text_area("Position Description")

        if st.button("Add Rule"):
            st.success(f"Rule added for {new_jewelry_type} (Note: This is a demo - rules are not persisted)")


def render_error_categories_tab():
    """Render the Error Categories reference tab."""
    st.markdown('<p class="section-label">Error Categories Reference</p>', unsafe_allow_html=True)
    st.markdown("Complete reference of all error categories used in QC validation.")

    error_categories = {
        "Image Quality Errors": [
            ("image_blurred", "Image is blurred or out of focus", "MAJOR", "Retake with steady camera and proper focus"),
            ("image_overexposed", "Image too bright/washed out", "MAJOR", "Reduce lighting or adjust exposure"),
            ("image_underexposed", "Image too dark", "MAJOR", "Increase lighting"),
            ("image_reflection", "Specular reflection on metal", "MINOR", "Use diffused lighting or change angle"),
            ("image_low_resolution", "Resolution too low for OCR", "MAJOR", "Use higher resolution or move closer"),
        ],
        "OCR/Detection Errors": [
            ("low_confidence", "OCR confidence below threshold", "MAJOR", "Retake image with better conditions"),
            ("partial_detection", "Only partial hallmark detected", "MAJOR", "Ensure full hallmark is visible"),
            ("ocr_misread", "Characters incorrectly read", "MAJOR", "Verify manually or retake"),
            ("no_detection", "No text detected", "CRITICAL", "Check if hallmark exists and is visible"),
        ],
        "Hallmark Content Errors": [
            ("invalid_purity_code", "Purity code not BIS standard", "CRITICAL", "Valid: 375, 585, 750, 875, 916, 958, 999"),
            ("invalid_huid_format", "HUID format incorrect", "CRITICAL", "Must be 6 alphanumeric with 1+ letter"),
            ("missing_huid", "HUID not present", "CRITICAL", "HUID mandatory since April 2023"),
            ("missing_purity_mark", "Purity mark not found", "CRITICAL", "Purity code required on all items"),
            ("missing_bis_logo", "BIS logo not detected", "MAJOR", "BIS certification mark required"),
        ],
        "Position/Placement Errors": [
            ("marking_at_6_oclock", "Ring marked at bottom (6 o'clock)", "CRITICAL", "Use 12, 3, or 9 o'clock position"),
            ("wrong_position", "Marking in unacceptable area", "MAJOR", "Refer to jewelry-specific rules"),
            ("marking_on_decorative", "Marking on design area", "MAJOR", "Place on non-decorative surface"),
        ],
        "Engraving Quality Errors": [
            ("shallow_engraving", "Engraving too shallow", "MAJOR", "Must be deep enough for longevity"),
            ("uneven_engraving", "Inconsistent engraving depth", "MINOR", "Check engraving equipment"),
            ("smudged_marking", "Marking is smudged/unclear", "MAJOR", "Requires re-engraving"),
        ],
    }

    for category, errors in error_categories.items():
        st.markdown(f"#### {category}")

        for error_code, description, severity, suggestion in errors:
            severity_color = {
                "CRITICAL": "#DC2626",
                "MAJOR": "#D97706",
                "MINOR": "#C8A44E"
            }.get(severity, "#6B6B7B")

            st.markdown(f"""
                <div style="background: #FFFFFF; padding: 12px 16px; border-radius: 8px; margin: 4px 0; border: 1px solid #E8E2E4; border-left: 3px solid {severity_color};">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <code style="color: #7B1F3A; font-size: 11px; font-weight: 600;">{error_code}</code>
                        <span style="background: {severity_color}15; color: {severity_color}; padding: 2px 8px; border-radius: 10px; font-size: 9px; font-weight: 700; letter-spacing: 0.5px;">{severity}</span>
                    </div>
                    <p style="color: #2D2D2D; margin: 6px 0 0 0; font-size: 12px;">{description}</p>
                    <p style="color: #16A34A; font-size: 11px; margin: 4px 0 0 0; font-weight: 500;">Suggestion: {suggestion}</p>
                </div>
            """, unsafe_allow_html=True)


def render_confidence_benchmarks_tab():
    """Render confidence benchmarks information."""
    st.markdown('<p class="section-label">Confidence Score Benchmarks</p>', unsafe_allow_html=True)
    st.markdown("Understanding confidence scores and thresholds.")

    st.markdown("#### Decision Thresholds")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
            <div style="background: #FFFFFF; border: 1px solid #E8E2E4; border-top: 3px solid #16A34A; padding: 18px; border-radius: 10px; text-align: center;">
                <div style="font-size: 2rem; font-weight: 800; color: #16A34A;">&ge; 85%</div>
                <div style="color: #2D2D2D; margin-top: 6px; font-weight: 700; font-size: 12px;">AUTO-APPROVE</div>
                <div style="color: #9B9BAB; font-size: 11px; margin-top: 4px;">No human review needed</div>
            </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
            <div style="background: #FFFFFF; border: 1px solid #E8E2E4; border-top: 3px solid #D97706; padding: 18px; border-radius: 10px; text-align: center;">
                <div style="font-size: 2rem; font-weight: 800; color: #D97706;">50-85%</div>
                <div style="color: #2D2D2D; margin-top: 6px; font-weight: 700; font-size: 12px;">MANUAL REVIEW</div>
                <div style="color: #9B9BAB; font-size: 11px; margin-top: 4px;">QC personnel verification</div>
            </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
            <div style="background: #FFFFFF; border: 1px solid #E8E2E4; border-top: 3px solid #DC2626; padding: 18px; border-radius: 10px; text-align: center;">
                <div style="font-size: 2rem; font-weight: 800; color: #DC2626;">&lt; 50%</div>
                <div style="color: #2D2D2D; margin-top: 6px; font-weight: 700; font-size: 12px;">AUTO-REJECT</div>
                <div style="color: #9B9BAB; font-size: 11px; margin-top: 4px;">Requires re-capture</div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("#### Score Interpretation")

    score_ranges = [
        ("Excellent", 95, 100, "#16A34A", "Hallmark clearly readable, all components detected with high accuracy"),
        ("Good", 85, 95, "#16A34A", "Reliable detection, suitable for auto-approval"),
        ("Acceptable", 70, 85, "#D97706", "Readable but some uncertainty, review recommended"),
        ("Poor", 50, 70, "#D97706", "Significant uncertainty, manual verification required"),
        ("Unacceptable", 0, 50, "#DC2626", "Cannot reliably read hallmark, retake image"),
    ]

    for label, low, high, color, description in score_ranges:
        st.markdown(f"""
            <div style="display: flex; align-items: center; padding: 12px 16px; background: #FFFFFF; border-radius: 8px; margin: 4px 0; border: 1px solid #E8E2E4; border-left: 3px solid {color};">
                <div style="width: 120px;">
                    <span style="color: {color}; font-weight: 700; font-size: 12px;">{label}</span>
                    <div style="color: #9B9BAB; font-size: 10px;">{low}% - {high}%</div>
                </div>
                <div style="flex: 1; color: #6B6B7B; font-size: 12px;">{description}</div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("#### Component-Specific Thresholds")

    components = [
        ("Purity Mark", 80, "916, 750, etc."),
        ("HUID", 80, "6-char alphanumeric"),
        ("BIS Logo", 70, "Triangle mark"),
    ]

    for comp_name, threshold, description in components:
        st.markdown(f"""
            <div style="background: #FFFFFF; padding: 12px 16px; border-radius: 8px; margin: 4px 0; border: 1px solid #E8E2E4;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="color: #7B1F3A; font-weight: 700; font-size: 12px;">{comp_name}</span>
                    <span style="color: #2D2D2D; font-weight: 600; font-size: 12px;">Min: {threshold}%</span>
                </div>
                <div style="color: #9B9BAB; font-size: 11px; margin-top: 4px;">{description}</div>
            </div>
        """, unsafe_allow_html=True)


def render_huid_validator_tab():
    """Render the HUID Validator tab."""
    st.markdown('<p class="section-label">HUID Format Validator</p>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])

    with col1:
        huid_input = st.text_input("Enter HUID", placeholder="e.g., AB1234", max_chars=10)

        if st.button("Validate HUID", use_container_width=True):
            if huid_input:
                try:
                    response = requests.post(f"{API_BASE_URL}/validate/huid", data={"huid": huid_input})
                    if response.status_code == 200:
                        st.session_state.huid_result = response.json()
                except Exception as e:
                    st.error(f"Error: {str(e)}")

    with col2:
        if "huid_result" in st.session_state and st.session_state.huid_result:
            result = st.session_state.huid_result
            if result.get("valid"):
                st.success(f"Valid HUID: {result.get('cleaned')}")
            else:
                st.error("Invalid HUID")
                for error in result.get("errors", []):
                    if error:
                        st.warning(error)

    st.markdown("---")
    st.markdown("#### HUID Requirements")
    st.markdown("""
    - **Length**: Exactly 6 characters
    - **Characters**: A-Z, 0-9
    - **Must have**: At least one letter
    - **Required since**: April 2023
    """)


def render_rules_tab():
    """Render the BIS Rules tab."""
    st.markdown('<p class="section-label">BIS Compliance Rules</p>', unsafe_allow_html=True)

    try:
        response = requests.get(f"{API_BASE_URL}/qc/rules")
        if response.status_code == 200:
            rules = response.json()

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### Gold Purity Grades")
                gold_grades = rules.get("bis_standards", {}).get("gold_grades", {})
                for code, info in gold_grades.items():
                    st.markdown(f"**{code}** ({info.get('karat')}) - {info.get('purity')}%")

            with col2:
                st.markdown("#### Silver Purity Grades")
                silver_grades = rules.get("bis_standards", {}).get("silver_grades", {})
                for code, info in silver_grades.items():
                    st.markdown(f"**{code}** ({info.get('grade')}) - {info.get('purity')}%")

    except Exception as e:
        st.error(f"Failed to load rules: {str(e)}")


def render_override_tab():
    """Render the QC Override tab."""
    st.markdown('<p class="section-label">QC Override</p>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])

    with col1:
        job_id = st.text_input("Job ID", placeholder="HM-123456789")
        override_decision = st.selectbox("Override Decision", ["approved", "rejected"])
        override_reason = st.text_area("Reason for Override")
        operator_id = st.text_input("Operator ID", placeholder="QC-001")
        notes = st.text_area("Additional Notes")

        if st.button("Apply Override", use_container_width=True):
            if job_id and override_reason and operator_id:
                try:
                    response = requests.post(
                        f"{API_BASE_URL}/qc/override",
                        json={
                            "job_id": job_id,
                            "override_decision": override_decision,
                            "override_reason": override_reason,
                            "operator_id": operator_id,
                            "notes": notes
                        }
                    )
                    if response.status_code == 200:
                        st.success("Override applied!")
                        st.json(response.json())
                except Exception as e:
                    st.error(f"Error: {str(e)}")

    with col2:
        st.markdown("#### Override Guidelines")
        st.markdown("""
        **When to Override:**
        - AI misread clear engraving
        - Unusual font not recognized
        - Image quality issue but readable

        **Required Information:**
        - Valid Job ID
        - Clear reason
        - Operator ID
        """)


def render_sidebar():
    """Render the sidebar."""
    with st.sidebar:
        st.markdown("""
        <div style="padding: 4px 0 12px 0;">
            <div style="display: flex; align-items: center; gap: 10px;">
                <div style="width: 32px; height: 32px; background: rgba(123, 31, 58, 0.08); border-radius: 8px; display: flex; align-items: center; justify-content: center; color: #7B1F3A; font-weight: 800; font-size: 13px;">B</div>
                <div>
                    <div style="font-size: 13px; font-weight: 800; color: #7B1F3A;">QC Dashboard</div>
                    <div style="font-size: 9px; color: #9B9BAB; text-transform: uppercase; letter-spacing: 1px;">Hallmark Validation</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("---")

        api_online = check_api_health()
        status_class = "status-online" if api_online else "status-offline"
        status_text = "Online" if api_online else "Offline"

        st.markdown(f"""
            <div style="display: flex; align-items: center; margin-bottom: 10px;">
                <span class="status-dot {status_class}"></span>
                <span style="color: #2D2D2D; font-size: 12px; font-weight: 600;">API: {status_text}</span>
            </div>
        """, unsafe_allow_html=True)

        st.markdown(f"[API Documentation]({API_BASE_URL}/docs)")

        st.markdown("---")
        st.markdown('<p style="font-size: 10px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.8px; color: #7B1F3A;">Features</p>', unsafe_allow_html=True)
        features = [
            "Hallmark validation",
            "Error categorization",
            "Jewelry rulesets",
            "Confidence benchmarks",
            "QC override",
        ]
        for f in features:
            st.markdown(f'<p style="font-size: 12px; color: #6B6B7B; margin: 4px 0;">&#10003; {f}</p>', unsafe_allow_html=True)


def main():
    render_sidebar()

    # Top bar
    st.markdown("""
    <div class="top-bar">
        <div class="top-bar-brand">
            <div class="top-bar-icon">B</div>
            <div>
                <div class="top-bar-title">BAC QC Dashboard</div>
                <div class="top-bar-sub">Hallmark Quality Control</div>
            </div>
        </div>
        <div style="font-size: 11px; font-weight: 500; opacity: 0.8; letter-spacing: 0.5px;">Jewelry Validation</div>
    </div>
    """, unsafe_allow_html=True)

    # Page header
    st.markdown("""
    <div class="page-header">
        <div>
            <h1 class="page-title">Hallmark QC Dashboard</h1>
            <div class="page-subtitle">Jewelry Hallmarking Quality Control</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "Validate",
        "Jewelry Rules",
        "Error Categories",
        "Benchmarks",
        "HUID",
        "BIS Rules",
        "Override"
    ])

    with tab1:
        render_hallmark_tab()
    with tab2:
        render_jewelry_rulesets_tab()
    with tab3:
        render_error_categories_tab()
    with tab4:
        render_confidence_benchmarks_tab()
    with tab5:
        render_huid_validator_tab()
    with tab6:
        render_rules_tab()
    with tab7:
        render_override_tab()


if __name__ == "__main__":
    main()
