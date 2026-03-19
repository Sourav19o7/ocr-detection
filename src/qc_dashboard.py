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
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Lexend:wght@300;400;500;600;700&display=swap');

    * {
        font-family: 'Lexend', sans-serif !important;
    }

    .stApp {
        background: linear-gradient(160deg, #0a0a0a 0%, #1a1a2e 100%);
    }

    /* Header styling */
    .main-header {
        color: #ffffff;
        font-size: 2.5rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 0.5rem;
        letter-spacing: -1px;
    }

    .sub-header {
        color: #fca311;
        font-size: 1rem;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 300;
        letter-spacing: 3px;
        text-transform: uppercase;
    }

    /* Card styling */
    .result-card {
        background: linear-gradient(145deg, #1a2744 0%, #14213d 100%);
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid rgba(252, 163, 17, 0.2);
    }

    .approved-card {
        background: linear-gradient(145deg, #1a4d2e 0%, #14213d 100%);
        border: 1px solid rgba(149, 213, 178, 0.3);
    }

    .rejected-card {
        background: linear-gradient(145deg, #4d1a1a 0%, #14213d 100%);
        border: 1px solid rgba(245, 165, 165, 0.3);
    }

    .review-card {
        background: linear-gradient(145deg, #4d3a1a 0%, #14213d 100%);
        border: 1px solid rgba(252, 163, 17, 0.3);
    }

    /* Error cards */
    .error-critical {
        background: linear-gradient(145deg, #4d1a1a 0%, #2d1515 100%);
        border-left: 4px solid #e74c3c;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 8px;
    }

    .error-major {
        background: linear-gradient(145deg, #4d3a1a 0%, #2d2515 100%);
        border-left: 4px solid #e67e22;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 8px;
    }

    .error-minor {
        background: linear-gradient(145deg, #3a3a1a 0%, #252515 100%);
        border-left: 4px solid #f1c40f;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 8px;
    }

    /* Decision badges */
    .decision-badge {
        display: inline-block;
        padding: 0.5rem 1.5rem;
        border-radius: 25px;
        font-size: 1rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .badge-approved {
        background: linear-gradient(135deg, #1a4d2e 0%, #2d6a4f 100%);
        color: #95d5b2;
    }

    .badge-rejected {
        background: linear-gradient(135deg, #4d1a1a 0%, #6a2d2d 100%);
        color: #f5a5a5;
    }

    .badge-review {
        background: linear-gradient(135deg, #4d3a1a 0%, #6a4d08 100%);
        color: #fca311;
    }

    /* Metric cards */
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }

    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #fca311;
    }

    .metric-label {
        font-size: 0.8rem;
        color: rgba(255, 255, 255, 0.6);
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* Ruleset cards */
    .ruleset-card {
        background: linear-gradient(145deg, #1a2744 0%, #14213d 100%);
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid rgba(252, 163, 17, 0.2);
    }

    .ruleset-header {
        color: #fca311;
        font-size: 1.2rem;
        font-weight: 600;
        margin-bottom: 1rem;
    }

    .position-acceptable {
        background: rgba(46, 204, 113, 0.1);
        border: 1px solid rgba(46, 204, 113, 0.3);
        padding: 0.5rem 1rem;
        border-radius: 8px;
        margin: 0.25rem 0;
    }

    .position-forbidden {
        background: rgba(231, 76, 60, 0.1);
        border: 1px solid rgba(231, 76, 60, 0.3);
        padding: 0.5rem 1rem;
        border-radius: 8px;
        margin: 0.25rem 0;
    }

    .position-preferred {
        border: 2px solid #2ecc71 !important;
    }

    /* Confidence gauge */
    .confidence-gauge {
        width: 100%;
        height: 20px;
        background: #1a1a2e;
        border-radius: 10px;
        overflow: hidden;
        margin: 0.5rem 0;
    }

    .confidence-fill {
        height: 100%;
        border-radius: 10px;
        transition: width 0.5s ease;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0a0a0a 0%, #1a1a2e 100%);
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(255, 255, 255, 0.03);
        border-radius: 16px;
        padding: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        height: 45px;
        background: transparent;
        border-radius: 12px;
        color: rgba(255, 255, 255, 0.5);
        font-weight: 500;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #fca311 0%, #e09000 100%) !important;
        color: #000000 !important;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #fca311 0%, #e09000 100%);
        color: #000000;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-weight: 600;
    }

    .stButton > button:hover {
        background: linear-gradient(135deg, #ffb733 0%, #fca311 100%);
        transform: translateY(-2px);
    }

    /* Status indicators */
    .status-dot {
        display: inline-block;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        margin-right: 8px;
    }

    .status-online { background: #95d5b2; }
    .status-offline { background: #f5a5a5; }

    /* Special rule badges */
    .rule-badge {
        display: inline-block;
        background: rgba(252, 163, 17, 0.2);
        color: #fca311;
        padding: 0.25rem 0.75rem;
        border-radius: 15px;
        font-size: 0.75rem;
        margin: 0.25rem;
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
        return "#2ecc71"
    elif score >= 0.70:
        return "#f1c40f"
    elif score >= 0.50:
        return "#e67e22"
    else:
        return "#e74c3c"


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
        <div style="margin: 1rem 0;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 0.25rem;">
                <span style="color: rgba(255,255,255,0.6); font-size: 0.8rem;">{label}</span>
                <span style="color: {color}; font-weight: 600;">{percentage:.1f}%</span>
            </div>
            <div class="confidence-gauge">
                <div class="confidence-fill" style="width: {percentage}%; background: {color};"></div>
            </div>
        </div>
    """, unsafe_allow_html=True)


def render_error_card(error_type, message, suggestion=None, severity="major"):
    """Render an error card with styling based on severity."""
    severity_class = f"error-{severity}"
    severity_icon = {
        "critical": "🚫",
        "major": "⚠️",
        "minor": "ℹ️"
    }.get(severity, "⚠️")

    suggestion_html = f"<p style='color: #95d5b2; font-size: 0.85rem; margin-top: 0.5rem;'>💡 {suggestion}</p>" if suggestion else ""

    st.markdown(f"""
        <div class="{severity_class}">
            <div style="display: flex; align-items: center; gap: 0.5rem;">
                <span style="font-size: 1.2rem;">{severity_icon}</span>
                <span style="color: #ffffff; font-weight: 600;">{error_type}</span>
            </div>
            <p style="color: rgba(255,255,255,0.8); margin: 0.5rem 0 0 0; font-size: 0.9rem;">{message}</p>
            {suggestion_html}
        </div>
    """, unsafe_allow_html=True)


def render_hallmark_tab():
    """Render the Hallmark Validation tab with error categorization."""
    st.markdown("### 🔍 Hallmark Validation")
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

            if st.button("🔍 Validate Hallmark", use_container_width=True, key="validate_btn"):
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
                "excellent": (0.95, 1.0, "#2ecc71"),
                "good": (0.85, 0.95, "#27ae60"),
                "acceptable": (0.70, 0.85, "#f1c40f"),
                "poor": (0.50, 0.70, "#e67e22"),
                "unacceptable": (0.0, 0.50, "#e74c3c"),
            }

            for label, (low, high, color) in benchmark_info.items():
                if low <= confidence <= high:
                    st.markdown(f"""
                        <div style="background: rgba(255,255,255,0.05); padding: 0.5rem 1rem; border-radius: 8px; margin: 0.5rem 0;">
                            <span style="color: {color}; font-weight: 600;">Score Interpretation: {label.upper()}</span>
                            <span style="color: rgba(255,255,255,0.6); font-size: 0.8rem; margin-left: 1rem;">
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
                    st.metric("Purity Code", hallmark_data.get("purity_code") or "—")
                    st.metric("Karat", hallmark_data.get("karat") or "—")
                with col_b:
                    purity_pct = hallmark_data.get("purity_percentage")
                    st.metric("Purity %", f"{purity_pct}%" if purity_pct else "—")
                    st.metric("HUID", hallmark_data.get("huid") or "—")

            # Validation status
            status = data.get("validation_status", {})
            st.markdown("##### Validation Checklist")
            checks = [
                ("Purity Valid", status.get("purity_valid", False)),
                ("HUID Valid", status.get("huid_valid", False)),
                ("BIS Certified", data.get("bis_certified", False)),
            ]
            for check_name, check_status in checks:
                icon = "✅" if check_status else "❌"
                color = "#2ecc71" if check_status else "#e74c3c"
                st.markdown(f"<span style='color: {color};'>{icon} {check_name}</span>", unsafe_allow_html=True)

            # Error analysis section
            rejection_info = data.get("rejection_info")
            if rejection_info or decision != "approved":
                st.markdown("##### ⚠️ Error Analysis")

                # Categorize errors
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
    st.markdown("### 📏 Jewelry-Specific Rulesets")
    st.markdown("View and understand marking rules for different jewelry types.")

    if not RULESETS_AVAILABLE:
        st.warning("Jewelry rulesets module not loaded. Using default rules.")

        # Show default rules
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
                <div class="ruleset-header">📿 {ruleset['display_name']}</div>
                <p style="color: rgba(255,255,255,0.7);">{ruleset['description']}</p>
            </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### ✅ Acceptable Positions")
            for pos in ruleset.get("acceptable_positions", []):
                preferred_class = "position-preferred" if pos.get("preferred") else ""
                preferred_badge = " ⭐ PREFERRED" if pos.get("preferred") else ""
                st.markdown(f"""
                    <div class="position-acceptable {preferred_class}">
                        <strong style="color: #2ecc71;">{pos['name']}</strong>{preferred_badge}
                        <p style="color: rgba(255,255,255,0.7); margin: 0.25rem 0 0 0; font-size: 0.85rem;">
                            {pos['description']}
                        </p>
                    </div>
                """, unsafe_allow_html=True)

        with col2:
            st.markdown("#### ❌ Forbidden Positions")
            for pos in ruleset.get("forbidden_positions", []):
                st.markdown(f"""
                    <div class="position-forbidden">
                        <strong style="color: #e74c3c;">{pos['name']}</strong>
                        <p style="color: rgba(255,255,255,0.7); margin: 0.25rem 0 0 0; font-size: 0.85rem;">
                            {pos['description']}
                        </p>
                        <p style="color: #f5a5a5; margin: 0.25rem 0 0 0; font-size: 0.8rem;">
                            ⚠️ {pos.get('reason', 'Not allowed')}
                        </p>
                    </div>
                """, unsafe_allow_html=True)

        st.markdown("#### 📋 Special Rules")
        for rule in ruleset.get("special_rules", []):
            st.markdown(f'<span class="rule-badge">{rule}</span>', unsafe_allow_html=True)

        if ruleset.get("common_issues"):
            st.markdown("#### ⚠️ Common Issues")
            for issue in ruleset.get("common_issues", []):
                st.markdown(f"• {issue.replace('_', ' ').title()}")

    # Add/Edit Rules section
    st.markdown("---")
    st.markdown("### ➕ Add Custom Rule")

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
    st.markdown("### 📊 Error Categories Reference")
    st.markdown("Complete reference of all error categories used in QC validation.")

    # Error categories
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
                "CRITICAL": "#e74c3c",
                "MAJOR": "#e67e22",
                "MINOR": "#f1c40f"
            }.get(severity, "#ffffff")

            st.markdown(f"""
                <div style="background: rgba(255,255,255,0.03); padding: 1rem; border-radius: 8px; margin: 0.5rem 0; border-left: 3px solid {severity_color};">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <code style="color: #fca311;">{error_code}</code>
                        <span style="background: {severity_color}22; color: {severity_color}; padding: 0.25rem 0.5rem; border-radius: 4px; font-size: 0.75rem;">{severity}</span>
                    </div>
                    <p style="color: rgba(255,255,255,0.8); margin: 0.5rem 0 0 0;">{description}</p>
                    <p style="color: #95d5b2; font-size: 0.85rem; margin: 0.25rem 0 0 0;">💡 {suggestion}</p>
                </div>
            """, unsafe_allow_html=True)


def render_confidence_benchmarks_tab():
    """Render confidence benchmarks information."""
    st.markdown("### 📈 Confidence Score Benchmarks")
    st.markdown("Understanding confidence scores and thresholds.")

    # Thresholds visualization
    st.markdown("#### Decision Thresholds")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
            <div style="background: linear-gradient(135deg, #1a4d2e 0%, #2d6a4f 100%); padding: 1.5rem; border-radius: 12px; text-align: center;">
                <div style="font-size: 2rem; font-weight: 700; color: #95d5b2;">≥ 85%</div>
                <div style="color: rgba(255,255,255,0.8); margin-top: 0.5rem;">AUTO-APPROVE</div>
                <div style="color: rgba(255,255,255,0.5); font-size: 0.8rem; margin-top: 0.25rem;">No human review needed</div>
            </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
            <div style="background: linear-gradient(135deg, #4d3a1a 0%, #6a4d08 100%); padding: 1.5rem; border-radius: 12px; text-align: center;">
                <div style="font-size: 2rem; font-weight: 700; color: #fca311;">50-85%</div>
                <div style="color: rgba(255,255,255,0.8); margin-top: 0.5rem;">MANUAL REVIEW</div>
                <div style="color: rgba(255,255,255,0.5); font-size: 0.8rem; margin-top: 0.25rem;">QC personnel verification</div>
            </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
            <div style="background: linear-gradient(135deg, #4d1a1a 0%, #6a2d2d 100%); padding: 1.5rem; border-radius: 12px; text-align: center;">
                <div style="font-size: 2rem; font-weight: 700; color: #f5a5a5;">< 50%</div>
                <div style="color: rgba(255,255,255,0.8); margin-top: 0.5rem;">AUTO-REJECT</div>
                <div style="color: rgba(255,255,255,0.5); font-size: 0.8rem; margin-top: 0.25rem;">Requires re-capture</div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Score interpretation
    st.markdown("#### Score Interpretation")

    score_ranges = [
        ("Excellent", 95, 100, "#2ecc71", "Hallmark clearly readable, all components detected with high accuracy"),
        ("Good", 85, 95, "#27ae60", "Reliable detection, suitable for auto-approval"),
        ("Acceptable", 70, 85, "#f1c40f", "Readable but some uncertainty, review recommended"),
        ("Poor", 50, 70, "#e67e22", "Significant uncertainty, manual verification required"),
        ("Unacceptable", 0, 50, "#e74c3c", "Cannot reliably read hallmark, retake image"),
    ]

    for label, low, high, color, description in score_ranges:
        st.markdown(f"""
            <div style="display: flex; align-items: center; padding: 1rem; background: rgba(255,255,255,0.03); border-radius: 8px; margin: 0.5rem 0; border-left: 4px solid {color};">
                <div style="width: 120px;">
                    <span style="color: {color}; font-weight: 600; font-size: 1.1rem;">{label}</span>
                    <div style="color: rgba(255,255,255,0.5); font-size: 0.8rem;">{low}% - {high}%</div>
                </div>
                <div style="flex: 1; color: rgba(255,255,255,0.7);">{description}</div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Component thresholds
    st.markdown("#### Component-Specific Thresholds")

    components = [
        ("Purity Mark", 80, "916, 750, etc."),
        ("HUID", 80, "6-char alphanumeric"),
        ("BIS Logo", 70, "Triangle mark"),
    ]

    for comp_name, threshold, description in components:
        st.markdown(f"""
            <div style="background: rgba(255,255,255,0.03); padding: 1rem; border-radius: 8px; margin: 0.5rem 0;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="color: #fca311; font-weight: 600;">{comp_name}</span>
                    <span style="color: rgba(255,255,255,0.8);">Min: {threshold}%</span>
                </div>
                <div style="color: rgba(255,255,255,0.5); font-size: 0.85rem;">{description}</div>
            </div>
        """, unsafe_allow_html=True)


def render_huid_validator_tab():
    """Render the HUID Validator tab."""
    st.markdown("### 🆔 HUID Format Validator")

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
                st.success(f"✅ Valid HUID: {result.get('cleaned')}")
            else:
                st.error("❌ Invalid HUID")
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
    st.markdown("### 📋 BIS Compliance Rules")

    try:
        response = requests.get(f"{API_BASE_URL}/qc/rules")
        if response.status_code == 200:
            rules = response.json()

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### 🥇 Gold Purity Grades")
                gold_grades = rules.get("bis_standards", {}).get("gold_grades", {})
                for code, info in gold_grades.items():
                    st.markdown(f"**{code}** ({info.get('karat')}) - {info.get('purity')}%")

            with col2:
                st.markdown("#### 🥈 Silver Purity Grades")
                silver_grades = rules.get("bis_standards", {}).get("silver_grades", {})
                for code, info in silver_grades.items():
                    st.markdown(f"**{code}** ({info.get('grade')}) - {info.get('purity')}%")

    except Exception as e:
        st.error(f"Failed to load rules: {str(e)}")


def render_override_tab():
    """Render the QC Override tab."""
    st.markdown("### 🔄 QC Override")

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
        st.markdown("## 💎 QC Dashboard")
        st.markdown("---")

        api_online = check_api_health()
        status_class = "status-online" if api_online else "status-offline"
        status_text = "Online" if api_online else "Offline"

        st.markdown(f"""
            <div style="display: flex; align-items: center; margin-bottom: 1rem;">
                <span class="status-dot {status_class}"></span>
                <span style="color: white;">API: {status_text}</span>
            </div>
        """, unsafe_allow_html=True)

        st.markdown(f"[📚 API Docs]({API_BASE_URL}/docs)")

        st.markdown("---")
        st.markdown("### Features")
        st.markdown("""
        - ✅ Hallmark validation
        - ✅ Error categorization
        - ✅ Jewelry rulesets
        - ✅ Confidence benchmarks
        - ✅ QC override
        """)


def main():
    render_sidebar()

    st.markdown('<h1 class="main-header">💎 Hallmark QC Dashboard</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Jewelry Hallmarking Quality Control</p>', unsafe_allow_html=True)

    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "🔍 Validate",
        "📏 Jewelry Rules",
        "📊 Error Categories",
        "📈 Benchmarks",
        "🆔 HUID",
        "📋 BIS Rules",
        "🔄 Override"
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
