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

    /* Info items */
    .info-item {
        display: flex;
        justify-content: space-between;
        padding: 0.75rem 0;
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
    }

    .info-label {
        color: rgba(255, 255, 255, 0.6);
    }

    .info-value {
        color: #fca311;
        font-weight: 600;
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

    /* JSON display */
    .json-display {
        background: #0d1521;
        border-radius: 12px;
        padding: 1rem;
        font-family: 'Courier New', monospace !important;
        font-size: 0.85rem;
        color: #95d5b2;
        overflow-x: auto;
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
</style>
""", unsafe_allow_html=True)


def check_api_health():
    """Check if API is running."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False


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


def render_hallmark_tab():
    """Render the Hallmark Validation tab."""
    st.markdown("### 🔍 Hallmark Validation")
    st.markdown("Upload an image to validate hallmark against BIS standards.")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("#### Upload Image")
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

        expected_purity = st.selectbox(
            "Expected Purity (optional)",
            ["", "916", "750", "585", "375", "875", "958", "999"],
            help="For cross-validation"
        )

        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)

            if st.button("🔍 Validate Hallmark", use_container_width=True, key="validate_btn"):
                with st.spinner("Validating..."):
                    try:
                        # Prepare the file
                        uploaded_file.seek(0)
                        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                        data = {"job_id": job_id}
                        if expected_purity:
                            data["expected_purity"] = expected_purity

                        response = requests.post(
                            f"{API_BASE_URL}/qc/validate/v2",
                            files=files
                        )

                        if response.status_code == 200:
                            st.session_state.validation_result = response.json()
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

            # Confidence meter
            confidence = data.get("confidence", 0) * 100
            st.markdown(f"""
                <div class="metric-card" style="margin: 1rem 0;">
                    <div class="metric-value">{confidence:.1f}%</div>
                    <div class="metric-label">Confidence Score</div>
                </div>
            """, unsafe_allow_html=True)

            # Hallmark data
            hallmark_data = data.get("hallmark_data", {})
            if hallmark_data:
                st.markdown("##### Detected Hallmark")
                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("Purity Code", hallmark_data.get("purity_code") or "—")
                    st.metric("Karat", hallmark_data.get("karat") or "—")
                with col_b:
                    st.metric("Purity %", f"{hallmark_data.get('purity_percentage') or '—'}%")
                    st.metric("HUID", hallmark_data.get("huid") or "—")

            # Validation status
            status = data.get("validation_status", {})
            st.markdown("##### Validation Status")
            st.write(f"✅ Purity Valid: {status.get('purity_valid', False)}")
            st.write(f"✅ HUID Valid: {status.get('huid_valid', False)}")
            st.write(f"✅ BIS Certified: {data.get('bis_certified', False)}")

            # Rejection reasons
            rejection_info = data.get("rejection_info")
            if rejection_info:
                st.markdown("##### ⚠️ Issues Found")
                for reason in rejection_info.get("reasons", []):
                    st.warning(reason.replace("_", " ").title())

            # Raw JSON expander
            with st.expander("View Raw Response"):
                st.json(result)
        else:
            st.info("Upload an image and click 'Validate Hallmark' to see results.")


def render_huid_validator_tab():
    """Render the HUID Validator tab."""
    st.markdown("### 🆔 HUID Format Validator")
    st.markdown("Validate HUID format against BIS requirements.")

    col1, col2 = st.columns([1, 1])

    with col1:
        huid_input = st.text_input(
            "Enter HUID",
            placeholder="e.g., AB1234",
            max_chars=10,
            help="6-character alphanumeric code"
        )

        if st.button("Validate HUID", use_container_width=True, key="huid_btn"):
            if huid_input:
                try:
                    response = requests.post(
                        f"{API_BASE_URL}/validate/huid",
                        data={"huid": huid_input}
                    )
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
                st.error(f"❌ Invalid HUID")
                for error in result.get("errors", []):
                    if error:
                        st.warning(error)

            with st.expander("Details"):
                st.json(result)

    st.markdown("---")
    st.markdown("#### HUID Requirements (BIS)")
    st.markdown("""
    - **Length**: Exactly 6 characters
    - **Characters**: Alphanumeric (A-Z, 0-9)
    - **Mandatory**: At least one letter (not pure digits)
    - **Required Since**: April 2023
    """)


def render_rules_tab():
    """Render the BIS Rules tab."""
    st.markdown("### 📋 BIS Compliance Rules")

    try:
        response = requests.get(f"{API_BASE_URL}/qc/rules")
        if response.status_code == 200:
            rules = response.json()

            # Gold grades
            st.markdown("#### 🥇 Gold Purity Grades (IS 1417)")
            gold_grades = rules.get("bis_standards", {}).get("gold_grades", {})

            gold_data = []
            for code, info in gold_grades.items():
                gold_data.append({
                    "Code": code,
                    "Karat": info.get("karat"),
                    "Purity %": info.get("purity")
                })
            st.table(gold_data)

            # Silver grades
            st.markdown("#### 🥈 Silver Purity Grades (IS 2112)")
            silver_grades = rules.get("bis_standards", {}).get("silver_grades", {})

            silver_data = []
            for code, info in silver_grades.items():
                silver_data.append({
                    "Code": code,
                    "Grade": info.get("grade"),
                    "Purity %": info.get("purity")
                })
            st.table(silver_data)

            # Validation rules
            st.markdown("#### ⚙️ Validation Rules")
            val_rules = rules.get("validation_rules", {})

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Auto-Approve", f"≥{val_rules.get('auto_approve_confidence', 0.85) * 100:.0f}%")
            with col2:
                st.metric("Auto-Reject", f"<{val_rules.get('auto_reject_confidence', 0.5) * 100:.0f}%")
            with col3:
                st.metric("Manual Review", "50-85%")

            # Rejection reasons
            st.markdown("#### ❌ Rejection Reasons")
            reasons = rules.get("rejection_reasons", [])
            cols = st.columns(2)
            for i, reason in enumerate(reasons):
                with cols[i % 2]:
                    st.markdown(f"• {reason.replace('_', ' ').title()}")

    except Exception as e:
        st.error(f"Failed to load rules: {str(e)}")


def render_override_tab():
    """Render the QC Override tab."""
    st.markdown("### 🔄 QC Override")
    st.markdown("Override AI decisions when QC personnel disagree.")

    col1, col2 = st.columns([1, 1])

    with col1:
        job_id = st.text_input("Job ID", placeholder="HM-123456789")
        override_decision = st.selectbox(
            "Override Decision",
            ["approved", "rejected"]
        )
        override_reason = st.text_area(
            "Reason for Override",
            placeholder="Explain why you're overriding the AI decision..."
        )
        operator_id = st.text_input("Operator ID", placeholder="QC-001")
        notes = st.text_area("Additional Notes", placeholder="Optional notes...")

        if st.button("Apply Override", use_container_width=True, key="override_btn"):
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
                        st.session_state.override_result = response.json()
                        st.success("Override applied successfully!")
                    else:
                        st.error(f"Error: {response.text}")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
            else:
                st.warning("Please fill in all required fields.")

    with col2:
        st.markdown("#### Override Result")
        if "override_result" in st.session_state and st.session_state.override_result:
            st.json(st.session_state.override_result)
        else:
            st.info("Submit an override to see the result.")

        st.markdown("---")
        st.markdown("#### Override Guidelines")
        st.markdown("""
        **When to Override:**
        - AI misread clear engraving
        - Unusual font/style not recognized
        - Image quality issue but readable by human

        **Required Information:**
        - Valid Job ID
        - Clear reason for override
        - Operator identification
        """)


def render_api_tester_tab():
    """Render the API Tester tab."""
    st.markdown("### 🧪 API Endpoint Tester")

    endpoint = st.selectbox(
        "Select Endpoint",
        [
            "GET /",
            "GET /health",
            "GET /qc/rules",
            "POST /validate/huid",
            "POST /extract (V1 OCR)",
            "POST /extract/v2 (Hallmark OCR)",
            "POST /qc/validate/v2"
        ]
    )

    col1, col2 = st.columns([1, 1])

    with col1:
        if endpoint.startswith("GET"):
            if st.button("Send Request", key="api_test_btn"):
                path = endpoint.split(" ")[1]
                try:
                    response = requests.get(f"{API_BASE_URL}{path}")
                    st.session_state.api_response = {
                        "status_code": response.status_code,
                        "data": response.json() if response.ok else response.text
                    }
                except Exception as e:
                    st.error(f"Error: {str(e)}")

        elif "validate/huid" in endpoint:
            huid = st.text_input("HUID Value", key="api_huid")
            if st.button("Send Request", key="api_test_btn"):
                try:
                    response = requests.post(
                        f"{API_BASE_URL}/validate/huid",
                        data={"huid": huid}
                    )
                    st.session_state.api_response = {
                        "status_code": response.status_code,
                        "data": response.json() if response.ok else response.text
                    }
                except Exception as e:
                    st.error(f"Error: {str(e)}")

        else:
            uploaded = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"], key="api_file")
            if uploaded and st.button("Send Request", key="api_test_btn"):
                path = endpoint.split(" ")[1].split(" ")[0]
                try:
                    uploaded.seek(0)
                    files = {"file": (uploaded.name, uploaded.getvalue(), uploaded.type)}
                    response = requests.post(f"{API_BASE_URL}{path}", files=files)
                    st.session_state.api_response = {
                        "status_code": response.status_code,
                        "data": response.json() if response.ok else response.text
                    }
                except Exception as e:
                    st.error(f"Error: {str(e)}")

    with col2:
        st.markdown("#### Response")
        if "api_response" in st.session_state and st.session_state.api_response:
            resp = st.session_state.api_response
            st.write(f"**Status Code:** {resp['status_code']}")
            st.json(resp["data"])
        else:
            st.info("Send a request to see the response.")


def render_sidebar():
    """Render the sidebar."""
    with st.sidebar:
        st.markdown("## 💎 QC Dashboard")
        st.markdown("---")

        # API Status
        api_online = check_api_health()
        status_class = "status-online" if api_online else "status-offline"
        status_text = "Online" if api_online else "Offline"

        st.markdown(f"""
            <div style="display: flex; align-items: center; margin-bottom: 1rem;">
                <span class="status-dot {status_class}"></span>
                <span style="color: white;">API Status: {status_text}</span>
            </div>
        """, unsafe_allow_html=True)

        st.markdown(f"**API URL:** `{API_BASE_URL}`")

        st.markdown("---")

        # Quick links
        st.markdown("### Quick Links")
        st.markdown(f"[📚 API Docs]({API_BASE_URL}/docs)")
        st.markdown(f"[🔄 ReDoc]({API_BASE_URL}/redoc)")

        st.markdown("---")

        # About
        st.markdown("### About")
        st.markdown("""
        This dashboard provides a UI for testing the Jewelry Hallmarking QC system.

        **Features:**
        - Hallmark validation
        - HUID format checking
        - BIS compliance rules
        - QC override workflow
        - API endpoint testing
        """)

        st.markdown("---")
        st.markdown("*Version 1.0.0*")


def main():
    render_sidebar()

    # Header
    st.markdown('<h1 class="main-header">💎 Hallmark QC Dashboard</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Jewelry Hallmarking Quality Control</p>', unsafe_allow_html=True)

    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🔍 Validate Hallmark",
        "🆔 HUID Validator",
        "📋 BIS Rules",
        "🔄 QC Override",
        "🧪 API Tester"
    ])

    with tab1:
        render_hallmark_tab()

    with tab2:
        render_huid_validator_tab()

    with tab3:
        render_rules_tab()

    with tab4:
        render_override_tab()

    with tab5:
        render_api_tester_tab()


if __name__ == "__main__":
    main()
