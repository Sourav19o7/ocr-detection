"""
Hallmark QC Dashboard - Modern Three-Stage Workflow.

A clean, minimal Streamlit dashboard with:
- Stage 1: Upload CSV/Excel with tag IDs and expected HUIDs
- Stage 2: Upload images for processing
- Stage 3: View and search results
- Dark/Light mode toggle
"""

import streamlit as st
import requests
import pandas as pd
from PIL import Image
import io
from datetime import datetime
import time

import os

# Get API URL from environment or use default
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

# Page config
st.set_page_config(
    page_title="Hallmark QC",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Initialize theme in session state
if "theme" not in st.session_state:
    st.session_state.theme = "light"

def toggle_theme():
    st.session_state.theme = "dark" if st.session_state.theme == "light" else "light"


# CSS with dark/light mode support
def get_styles():
    is_dark = st.session_state.theme == "dark"

    if is_dark:
        bg_primary = "#0A0A0F"
        bg_secondary = "#13131A"
        bg_card = "#1A1A24"
        bg_input = "#20202E"
        border_color = "#2A2A38"
        border_light = "#1F1F2C"
        text_primary = "#F5F5F7"
        text_secondary = "#B4B4BC"
        text_muted = "#6E6E7A"
        accent = "#6366F1"
        accent_hover = "#818CF8"
        success = "#059669"
        danger = "#DC2626"
        warning = "#D97706"
    else:
        bg_primary = "#F8F9FA"
        bg_secondary = "#FFFFFF"
        bg_card = "#FFFFFF"
        bg_input = "#F1F3F5"
        border_color = "#DEE2E6"
        border_light = "#E9ECEF"
        text_primary = "#0A0A0F"
        text_secondary = "#495057"
        text_muted = "#868E96"
        accent = "#6366F1"
        accent_hover = "#4F46E5"
        success = "#059669"
        danger = "#DC2626"
        warning = "#D97706"

    return f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Lexend:wght@300;400;500;600;700;800&display=swap');

    :root {{
        --bg-primary: {bg_primary};
        --bg-secondary: {bg_secondary};
        --bg-card: {bg_card};
        --bg-input: {bg_input};
        --border: {border_color};
        --border-light: {border_light};
        --text-primary: {text_primary};
        --text-secondary: {text_secondary};
        --text-muted: {text_muted};
        --accent: {accent};
        --accent-hover: {accent_hover};
        --success: {success};
        --danger: {danger};
        --warning: {warning};
    }}

    * {{
        font-family: 'Lexend', -apple-system, BlinkMacSystemFont, sans-serif !important;
        letter-spacing: -0.01em;
    }}

    .stApp {{
        background: var(--bg-primary) !important;
    }}

    /* Override Streamlit's main container backgrounds */
    [data-testid="stAppViewContainer"],
    [data-testid="stAppViewBlockContainer"],
    .main .block-container {{
        background: var(--bg-primary) !important;
    }}

    #MainMenu, footer, header, [data-testid="stToolbar"],
    [data-testid="stDecoration"], [data-testid="stStatusWidget"] {{
        display: none !important;
    }}

    .block-container {{
        padding: 2.5rem 4rem !important;
        max-width: 1500px;
    }}

    /* Add subtle grain texture for premium feel */
    .stApp::before {{
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        opacity: {'0.015' if not is_dark else '0.025'};
        background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 400 400' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noiseFilter'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='1.5' numOctaves='4' /%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noiseFilter)' /%3E%3C/svg%3E");
        pointer-events: none;
        z-index: 1;
    }}

    /* --- Global text color overrides for Streamlit elements --- */
    .stApp p, .stApp span, .stApp label, .stApp div,
    .stMarkdown, .stMarkdown p, .stMarkdown span,
    .stMarkdown strong, .stMarkdown em, .stMarkdown li,
    [data-testid="stText"],
    [data-testid="stCaptionContainer"] {{
        color: var(--text-primary) !important;
    }}

    /* Muted / secondary text */
    .stCaption, [data-testid="stCaptionContainer"] p,
    [data-testid="stCaptionContainer"] span {{
        color: var(--text-muted) !important;
    }}

    /* Widget labels */
    .stTextInput label, .stSelectbox label, .stMultiSelect label,
    .stNumberInput label, .stTextArea label, .stFileUploader label,
    .stRadio label, .stCheckbox label, .stDateInput label,
    [data-testid="stWidgetLabel"] label,
    [data-testid="stWidgetLabel"] p {{
        color: var(--text-primary) !important;
    }}

    /* Radio button text */
    .stRadio [role="radiogroup"] label p,
    .stRadio [role="radiogroup"] label span,
    .stRadio [role="radiogroup"] label div {{
        color: var(--text-primary) !important;
    }}

    /* Selectbox dropdown menu */
    [data-baseweb="popover"],
    [data-baseweb="menu"],
    [data-baseweb="popover"] li {{
        background: var(--bg-card) !important;
        color: var(--text-primary) !important;
    }}

    [data-baseweb="menu"] [role="option"] {{
        color: var(--text-primary) !important;
    }}

    [data-baseweb="menu"] [role="option"]:hover {{
        background: var(--bg-input) !important;
    }}

    /* Spinner text */
    .stSpinner > div > span {{
        color: var(--text-secondary) !important;
    }}

    /* Disabled button */
    .stButton > button:disabled {{
        background: var(--bg-input) !important;
        color: var(--text-muted) !important;
        box-shadow: none !important;
        opacity: 0.6;
    }}

    /* Streamlit's info/success/error/warning boxes */
    [data-testid="stAlert"] {{
        background: var(--bg-card) !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--border) !important;
    }}

    /* Progress bar track override */
    .stProgress > div > div {{
        background: var(--bg-input) !important;
    }}

    /* st.dataframe container */
    [data-testid="stDataFrame"],
    [data-testid="stDataFrame"] > div {{
        background: var(--bg-card) !important;
    }}

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 0;
        background: var(--bg-card);
        border-radius: 16px;
        padding: 8px;
        border: 1px solid var(--border);
        box-shadow: 0 1px 3px rgba(0, 0, 0, {'0.05' if not is_dark else '0.1'});
    }}

    .stTabs [data-baseweb="tab"],
    .stTabs [data-baseweb="tab"] span,
    .stTabs [data-baseweb="tab"] p,
    .stTabs [data-baseweb="tab"] div {{
        height: 48px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 14px;
        padding: 0 28px;
        border: none;
        color: var(--text-secondary) !important;
        background: transparent !important;
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
        letter-spacing: 0.01em;
    }}

    .stTabs [data-baseweb="tab"]:hover,
    .stTabs [data-baseweb="tab"]:hover span {{
        color: var(--text-primary) !important;
        background: var(--bg-input) !important;
    }}

    .stTabs [data-baseweb="tab-highlight"],
    .stTabs [data-baseweb="tab-border"] {{ display: none !important; }}

    .stTabs [aria-selected="true"],
    .stTabs [aria-selected="true"] span,
    .stTabs [aria-selected="true"] p,
    .stTabs [aria-selected="true"] div {{
        background: var(--accent) !important;
        color: #FFFFFF !important;
        box-shadow: 0 2px 8px rgba(99, 102, 241, 0.2), 0 8px 16px rgba(99, 102, 241, 0.15);
    }}

    .stTabs [aria-selected="true"]:hover,
    .stTabs [aria-selected="true"]:hover span {{
        background: var(--accent-hover) !important;
        color: #FFFFFF !important;
        box-shadow: 0 4px 12px rgba(99, 102, 241, 0.25), 0 12px 24px rgba(99, 102, 241, 0.2);
    }}

    /* Buttons */
    .stButton > button {{
        background: var(--accent) !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 12px;
        padding: 14px 32px;
        font-weight: 600;
        font-size: 14px;
        letter-spacing: 0.01em;
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 1px 3px rgba(99, 102, 241, 0.12), 0 8px 24px rgba(99, 102, 241, 0.15);
        position: relative;
        overflow: hidden;
    }}

    .stButton > button::before {{
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: linear-gradient(135deg, rgba(255,255,255,0.1) 0%, rgba(255,255,255,0) 100%);
        opacity: 0;
        transition: opacity 0.25s ease;
    }}

    .stButton > button:hover {{
        background: var(--accent-hover) !important;
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(99, 102, 241, 0.2), 0 16px 32px rgba(99, 102, 241, 0.25);
    }}

    .stButton > button:hover::before {{
        opacity: 1;
    }}

    .stButton > button:active {{
        transform: translateY(0);
        box-shadow: 0 1px 3px rgba(99, 102, 241, 0.12), 0 4px 12px rgba(99, 102, 241, 0.15);
    }}

    /* Inputs */
    .stSelectbox [data-baseweb="select"] > div,
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input {{
        background: var(--bg-input) !important;
        border: 1px solid var(--border) !important;
        border-radius: 12px !important;
        color: var(--text-primary) !important;
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
        padding: 12px 16px !important;
        font-size: 14px !important;
        font-weight: 500 !important;
    }}

    .stSelectbox [data-baseweb="select"] > div:focus-within,
    .stTextInput > div > div > input:focus,
    .stNumberInput > div > div > input:focus {{
        border-color: var(--accent) !important;
        box-shadow: 0 0 0 4px rgba(99, 102, 241, 0.08) !important;
        background: var(--bg-card) !important;
    }}

    .stTextArea textarea {{
        background: var(--bg-input) !important;
        border: 1px solid var(--border) !important;
        border-radius: 10px !important;
        color: var(--text-primary) !important;
    }}

    /* File Uploader */
    [data-testid="stFileUploader"] section {{
        background: var(--bg-card) !important;
        border: 2px dashed var(--border) !important;
        border-radius: 20px !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        padding: 24px !important;
    }}

    [data-testid="stFileUploader"] section:hover {{
        border-color: var(--accent) !important;
        background: {'rgba(99, 102, 241, 0.03)' if not is_dark else 'rgba(99, 102, 241, 0.08)'} !important;
        box-shadow: 0 8px 24px rgba(99, 102, 241, 0.1);
    }}

    /* Metrics */
    [data-testid="stMetric"] {{
        background: var(--bg-card) !important;
        border: 1px solid var(--border) !important;
        border-radius: 16px !important;
        padding: 24px !important;
        box-shadow: 0 1px 3px rgba(0, 0, 0, {'0.04' if not is_dark else '0.12'});
        transition: all 0.25s ease;
    }}

    [data-testid="stMetric"]:hover {{
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, {'0.06' if not is_dark else '0.16'});
    }}

    [data-testid="stMetricValue"] {{
        color: var(--accent) !important;
        font-weight: 700 !important;
        font-size: 32px !important;
        letter-spacing: -0.02em !important;
    }}

    [data-testid="stMetricLabel"] {{
        color: var(--text-secondary) !important;
        font-weight: 500 !important;
        font-size: 13px !important;
        letter-spacing: 0.02em !important;
        text-transform: uppercase;
    }}

    /* Expander */
    [data-testid="stExpander"] {{
        border: 1px solid var(--border) !important;
        border-radius: 12px !important;
        background: var(--bg-card) !important;
    }}

    [data-testid="stExpander"] summary {{
        background: var(--bg-input) !important;
        color: var(--text-primary) !important;
    }}

    [data-testid="stExpander"] summary span,
    [data-testid="stExpander"] summary p {{
        color: var(--text-primary) !important;
    }}

    [data-testid="stExpander"] [data-testid="stExpanderDetails"] {{
        background: var(--bg-card) !important;
    }}

    [data-testid="stExpander"] [data-testid="stExpanderDetails"] p,
    [data-testid="stExpander"] [data-testid="stExpanderDetails"] span {{
        color: var(--text-primary) !important;
    }}

    /* Custom components */
    .header-bar {{
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 20px;
        padding: 24px 32px;
        margin-bottom: 32px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        box-shadow: 0 1px 3px rgba(0, 0, 0, {'0.05' if not is_dark else '0.15'});
        position: relative;
        overflow: hidden;
    }}

    .header-bar::before {{
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 4px;
        height: 100%;
        background: linear-gradient(180deg, var(--accent) 0%, var(--accent-hover) 100%);
    }}

    .header-title {{
        font-size: 24px;
        font-weight: 700;
        color: var(--text-primary);
        letter-spacing: -0.02em;
        background: linear-gradient(135deg, var(--text-primary) 0%, var(--text-secondary) 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }}

    .header-subtitle {{
        font-size: 14px;
        color: var(--text-muted);
        margin-top: 4px;
        font-weight: 400;
        letter-spacing: 0;
    }}

    .stage-card {{
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 20px;
        padding: 32px;
        margin-bottom: 24px;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 1px 3px rgba(0, 0, 0, {'0.04' if not is_dark else '0.12'}), 0 8px 24px rgba(0, 0, 0, {'0.02' if not is_dark else '0.08'});
        position: relative;
        overflow: hidden;
    }}

    .stage-card::before {{
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: radial-gradient(circle at top right, rgba(99, 102, 241, 0.05) 0%, transparent 70%);
        pointer-events: none;
    }}

    .stage-card:hover {{
        border-color: var(--accent);
        box-shadow: 0 4px 12px rgba(99, 102, 241, 0.08), 0 16px 48px rgba(99, 102, 241, 0.12);
        transform: translateY(-2px);
    }}

    .stage-number {{
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 40px;
        height: 40px;
        background: linear-gradient(135deg, var(--accent) 0%, var(--accent-hover) 100%);
        color: #FFFFFF;
        border-radius: 12px;
        font-weight: 700;
        font-size: 16px;
        margin-right: 16px;
        box-shadow: 0 4px 12px rgba(99, 102, 241, 0.25);
    }}

    .stage-title {{
        font-size: 20px;
        font-weight: 700;
        color: var(--text-primary);
        display: inline;
        letter-spacing: -0.01em;
    }}

    .stage-desc {{
        font-size: 14px;
        color: var(--text-secondary);
        margin-top: 12px;
        margin-bottom: 24px;
        line-height: 1.6;
        font-weight: 400;
    }}

    .result-card {{
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 20px;
        margin: 12px 0;
        animation: fadeIn 0.3s ease;
    }}

    @keyframes fadeIn {{
        from {{ opacity: 0; transform: translateY(10px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}

    .badge {{
        display: inline-flex;
        align-items: center;
        padding: 8px 16px;
        border-radius: 100px;
        font-size: 11px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        backdrop-filter: blur(8px);
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
    }}

    .badge-success {{
        background: linear-gradient(135deg, rgba(5, 150, 105, 0.15) 0%, rgba(5, 150, 105, 0.08) 100%);
        color: var(--success);
        border: 1px solid rgba(5, 150, 105, 0.2);
    }}

    .badge-danger {{
        background: linear-gradient(135deg, rgba(220, 38, 38, 0.15) 0%, rgba(220, 38, 38, 0.08) 100%);
        color: var(--danger);
        border: 1px solid rgba(220, 38, 38, 0.2);
    }}

    .badge-warning {{
        background: linear-gradient(135deg, rgba(217, 119, 6, 0.15) 0%, rgba(217, 119, 6, 0.08) 100%);
        color: var(--warning);
        border: 1px solid rgba(217, 119, 6, 0.2);
    }}

    .badge-info {{
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.15) 0%, rgba(99, 102, 241, 0.08) 100%);
        color: var(--accent);
        border: 1px solid rgba(99, 102, 241, 0.2);
    }}

    .stat-grid {{
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 16px;
        margin: 20px 0;
    }}

    .stat-card {{
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        transition: all 0.2s ease;
    }}

    .stat-card:hover {{
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
    }}

    .stat-value {{
        font-size: 28px;
        font-weight: 800;
        color: var(--accent);
    }}

    .stat-label {{
        font-size: 12px;
        font-weight: 600;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-top: 4px;
    }}

    .data-row {{
        display: flex;
        justify-content: space-between;
        padding: 12px 0;
        border-bottom: 1px solid var(--border-light);
    }}

    .data-row:last-child {{ border-bottom: none; }}

    .data-label {{
        font-weight: 600;
        color: var(--text-secondary);
        font-size: 13px;
    }}

    .data-value {{
        font-weight: 600;
        color: var(--text-primary);
        font-size: 13px;
    }}

    .progress-bar {{
        height: 8px;
        background: var(--bg-input);
        border-radius: 4px;
        overflow: hidden;
        margin: 8px 0;
    }}

    .progress-fill {{
        height: 100%;
        border-radius: 4px;
        transition: width 0.5s ease;
    }}

    .empty-state {{
        text-align: center;
        padding: 60px 20px;
        color: var(--text-muted);
    }}

    .empty-icon {{
        font-size: 48px;
        margin-bottom: 16px;
        opacity: 0.3;
    }}

    .empty-title {{
        font-size: 16px;
        font-weight: 600;
        color: var(--text-secondary);
        margin-bottom: 8px;
    }}

    .empty-hint {{
        font-size: 14px;
        color: var(--text-muted);
    }}

    .theme-toggle {{
        background: var(--bg-input);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 8px 16px;
        font-size: 13px;
        font-weight: 500;
        color: var(--text-secondary);
        cursor: pointer;
        transition: all 0.2s ease;
    }}

    .theme-toggle:hover {{
        background: var(--accent);
        color: #FFFFFF;
        border-color: var(--accent);
    }}

    /* Hide default Streamlit elements we don't need */
    .stDeployButton {{ display: none !important; }}

    /* Scrollbar */
    ::-webkit-scrollbar {{ width: 8px; }}
    ::-webkit-scrollbar-track {{ background: var(--bg-primary); }}
    ::-webkit-scrollbar-thumb {{
        background: var(--border);
        border-radius: 4px;
    }}
    ::-webkit-scrollbar-thumb:hover {{ background: var(--accent); }}

    /* Table styling */
    .dataframe {{
        background: var(--bg-card) !important;
        border: 1px solid var(--border) !important;
        border-radius: 12px !important;
    }}

    .dataframe th {{
        background: var(--bg-input) !important;
        color: var(--text-primary) !important;
    }}

    .dataframe td {{
        color: var(--text-primary) !important;
        background: var(--bg-card) !important;
    }}

    /* Glide data grid (st.dataframe uses this internally) */
    [data-testid="stDataFrame"] canvas + div {{
        background: var(--bg-card) !important;
    }}

    /* Success/Error messages */
    .stSuccess, .stError, .stWarning, .stInfo {{
        border-radius: 10px !important;
    }}

    /* File uploader text */
    [data-testid="stFileUploader"] section div,
    [data-testid="stFileUploader"] section span,
    [data-testid="stFileUploader"] section small,
    [data-testid="stFileUploader"] section p {{
        color: var(--text-secondary) !important;
    }}

    [data-testid="stFileUploader"] section button {{
        color: var(--accent) !important;
    }}

    /* Uploaded file name chip */
    [data-testid="stFileUploader"] [data-testid="stMarkdown"] {{
        color: var(--text-primary) !important;
    }}

    /* Sidebar */
    [data-testid="stSidebar"] {{
        background: var(--bg-secondary) !important;
    }}

    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] label {{
        color: var(--text-primary) !important;
    }}

    /* Streamlit tab content panel */
    .stTabs [data-baseweb="tab-panel"] {{
        background: transparent !important;
    }}

    /* Horizontal rule / dividers */
    hr {{
        border-color: var(--border) !important;
    }}
    </style>
    """


def check_api_health():
    try:
        return requests.get(f"{API_BASE_URL}/health", timeout=5).status_code == 200
    except:
        return False


def render_header():
    is_dark = st.session_state.theme == "dark"
    theme_icon = "Light" if is_dark else "Dark"

    col1, col2 = st.columns([6, 1])
    with col1:
        st.markdown(f"""
        <div class="header-bar">
            <div>
                <div class="header-title">Hallmark QC System</div>
                <div class="header-subtitle">AI-powered quality control with BIS compliance</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        if st.button(f"{theme_icon} Mode", key="theme_toggle", use_container_width=True):
            toggle_theme()
            st.rerun()


def render_stage1():
    """Stage 1: Upload batch data (CSV/Excel)."""
    st.markdown("""
    <div class="stage-card">
        <span class="stage-number">1</span>
        <span class="stage-title">Upload Batch Data</span>
        <div class="stage-desc">Upload a CSV or Excel file containing tag IDs and expected HUIDs</div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])

    with col1:
        uploaded_file = st.file_uploader(
            "Drop your file here or click to browse",
            type=["csv", "xlsx", "xls"],
            key="batch_uploader",
            help="File should have columns: tag_id, expected_huid"
        )

        batch_name = st.text_input(
            "Batch Name (optional)",
            placeholder="e.g., Morning_Batch_001",
            key="batch_name_input"
        )

        if uploaded_file:
            # Preview the data
            try:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)

                st.markdown("**Preview:**")
                st.dataframe(df.head(10), use_container_width=True)
                st.caption(f"Total rows: {len(df)}")

                uploaded_file.seek(0)  # Reset file position

                if st.button("Upload Batch", key="upload_batch_btn", use_container_width=True):
                    with st.spinner("Uploading..."):
                        files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
                        data = {"batch_name": batch_name} if batch_name else {}

                        response = requests.post(
                            f"{API_BASE_URL}/stage1/upload-batch",
                            files=files,
                            data=data
                        )

                        if response.status_code == 200:
                            result = response.json()
                            st.success(f"Uploaded {result['total_items']} items in batch #{result['batch_id']}")
                            st.session_state.current_batch_id = result['batch_id']
                        else:
                            st.error(f"Error: {response.json().get('detail', 'Upload failed')}")
            except Exception as e:
                st.error(f"Error reading file: {str(e)}")

    with col2:
        st.markdown("**Recent Batches**")
        try:
            response = requests.get(f"{API_BASE_URL}/stage1/batches")
            if response.status_code == 200:
                batches = response.json().get("batches", [])
                if batches:
                    for batch in batches[:5]:
                        status_class = "badge-success" if batch.get("status") == "completed" else "badge-info"
                        st.markdown(f"""
                        <div class="result-card" style="padding: 14px;">
                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                <div>
                                    <div style="font-weight: 600; color: var(--text-primary);">#{batch['id']} {batch['batch_name']}</div>
                                    <div style="font-size: 12px; color: var(--text-muted);">{batch['total_items']} items</div>
                                </div>
                                <span class="badge {status_class}">{batch.get('status', 'pending')}</span>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.caption("No batches yet")
        except:
            st.caption("Could not load batches")


def render_ocr_detect_results(ocr_data: dict):
    """Render standalone OCR detection results."""
    hallmark = ocr_data.get("hallmark", {})
    detections = ocr_data.get("detections", [])
    avg_conf = ocr_data.get("average_confidence", 0)
    bis = hallmark.get("bis_certified", False)

    bis_class = "badge-success" if bis else "badge-warning"
    bis_text = "BIS CERTIFIED" if bis else "NOT CERTIFIED"

    # Hallmark info card
    hallmark_rows = ""
    if hallmark.get("purity_code"):
        hallmark_rows += f"""
        <div class="data-row">
            <span class="data-label">Purity Code</span>
            <span class="data-value">{hallmark['purity_code']}</span>
        </div>"""
    if hallmark.get("karat"):
        hallmark_rows += f"""
        <div class="data-row">
            <span class="data-label">Karat</span>
            <span class="data-value">{hallmark['karat']}</span>
        </div>"""
    if hallmark.get("purity_percentage"):
        hallmark_rows += f"""
        <div class="data-row">
            <span class="data-label">Purity</span>
            <span class="data-value">{hallmark['purity_percentage']}%</span>
        </div>"""
    if hallmark.get("huid"):
        hallmark_rows += f"""
        <div class="data-row">
            <span class="data-label">HUID</span>
            <span class="data-value">{hallmark['huid']}</span>
        </div>"""

    st.markdown(f"""
    <div class="result-card">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px;">
            <div style="font-size: 16px; font-weight: 700; color: var(--text-primary);">
                OCR Detection Results
            </div>
            <div style="display: flex; gap: 8px;">
                <span class="badge badge-info">{avg_conf*100:.1f}% Confidence</span>
                <span class="badge {bis_class}">{bis_text}</span>
            </div>
        </div>
        {hallmark_rows if hallmark_rows else '<div style="color: var(--text-muted); font-size: 13px;">No hallmark data detected</div>'}
    </div>
    """, unsafe_allow_html=True)

    # Individual detections
    if detections:
        with st.expander("Detected Text", expanded=True):
            for det in detections:
                conf = det["confidence"]
                conf_pct = conf * 100
                det_type = det.get("type", "unknown")
                validated = det.get("validated", False)

                if conf >= 0.85:
                    conf_color = "var(--success)"
                elif conf >= 0.50:
                    conf_color = "var(--warning)"
                else:
                    conf_color = "var(--danger)"

                type_badge = ""
                if det_type == "purity_mark":
                    type_badge = '<span class="badge badge-info" style="font-size: 10px; padding: 2px 8px;">Purity</span>'
                elif det_type == "huid":
                    type_badge = '<span class="badge badge-info" style="font-size: 10px; padding: 2px 8px;">HUID</span>'
                elif det_type == "check":
                    type_badge = '<span class="badge badge-warning" style="font-size: 10px; padding: 2px 8px;">Check</span>'

                validated_badge = ""
                if validated:
                    validated_badge = '<span class="badge badge-success" style="font-size: 10px; padding: 2px 8px;">Validated</span>'

                st.markdown(f"""
                <div style="display: flex; justify-content: space-between; align-items: center;
                            padding: 10px 0; border-bottom: 1px solid var(--border-light);">
                    <div style="display: flex; align-items: center; gap: 8px;">
                        <span style="font-weight: 600; color: var(--text-primary); font-size: 14px;">
                            {det['text']}
                        </span>
                        {type_badge}
                        {validated_badge}
                    </div>
                    <span style="font-weight: 700; color: {conf_color}; font-size: 13px;">
                        {conf_pct:.0f}%
                    </span>
                </div>
                """, unsafe_allow_html=True)

    # Full text
    full_text = ocr_data.get("full_text", "")
    if full_text:
        with st.expander("Full Extracted Text"):
            st.text_area("OCR Text", full_text, height=100, label_visibility="collapsed", key="ocr_detect_text")


def render_stage2():
    """Stage 2: Upload and process images."""
    st.markdown("""
    <div class="stage-card">
        <span class="stage-number">2</span>
        <span class="stage-title">Process Images</span>
        <div class="stage-desc">Upload hallmark images with their tag IDs for OCR processing</div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("**Single Image Upload**")

        tag_id = st.text_input("Tag ID", placeholder="Enter the tag ID", key="single_tag_id")
        uploaded_image = st.file_uploader(
            "Upload Image",
            type=["png", "jpg", "jpeg", "bmp", "webp"],
            key="single_image_uploader"
        )

        if uploaded_image:
            image = Image.open(uploaded_image)
            st.image(image, caption="Preview", use_container_width=True)

            # --- Detect OCR button (standalone, no tag_id needed) ---
            btn_col1, btn_col2 = st.columns(2)

            with btn_col1:
                detect_ocr_clicked = st.button("Detect OCR", key="detect_ocr_btn", use_container_width=True)

            with btn_col2:
                process_clicked = st.button(
                    "Process Image",
                    key="process_single_btn",
                    use_container_width=True,
                    disabled=not tag_id
                )

            # Handle Detect OCR
            if detect_ocr_clicked:
                with st.spinner("Running OCR detection..."):
                    uploaded_image.seek(0)
                    files = {"file": (uploaded_image.name, uploaded_image.getvalue(), uploaded_image.type)}

                    response = requests.post(
                        f"{API_BASE_URL}/extract/v2",
                        files=files,
                    )

                    if response.status_code == 200:
                        ocr_data = response.json()
                        st.session_state.ocr_detect_result = ocr_data
                    else:
                        st.error(f"OCR detection failed: {response.json().get('detail', 'Unknown error')}")
                        st.session_state.ocr_detect_result = None

            # Handle Process Image (requires tag_id)
            if process_clicked and tag_id:
                with st.spinner("Processing..."):
                    uploaded_image.seek(0)
                    files = {"file": (uploaded_image.name, uploaded_image.getvalue(), uploaded_image.type)}
                    data = {"tag_id": tag_id}

                    response = requests.post(
                        f"{API_BASE_URL}/stage2/upload-image",
                        files=files,
                        data=data
                    )

                    if response.status_code == 200:
                        result = response.json()
                        st.session_state.last_result = result

                        # Show result
                        match_class = "badge-success" if result['huid_match'] else "badge-danger"
                        match_text = "MATCH" if result['huid_match'] else "MISMATCH"

                        decision_raw = result.get('decision', 'pending')
                        decision_display = decision_raw.replace('_', ' ').upper()
                        decision_class = {
                            'approved': 'badge-success', 'rejected': 'badge-danger',
                            'manual_review': 'badge-warning',
                        }.get(decision_raw, 'badge-info')

                        conf_val = result.get('confidence', 0)
                        if conf_val >= 0.85:
                            conf_color = "var(--success)"
                        elif conf_val >= 0.50:
                            conf_color = "var(--warning)"
                        else:
                            conf_color = "var(--danger)"

                        st.markdown(f"""
                        <div class="result-card">
                            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px;">
                                <span class="badge {match_class}">{match_text}</span>
                                <span class="badge {decision_class}">{decision_display}</span>
                            </div>
                            <div class="data-row">
                                <span class="data-label">Expected HUID</span>
                                <span class="data-value">{result['expected_huid']}</span>
                            </div>
                            <div class="data-row">
                                <span class="data-label">Actual HUID</span>
                                <span class="data-value">{result['actual_huid'] or 'Not detected'}</span>
                            </div>
                            <div class="data-row">
                                <span class="data-label">Confidence</span>
                                <span class="data-value" style="color: {conf_color};">{conf_val*100:.1f}%</span>
                            </div>
                            <div class="progress-bar">
                                <div class="progress-fill" style="width: {conf_val*100:.0f}%; background: {conf_color};"></div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        error_detail = response.json().get('detail', 'Processing failed')
                        st.error(f"Error: {error_detail}")

            # Display OCR detection results
            if hasattr(st.session_state, 'ocr_detect_result') and st.session_state.ocr_detect_result:
                render_ocr_detect_results(st.session_state.ocr_detect_result)
        else:
            st.session_state.ocr_detect_result = None

    with col2:
        st.markdown("**Bulk Image Upload**")
        st.caption("Upload multiple images. Filename should match tag ID (e.g., TAG001.jpg)")

        bulk_files = st.file_uploader(
            "Upload Multiple Images",
            type=["png", "jpg", "jpeg", "bmp", "webp"],
            accept_multiple_files=True,
            key="bulk_image_uploader"
        )

        if bulk_files:
            st.caption(f"{len(bulk_files)} files selected")

            if st.button("Process All Images", key="process_bulk_btn", use_container_width=True):
                progress = st.progress(0)
                status_text = st.empty()

                results = []
                for i, file in enumerate(bulk_files):
                    status_text.text(f"Processing {file.name}...")

                    tag_id = file.name.rsplit('.', 1)[0]
                    files = {"file": (file.name, file.getvalue(), file.type)}
                    data = {"tag_id": tag_id}

                    try:
                        response = requests.post(
                            f"{API_BASE_URL}/stage2/upload-image",
                            files=files,
                            data=data
                        )

                        if response.status_code == 200:
                            result = response.json()
                            results.append({
                                "tag_id": tag_id,
                                "status": "success",
                                "match": result['huid_match'],
                                "decision": result['decision']
                            })
                        else:
                            results.append({
                                "tag_id": tag_id,
                                "status": "error",
                                "error": response.json().get('detail', 'Failed')
                            })
                    except Exception as e:
                        results.append({
                            "tag_id": tag_id,
                            "status": "error",
                            "error": str(e)
                        })

                    progress.progress((i + 1) / len(bulk_files))

                status_text.text("Processing complete!")

                # Summary
                success_count = len([r for r in results if r['status'] == 'success'])
                error_count = len([r for r in results if r['status'] == 'error'])
                match_count = len([r for r in results if r.get('match')])

                st.markdown(f"""
                <div class="stat-grid" style="grid-template-columns: repeat(4, 1fr);">
                    <div class="stat-card">
                        <div class="stat-value">{len(bulk_files)}</div>
                        <div class="stat-label">Total</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value" style="color: var(--success);">{success_count}</div>
                        <div class="stat-label">Processed</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value" style="color: {'var(--success)' if match_count > 0 else 'var(--danger)'};">{match_count}</div>
                        <div class="stat-label">Matches</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value" style="color: {'var(--danger)' if error_count > 0 else 'var(--success)'};">{error_count}</div>
                        <div class="stat-label">Errors</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Per-file results table
                if results:
                    df_bulk = pd.DataFrame(results)
                    if 'match' in df_bulk.columns:
                        df_bulk['match'] = df_bulk['match'].map({True: 'Match', False: 'Mismatch', None: '--'})
                    if 'decision' in df_bulk.columns:
                        df_bulk['decision'] = df_bulk['decision'].apply(
                            lambda x: (x or '--').replace('_', ' ').title() if isinstance(x, str) else '--'
                        )
                    if 'error' in df_bulk.columns:
                        df_bulk['error'] = df_bulk['error'].fillna('')
                    if 'status' in df_bulk.columns:
                        df_bulk['status'] = df_bulk['status'].apply(lambda x: x.title())

                    col_renames = {
                        'tag_id': 'Tag ID', 'status': 'Status',
                        'match': 'Match', 'decision': 'Decision', 'error': 'Error'
                    }
                    df_bulk = df_bulk.rename(columns={k: v for k, v in col_renames.items() if k in df_bulk.columns})
                    st.dataframe(df_bulk, use_container_width=True, hide_index=True)


def render_stage3():
    """Stage 3: View results."""
    st.markdown("""
    <div class="stage-card">
        <span class="stage-number">3</span>
        <span class="stage-title">View Results</span>
        <div class="stage-desc">Search and view OCR results by tag ID or batch</div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("**Search**")

        search_type = st.radio(
            "Search by",
            ["Tag ID", "Batch ID"],
            horizontal=True,
            key="search_type"
        )

        if search_type == "Tag ID":
            search_tag = st.text_input("Enter Tag ID", key="search_tag_input")

            if st.button("Search", key="search_btn", use_container_width=True):
                if search_tag:
                    with st.spinner("Searching..."):
                        response = requests.get(f"{API_BASE_URL}/stage3/result/{search_tag}")

                        if response.status_code == 200:
                            st.session_state.search_result = response.json()
                        else:
                            st.error("Tag ID not found")
                            st.session_state.search_result = None
        else:
            # Get list of batches
            try:
                batches_resp = requests.get(f"{API_BASE_URL}/stage1/batches")
                if batches_resp.status_code == 200:
                    batches = batches_resp.json().get("batches", [])
                    batch_options = {f"#{b['id']} - {b['batch_name']}": b['id'] for b in batches}

                    if batch_options:
                        selected_batch = st.selectbox(
                            "Select Batch",
                            options=list(batch_options.keys()),
                            key="batch_select"
                        )

                        if st.button("Load Results", key="load_batch_btn", use_container_width=True):
                            batch_id = batch_options[selected_batch]
                            with st.spinner("Loading..."):
                                response = requests.get(f"{API_BASE_URL}/stage3/batch/{batch_id}/results")

                                if response.status_code == 200:
                                    st.session_state.batch_results = response.json()
                                else:
                                    st.error("Failed to load batch results")
                    else:
                        st.caption("No batches available")
            except:
                st.error("Could not load batches")

    with col2:
        st.markdown("**Results**")

        # Display single result
        if hasattr(st.session_state, 'search_result') and st.session_state.search_result:
            result = st.session_state.search_result

            match_status = result.get('huid_match')
            if match_status is True:
                match_badge = '<span class="badge badge-success">MATCH</span>'
            elif match_status is False:
                match_badge = '<span class="badge badge-danger">MISMATCH</span>'
            else:
                match_badge = '<span class="badge badge-warning">PENDING</span>'

            decision = result.get('decision') or 'pending'
            decision_class = {
                'approved': 'badge-success',
                'rejected': 'badge-danger',
                'manual_review': 'badge-warning',
            }.get(decision, 'badge-info')

            # Build rejection reasons HTML
            rejection_reasons = result.get('rejection_reasons', [])
            rejection_html = ""
            if rejection_reasons:
                reason_badges = ""
                reason_labels = {
                    "missing_huid": "Missing HUID",
                    "missing_purity_mark": "Missing Purity Mark",
                    "huid_mismatch": "HUID Mismatch",
                    "low_confidence": "Low Confidence",
                    "invalid_purity_code": "Invalid Purity Code",
                    "invalid_huid_format": "Invalid HUID Format",
                    "missing_bis_logo": "Missing BIS Logo",
                    "unclear_engraving": "Unclear Engraving",
                    "ocr_mismatch": "OCR Mismatch",
                    "incomplete_hallmark": "Incomplete Hallmark",
                    "non_compliant_format": "Non-Compliant Format",
                    "purity_huid_mismatch": "Purity-HUID Mismatch",
                }
                for reason in rejection_reasons:
                    label = reason_labels.get(reason, reason.replace('_', ' ').title())
                    reason_badges += f'<span class="badge badge-danger" style="margin: 2px;">{label}</span> '
                rejection_html = f"""
                <div class="data-row" style="flex-wrap: wrap;">
                    <span class="data-label">Rejection Reasons</span>
                    <div>{reason_badges}</div>
                </div>"""

            # Processing status
            proc_status = result.get('processing_status', 'pending')
            status_class = {
                'completed': 'badge-success', 'failed': 'badge-danger',
                'processing': 'badge-info', 'manual_review': 'badge-warning',
            }.get(proc_status, 'badge-info')

            # Confidence bar color
            conf_val = result.get('confidence') or 0
            if conf_val >= 0.85:
                conf_bar_color = "var(--success)"
            elif conf_val >= 0.50:
                conf_bar_color = "var(--warning)"
            else:
                conf_bar_color = "var(--danger)"

            st.markdown(f"""
            <div class="result-card">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px;">
                    <div style="font-size: 18px; font-weight: 700; color: var(--text-primary);">
                        Tag: {result['tag_id']}
                    </div>
                    <div style="display: flex; gap: 8px;">
                        <span class="badge {status_class}">{proc_status.replace('_', ' ').upper()}</span>
                        {match_badge}
                        <span class="badge {decision_class}">{decision.replace('_', ' ').upper() if decision else 'PENDING'}</span>
                    </div>
                </div>

                <div class="data-row">
                    <span class="data-label">Expected HUID</span>
                    <span class="data-value">{result.get('expected_huid', '--')}</span>
                </div>
                <div class="data-row">
                    <span class="data-label">Actual HUID</span>
                    <span class="data-value">{result.get('actual_huid') or 'Not detected'}</span>
                </div>
                <div class="data-row">
                    <span class="data-label">Purity Code</span>
                    <span class="data-value">{result.get('purity_code') or '--'}</span>
                </div>
                <div class="data-row">
                    <span class="data-label">Karat</span>
                    <span class="data-value">{result.get('karat') or '--'}</span>
                </div>
                <div class="data-row">
                    <span class="data-label">Purity</span>
                    <span class="data-value">{f"{result['purity_percentage']}%" if result.get('purity_percentage') else '--'}</span>
                </div>
                <div class="data-row">
                    <span class="data-label">Confidence</span>
                    <span class="data-value" style="color: {conf_bar_color};">{conf_val*100:.1f}%</span>
                </div>
                <div class="progress-bar">
                    <div class="progress-fill" style="width: {conf_val*100:.0f}%; background: {conf_bar_color};"></div>
                </div>
                {rejection_html}
            </div>
            """, unsafe_allow_html=True)

            # Raw OCR text in an expander
            raw_text = result.get('raw_ocr_text')
            if raw_text:
                with st.expander("Raw OCR Text"):
                    st.text_area("OCR output", raw_text, height=80, label_visibility="collapsed", key="stage3_raw_text")

            # Show image if available
            if result.get('image_url'):
                st.image(result['image_url'], caption="Uploaded Image", use_container_width=True)

        # Display batch results
        elif hasattr(st.session_state, 'batch_results') and st.session_state.batch_results:
            batch_data = st.session_state.batch_results
            stats = batch_data.get('statistics', {})

            # Statistics
            total = batch_data.get('total_items', 0)
            processed = batch_data.get('processed_items', 0)
            matches = stats.get('huid_matches', 0)
            mismatches = stats.get('huid_mismatches', 0)
            decision_counts = stats.get('decision_counts', {})
            approved = decision_counts.get('approved', 0)
            rejected = decision_counts.get('rejected', 0)
            manual_rev = decision_counts.get('manual_review', 0)

            st.markdown(f"""
            <div class="stat-grid">
                <div class="stat-card">
                    <div class="stat-value">{total}</div>
                    <div class="stat-label">Total Items</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{processed}</div>
                    <div class="stat-label">Processed</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value" style="color: var(--success);">{matches}</div>
                    <div class="stat-label">Matches</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value" style="color: var(--danger);">{mismatches}</div>
                    <div class="stat-label">Mismatches</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Decision breakdown
            if approved or rejected or manual_rev:
                st.markdown(f"""
                <div class="stat-grid" style="grid-template-columns: repeat(3, 1fr); margin-top: 0;">
                    <div class="stat-card">
                        <div class="stat-value" style="color: var(--success);">{approved}</div>
                        <div class="stat-label">Approved</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value" style="color: var(--danger);">{rejected}</div>
                        <div class="stat-label">Rejected</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value" style="color: var(--warning);">{manual_rev}</div>
                        <div class="stat-label">Manual Review</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # Results table
            results = batch_data.get('results', [])
            if results:
                df = pd.DataFrame(results)
                # Select and rename columns safely
                cols_available = [c for c in ['tag_id', 'expected_huid', 'actual_huid', 'huid_match', 'confidence', 'decision', 'status'] if c in df.columns]
                df = df[cols_available]

                # Format values for display
                if 'huid_match' in df.columns:
                    df['huid_match'] = df['huid_match'].map({True: 'Match', False: 'Mismatch', None: 'Pending'})
                if 'confidence' in df.columns:
                    df['confidence'] = df['confidence'].apply(lambda x: f"{(x or 0)*100:.1f}%" if x is not None else '--')
                if 'decision' in df.columns:
                    df['decision'] = df['decision'].apply(lambda x: (x or 'pending').replace('_', ' ').title())
                if 'status' in df.columns:
                    df['status'] = df['status'].apply(lambda x: (x or 'pending').replace('_', ' ').title())
                if 'actual_huid' in df.columns:
                    df['actual_huid'] = df['actual_huid'].fillna('Not detected')

                col_renames = {
                    'tag_id': 'Tag ID', 'expected_huid': 'Expected HUID',
                    'actual_huid': 'Actual HUID', 'huid_match': 'Match',
                    'confidence': 'Confidence', 'decision': 'Decision', 'status': 'Status'
                }
                df = df.rename(columns={k: v for k, v in col_renames.items() if k in df.columns})

                st.dataframe(df, use_container_width=True, hide_index=True)

        else:
            st.markdown("""
            <div class="empty-state">
                <div class="empty-icon">&#128269;</div>
                <div class="empty-title">No results to display</div>
                <div class="empty-hint">Search by Tag ID or select a batch to view results</div>
            </div>
            """, unsafe_allow_html=True)


def render_erp_monitor():
    """ERP Integration Monitor: View pending items and manual review queue."""
    st.markdown("""
    <div class="stage-card">
        <span class="stage-number">E</span>
        <span class="stage-title">ERP Integration Monitor</span>
        <div class="stage-desc">Monitor items from ERP system, manage manual reviews, and view statistics</div>
    </div>
    """, unsafe_allow_html=True)

    # Statistics section
    st.markdown("**Today's Statistics**")
    try:
        response = requests.get(f"{API_BASE_URL}/api/erp/statistics")
        if response.status_code == 200:
            stats = response.json()

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Items", stats.get("total_items", 0))
            with col2:
                st.metric("Processed", stats.get("processed", 0))
            with col3:
                st.metric("Approved", stats.get("approved", 0))
            with col4:
                st.metric("Pending Review", stats.get("manual_review", 0))

            # Decision breakdown
            st.markdown(f"""
            <div class="stat-grid" style="grid-template-columns: repeat(5, 1fr); margin-top: 16px;">
                <div class="stat-card">
                    <div class="stat-value">{stats.get('huid_matches', 0)}</div>
                    <div class="stat-label">HUID Matches</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value" style="color: var(--danger);">{stats.get('huid_mismatches', 0)}</div>
                    <div class="stat-label">Mismatches</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value" style="color: var(--success);">{stats.get('approved', 0)}</div>
                    <div class="stat-label">Approved</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value" style="color: var(--danger);">{stats.get('rejected', 0)}</div>
                    <div class="stat-label">Rejected</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value" style="color: var(--warning);">{stats.get('manual_review', 0)}</div>
                    <div class="stat-label">Manual Review</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            avg_conf = stats.get("average_confidence", 0)
            if avg_conf > 0:
                st.markdown(f"**Average Confidence:** {avg_conf*100:.1f}%")

    except Exception as e:
        st.error(f"Could not load statistics: {e}")

    st.markdown("---")

    # Manual Review Queue
    st.markdown("**Manual Review Queue**")

    col1, col2 = st.columns([3, 1])

    with col2:
        decision_filter = st.selectbox(
            "Filter by",
            ["manual_review", "pending", "all"],
            key="erp_decision_filter"
        )

    try:
        params = {}
        if decision_filter != "all":
            params["decision"] = decision_filter

        response = requests.get(f"{API_BASE_URL}/api/erp/pending-items", params=params)
        if response.status_code == 200:
            data = response.json()
            items = data.get("items", [])

            if items:
                st.caption(f"Showing {len(items)} of {data.get('total', 0)} items")

                for item in items:
                    result = item.get("result", {})
                    tag_id = item.get("tag_id", "")
                    expected_huid = item.get("expected_huid", "")
                    actual_huid = result.get("actual_huid") if result else None
                    confidence = result.get("confidence", 0) if result else 0
                    current_decision = result.get("decision", "pending") if result else "pending"

                    # Decision badge
                    decision_class = {
                        'approved': 'badge-success',
                        'rejected': 'badge-danger',
                        'manual_review': 'badge-warning',
                    }.get(current_decision, 'badge-info')

                    # Match status
                    huid_match = result.get("huid_match") if result else None
                    match_badge = ""
                    if huid_match is True:
                        match_badge = '<span class="badge badge-success">MATCH</span>'
                    elif huid_match is False:
                        match_badge = '<span class="badge badge-danger">MISMATCH</span>'

                    # Confidence color
                    if confidence >= 0.85:
                        conf_color = "var(--success)"
                    elif confidence >= 0.50:
                        conf_color = "var(--warning)"
                    else:
                        conf_color = "var(--danger)"

                    with st.expander(f"**{tag_id}** - {expected_huid} | {current_decision.replace('_', ' ').upper()}", expanded=False):
                        col_img, col_details = st.columns([1, 2])

                        with col_img:
                            if item.get("image_url"):
                                try:
                                    st.image(item["image_url"], caption="Hallmark Image", use_container_width=True)
                                except:
                                    st.caption("Image not available")
                            else:
                                st.caption("No image uploaded")

                        with col_details:
                            st.markdown(f"""
                            <div class="result-card" style="padding: 16px;">
                                <div style="display: flex; gap: 8px; margin-bottom: 12px;">
                                    {match_badge}
                                    <span class="badge {decision_class}">{current_decision.replace('_', ' ').upper()}</span>
                                </div>
                                <div class="data-row">
                                    <span class="data-label">Expected HUID</span>
                                    <span class="data-value">{expected_huid}</span>
                                </div>
                                <div class="data-row">
                                    <span class="data-label">Detected HUID</span>
                                    <span class="data-value">{actual_huid or 'Not detected'}</span>
                                </div>
                                <div class="data-row">
                                    <span class="data-label">Purity Code</span>
                                    <span class="data-value">{result.get('purity_code') or '--'}</span>
                                </div>
                                <div class="data-row">
                                    <span class="data-label">Confidence</span>
                                    <span class="data-value" style="color: {conf_color};">{confidence*100:.1f}%</span>
                                </div>
                                <div class="progress-bar">
                                    <div class="progress-fill" style="width: {confidence*100:.0f}%; background: {conf_color};"></div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)

                            # Rejection reasons
                            rejection_reasons = result.get("rejection_reasons", []) if result else []
                            if rejection_reasons:
                                reason_labels = {
                                    "missing_huid": "Missing HUID",
                                    "missing_purity_mark": "Missing Purity",
                                    "huid_mismatch": "HUID Mismatch",
                                    "low_confidence": "Low Confidence",
                                }
                                reasons_text = ", ".join([reason_labels.get(r, r) for r in rejection_reasons])
                                st.warning(f"Issues: {reasons_text}")

                            # Manual decision buttons
                            if current_decision == "manual_review":
                                st.markdown("**Make Decision:**")
                                btn_col1, btn_col2 = st.columns(2)

                                with btn_col1:
                                    if st.button("Approve", key=f"approve_{tag_id}", use_container_width=True):
                                        try:
                                            resp = requests.post(
                                                f"{API_BASE_URL}/api/erp/manual-decision",
                                                data={"tag_id": tag_id, "decision": "approved"}
                                            )
                                            if resp.status_code == 200:
                                                st.success("Approved!")
                                                st.rerun()
                                            else:
                                                st.error("Failed to update")
                                        except Exception as e:
                                            st.error(f"Error: {e}")

                                with btn_col2:
                                    if st.button("Reject", key=f"reject_{tag_id}", use_container_width=True):
                                        try:
                                            resp = requests.post(
                                                f"{API_BASE_URL}/api/erp/manual-decision",
                                                data={"tag_id": tag_id, "decision": "rejected"}
                                            )
                                            if resp.status_code == 200:
                                                st.success("Rejected!")
                                                st.rerun()
                                            else:
                                                st.error("Failed to update")
                                        except Exception as e:
                                            st.error(f"Error: {e}")

            else:
                st.markdown("""
                <div class="empty-state">
                    <div class="empty-icon">&#10004;</div>
                    <div class="empty-title">All caught up!</div>
                    <div class="empty-hint">No items pending review</div>
                </div>
                """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Could not load pending items: {e}")

    st.markdown("---")

    # Quick Test Section
    st.markdown("**Quick Test: Upload & Process**")
    st.caption("Test the ERP integration by uploading an image directly")

    col1, col2 = st.columns(2)

    with col1:
        test_tag_id = st.text_input("Tag ID", placeholder="e.g., TAG001", key="erp_test_tag")
        test_huid = st.text_input("Expected HUID", placeholder="e.g., AB1234", key="erp_test_huid")

    with col2:
        test_image = st.file_uploader(
            "Upload Image",
            type=["png", "jpg", "jpeg"],
            key="erp_test_image"
        )

    if test_image and test_tag_id and test_huid:
        if st.button("Process", key="erp_test_process", use_container_width=True):
            with st.spinner("Processing..."):
                try:
                    files = {"file": (test_image.name, test_image.getvalue(), test_image.type)}
                    data = {
                        "tag_id": test_tag_id,
                        "expected_huid": test_huid
                    }

                    response = requests.post(
                        f"{API_BASE_URL}/api/erp/upload-and-process",
                        files=files,
                        data=data
                    )

                    if response.status_code == 200:
                        result = response.json()

                        decision = result.get("decision", "pending")
                        huid_match = result.get("huid_match", False)
                        conf = result.get("confidence", 0)

                        decision_class = {
                            'approved': 'badge-success',
                            'rejected': 'badge-danger',
                            'manual_review': 'badge-warning',
                        }.get(decision, 'badge-info')

                        match_class = "badge-success" if huid_match else "badge-danger"
                        match_text = "MATCH" if huid_match else "MISMATCH"

                        st.markdown(f"""
                        <div class="result-card">
                            <div style="display: flex; justify-content: space-between; margin-bottom: 12px;">
                                <span class="badge {match_class}">{match_text}</span>
                                <span class="badge {decision_class}">{decision.replace('_', ' ').upper()}</span>
                            </div>
                            <div class="data-row">
                                <span class="data-label">Expected HUID</span>
                                <span class="data-value">{result.get('expected_huid', '--')}</span>
                            </div>
                            <div class="data-row">
                                <span class="data-label">Detected HUID</span>
                                <span class="data-value">{result.get('actual_huid') or 'Not detected'}</span>
                            </div>
                            <div class="data-row">
                                <span class="data-label">Purity</span>
                                <span class="data-value">{result.get('purity_code') or '--'} ({result.get('karat') or '--'})</span>
                            </div>
                            <div class="data-row">
                                <span class="data-label">Confidence</span>
                                <span class="data-value">{conf*100:.1f}%</span>
                            </div>
                            <div class="data-row">
                                <span class="data-label">Processing Time</span>
                                <span class="data-value">{result.get('processing_time_ms', 0)}ms</span>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                    else:
                        st.error(f"Error: {response.json().get('detail', 'Processing failed')}")

                except Exception as e:
                    st.error(f"Error: {e}")


def main():
    # Apply styles
    st.markdown(get_styles(), unsafe_allow_html=True)

    # Header
    render_header()

    # Check API status
    if not check_api_health():
        st.warning("API is not responding. Make sure the server is running on port 8000.")

    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "Stage 1: Upload Data",
        "Stage 2: Process Images",
        "Stage 3: View Results",
        "ERP Monitor"
    ])

    with tab1:
        render_stage1()

    with tab2:
        render_stage2()

    with tab3:
        render_stage3()

    with tab4:
        render_erp_monitor()


if __name__ == "__main__":
    main()
