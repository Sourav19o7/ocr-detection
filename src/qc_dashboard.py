"""
Hallmark QC Dashboard - Premium Obsidian Luxe Edition.

An ultra-premium dark mode dashboard featuring:
- Obsidian black with luminous gold/amber jewelry-inspired accents
- Sophisticated glassmorphism with volcanic glass effects
- Syne + Outfit typography for luxury brand feel
- Micro-animations and premium interactions
- Stage 1: Upload CSV/Excel with tag IDs and expected HUIDs
- Stage 2: Upload images for processing
- Stage 3: View and search results
- ERP Integration Monitor
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


def get_full_image_url(image_url: str) -> str:
    """Convert relative image URLs to full URLs for display."""
    if not image_url:
        return None
    # If it's already a full URL (http/https), return as-is
    if image_url.startswith("http://") or image_url.startswith("https://"):
        return image_url
    # If it's a relative path starting with /, prepend the API base URL
    if image_url.startswith("/"):
        return f"{API_BASE_URL}{image_url}"
    return image_url


def fetch_image_bytes(image_url: str):
    """Fetch image from URL and return bytes for Streamlit display."""
    if not image_url:
        return None
    full_url = get_full_image_url(image_url)
    if not full_url:
        return None
    try:
        response = requests.get(full_url, timeout=10)
        if response.status_code == 200:
            return io.BytesIO(response.content)
    except:
        pass
    return None

# Page config
st.set_page_config(
    page_title="Hallmark QC | Premium",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Initialize theme in session state - Default to dark (premium mode)
if "theme" not in st.session_state:
    st.session_state.theme = "dark"

def toggle_theme():
    st.session_state.theme = "dark" if st.session_state.theme == "light" else "light"


# CSS - OBSIDIAN LUXE PREMIUM REDESIGN
def get_styles():
    is_dark = st.session_state.theme == "dark"

    if is_dark:
        # OBSIDIAN LUXE - Volcanic Black with Luminous Gold
        bg_primary = "#09090B"           # True obsidian black
        bg_secondary = "#0C0C0F"         # Slightly elevated black
        bg_card = "#111114"              # Card surface - volcanic glass
        bg_input = "#16161A"             # Input fields - charcoal
        bg_elevated = "#1A1A1F"          # Elevated surfaces
        border_color = "#27272A"         # Zinc borders
        border_light = "#1E1E22"         # Subtle borders
        border_glow = "#D4AF37"          # Gold glow border
        text_primary = "#FAFAFA"         # Pure white text
        text_secondary = "#A1A1AA"       # Zinc-400
        text_muted = "#71717A"           # Zinc-500
        accent = "#D4AF37"               # Luxurious gold
        accent_hover = "#F5D061"         # Bright gold hover
        accent_secondary = "#B8860B"     # Dark goldenrod
        accent_glow = "rgba(212, 175, 55, 0.35)"
        accent_glow_strong = "rgba(212, 175, 55, 0.5)"
        success = "#22C55E"              # Emerald success
        success_glow = "rgba(34, 197, 94, 0.25)"
        danger = "#EF4444"               # Ruby red
        danger_glow = "rgba(239, 68, 68, 0.25)"
        warning = "#F59E0B"              # Amber warning
        warning_glow = "rgba(245, 158, 11, 0.25)"
        info = "#38BDF8"                 # Sky blue info
        info_glow = "rgba(56, 189, 248, 0.25)"
        gradient_start = "#D4AF37"
        gradient_end = "#B8860B"
    else:
        # IVORY LUXE - Cream with Gold Accents (Light mode alternative)
        bg_primary = "#FFFEF9"
        bg_secondary = "#FDFCF7"
        bg_card = "#FFFFFF"
        bg_input = "#F8F7F2"
        bg_elevated = "#FFFFFF"
        border_color = "#E8E4D9"
        border_light = "#F0EDE5"
        border_glow = "#D4AF37"
        text_primary = "#18181B"
        text_secondary = "#52525B"
        text_muted = "#71717A"
        accent = "#B8860B"
        accent_hover = "#D4AF37"
        accent_secondary = "#996515"
        accent_glow = "rgba(184, 134, 11, 0.2)"
        accent_glow_strong = "rgba(184, 134, 11, 0.35)"
        success = "#16A34A"
        success_glow = "rgba(22, 163, 74, 0.15)"
        danger = "#DC2626"
        danger_glow = "rgba(220, 38, 38, 0.15)"
        warning = "#D97706"
        warning_glow = "rgba(217, 119, 6, 0.15)"
        info = "#0284C7"
        info_glow = "rgba(2, 132, 199, 0.15)"
        gradient_start = "#D4AF37"
        gradient_end = "#B8860B"

    return f"""
    <style>
    /* ═══════════════════════════════════════════════════════════════════════════
       OBSIDIAN LUXE - Premium Hallmark QC Dashboard
       A jewelry-grade interface with volcanic glass aesthetics
       ═══════════════════════════════════════════════════════════════════════════ */

    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700;800&family=Outfit:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

    :root {{
        --bg-primary: {bg_primary};
        --bg-secondary: {bg_secondary};
        --bg-card: {bg_card};
        --bg-input: {bg_input};
        --bg-elevated: {bg_elevated};
        --border: {border_color};
        --border-light: {border_light};
        --border-glow: {border_glow};
        --text-primary: {text_primary};
        --text-secondary: {text_secondary};
        --text-muted: {text_muted};
        --accent: {accent};
        --accent-hover: {accent_hover};
        --accent-secondary: {accent_secondary};
        --accent-glow: {accent_glow};
        --accent-glow-strong: {accent_glow_strong};
        --success: {success};
        --success-glow: {success_glow};
        --danger: {danger};
        --danger-glow: {danger_glow};
        --warning: {warning};
        --warning-glow: {warning_glow};
        --info: {info};
        --info-glow: {info_glow};
        --gradient-gold: linear-gradient(135deg, {gradient_start} 0%, {gradient_end} 100%);
        --gradient-gold-reverse: linear-gradient(135deg, {gradient_end} 0%, {gradient_start} 100%);
    }}

    /* ─── Global Typography ─────────────────────────────────────────────────── */
    * {{
        font-family: 'Outfit', -apple-system, BlinkMacSystemFont, system-ui, sans-serif !important;
        letter-spacing: -0.01em;
        -webkit-font-smoothing: antialiased;
        -moz-osx-font-smoothing: grayscale;
    }}

    h1, h2, h3, .header-title, .stage-title {{
        font-family: 'Syne', sans-serif !important;
        font-weight: 700;
        letter-spacing: -0.03em;
    }}

    code, pre, .mono {{
        font-family: 'JetBrains Mono', monospace !important;
    }}

    /* ─── Base App Styling ──────────────────────────────────────────────────── */
    .stApp {{
        background: var(--bg-primary) !important;
        position: relative;
        min-height: 100vh;
    }}

    /* Luxurious Ambient Glow - Top Right */
    .stApp::after {{
        content: '';
        position: fixed;
        top: -30%;
        right: -15%;
        width: 70%;
        height: 100%;
        background: radial-gradient(ellipse at center, var(--accent-glow) 0%, transparent 65%);
        pointer-events: none;
        z-index: 0;
        animation: ambientFloat 20s ease-in-out infinite;
        filter: blur(60px);
    }}

    /* Secondary Glow - Bottom Left */
    .stApp::before {{
        content: '';
        position: fixed;
        bottom: -20%;
        left: -10%;
        width: 50%;
        height: 80%;
        background: radial-gradient(ellipse at center, {'rgba(212, 175, 55, 0.08)' if is_dark else 'rgba(184, 134, 11, 0.06)'} 0%, transparent 60%);
        pointer-events: none;
        z-index: 0;
        animation: ambientFloat 25s ease-in-out infinite reverse;
        filter: blur(80px);
    }}

    @keyframes ambientFloat {{
        0%, 100% {{
            opacity: {'0.6' if is_dark else '0.4'};
            transform: translate(0, 0) scale(1);
        }}
        33% {{
            opacity: {'0.8' if is_dark else '0.5'};
            transform: translate(5%, -3%) scale(1.05);
        }}
        66% {{
            opacity: {'0.5' if is_dark else '0.35'};
            transform: translate(-3%, 5%) scale(0.98);
        }}
    }}

    /* Premium Noise Texture Overlay */
    body::after {{
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        opacity: {'0.025' if is_dark else '0.015'};
        background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 512 512' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noise'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noise)'/%3E%3C/svg%3E");
        pointer-events: none;
        z-index: 9999;
    }}

    /* Override Streamlit containers */
    [data-testid="stAppViewContainer"],
    [data-testid="stAppViewBlockContainer"],
    .main .block-container {{
        background: transparent !important;
        position: relative;
        z-index: 1;
    }}

    #MainMenu, footer, header, [data-testid="stToolbar"],
    [data-testid="stDecoration"], [data-testid="stStatusWidget"] {{
        display: none !important;
    }}

    .block-container {{
        padding: 2.5rem 4rem !important;
        max-width: 1700px;
    }}

    @media (max-width: 1200px) {{
        .block-container {{
            padding: 2rem 2rem !important;
        }}
    }}

    /* ─── Global Text Styling ───────────────────────────────────────────────── */
    .stApp p, .stApp span, .stApp label, .stApp div,
    .stMarkdown, .stMarkdown p, .stMarkdown span,
    .stMarkdown strong, .stMarkdown em, .stMarkdown li,
    [data-testid="stText"],
    [data-testid="stCaptionContainer"] {{
        color: var(--text-primary) !important;
    }}

    .stCaption, [data-testid="stCaptionContainer"] p,
    [data-testid="stCaptionContainer"] span {{
        color: var(--text-muted) !important;
        font-size: 13px;
        letter-spacing: 0.01em;
    }}

    /* Widget labels - Premium styling */
    .stTextInput label, .stSelectbox label, .stMultiSelect label,
    .stNumberInput label, .stTextArea label, .stFileUploader label,
    .stRadio label, .stCheckbox label, .stDateInput label,
    [data-testid="stWidgetLabel"] label,
    [data-testid="stWidgetLabel"] p {{
        color: var(--text-secondary) !important;
        font-weight: 500 !important;
        font-size: 13px !important;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 8px !important;
    }}

    /* Radio button styling */
    .stRadio [role="radiogroup"] label p,
    .stRadio [role="radiogroup"] label span,
    .stRadio [role="radiogroup"] label div {{
        color: var(--text-primary) !important;
        font-weight: 500;
    }}

    .stRadio [role="radiogroup"] {{
        gap: 8px;
    }}

    .stRadio [role="radiogroup"] label {{
        background: var(--bg-card) !important;
        border: 1px solid var(--border) !important;
        border-radius: 12px !important;
        padding: 12px 20px !important;
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
    }}

    .stRadio [role="radiogroup"] label:hover {{
        border-color: var(--accent) !important;
        background: var(--bg-elevated) !important;
    }}

    .stRadio [role="radiogroup"] label[data-checked="true"] {{
        border-color: var(--accent) !important;
        background: {'rgba(212, 175, 55, 0.1)' if is_dark else 'rgba(184, 134, 11, 0.08)'} !important;
        box-shadow: 0 0 0 1px var(--accent), 0 4px 12px var(--accent-glow);
    }}

    /* Dropdown/Selectbox Menus */
    [data-baseweb="popover"],
    [data-baseweb="menu"] {{
        background: var(--bg-elevated) !important;
        border: 1px solid var(--border) !important;
        border-radius: 16px !important;
        box-shadow: 0 20px 50px rgba(0, 0, 0, {'0.5' if is_dark else '0.15'}),
                    0 0 0 1px var(--border) !important;
        backdrop-filter: blur(20px);
        overflow: hidden;
    }}

    [data-baseweb="menu"] [role="option"] {{
        color: var(--text-primary) !important;
        padding: 14px 18px !important;
        transition: all 0.2s ease;
    }}

    [data-baseweb="menu"] [role="option"]:hover {{
        background: {'rgba(212, 175, 55, 0.12)' if is_dark else 'rgba(184, 134, 11, 0.08)'} !important;
        color: var(--accent) !important;
    }}

    /* Spinner */
    .stSpinner > div > span {{
        color: var(--accent) !important;
        font-weight: 500;
    }}

    .stSpinner svg {{
        stroke: var(--accent) !important;
    }}

    /* Disabled Button */
    .stButton > button:disabled {{
        background: var(--bg-input) !important;
        color: var(--text-muted) !important;
        box-shadow: none !important;
        opacity: 0.5;
        cursor: not-allowed;
    }}

    /* Alert Boxes */
    [data-testid="stAlert"] {{
        background: var(--bg-card) !important;
        border: 1px solid var(--border) !important;
        border-radius: 16px !important;
        backdrop-filter: blur(12px);
    }}

    .stSuccess {{
        border-left: 4px solid var(--success) !important;
        background: {'rgba(34, 197, 94, 0.08)' if is_dark else 'rgba(22, 163, 74, 0.06)'} !important;
    }}

    .stError {{
        border-left: 4px solid var(--danger) !important;
        background: {'rgba(239, 68, 68, 0.08)' if is_dark else 'rgba(220, 38, 38, 0.06)'} !important;
    }}

    .stWarning {{
        border-left: 4px solid var(--warning) !important;
        background: {'rgba(245, 158, 11, 0.08)' if is_dark else 'rgba(217, 119, 6, 0.06)'} !important;
    }}

    .stInfo {{
        border-left: 4px solid var(--info) !important;
        background: {'rgba(56, 189, 248, 0.08)' if is_dark else 'rgba(2, 132, 199, 0.06)'} !important;
    }}

    /* Progress bar */
    .stProgress > div > div {{
        background: var(--bg-input) !important;
        border-radius: 8px;
        overflow: hidden;
    }}

    .stProgress > div > div > div {{
        background: var(--gradient-gold) !important;
        border-radius: 8px;
    }}

    /* DataFrames */
    [data-testid="stDataFrame"],
    [data-testid="stDataFrame"] > div {{
        background: var(--bg-card) !important;
        border-radius: 16px !important;
        overflow: hidden;
    }}

    /* ═══════════════════════════════════════════════════════════════════════════
       PREMIUM NAVIGATION TABS - Volcanic Glass Morphism
       ═══════════════════════════════════════════════════════════════════════════ */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 8px;
        background: {'rgba(17, 17, 20, 0.7)' if is_dark else 'rgba(255, 255, 255, 0.9)'};
        backdrop-filter: blur(40px) saturate(180%);
        -webkit-backdrop-filter: blur(40px) saturate(180%);
        border-radius: 20px;
        padding: 8px 10px;
        border: 1px solid {'rgba(212, 175, 55, 0.15)' if is_dark else 'rgba(184, 134, 11, 0.2)'};
        box-shadow: 0 4px 24px rgba(0, 0, 0, {'0.4' if is_dark else '0.1'}),
                    inset 0 1px 0 {'rgba(255, 255, 255, 0.05)' if is_dark else 'rgba(255, 255, 255, 0.8)'},
                    0 0 0 1px {'rgba(255, 255, 255, 0.03)' if is_dark else 'transparent'};
    }}

    .stTabs [data-baseweb="tab"] {{
        height: 52px;
        border-radius: 14px;
        font-weight: 600;
        font-size: 13px;
        padding: 0 28px;
        border: none;
        color: var(--text-muted) !important;
        background: transparent !important;
        transition: all 0.4s cubic-bezier(0.34, 1.56, 0.64, 1);
        letter-spacing: 0.02em;
        position: relative;
        overflow: hidden;
    }}

    .stTabs [data-baseweb="tab"] span,
    .stTabs [data-baseweb="tab"] p,
    .stTabs [data-baseweb="tab"] div {{
        color: inherit !important;
        font-weight: inherit !important;
    }}

    .stTabs [data-baseweb="tab"]::before {{
        content: '';
        position: absolute;
        inset: 0;
        background: var(--gradient-gold);
        opacity: 0;
        transition: opacity 0.3s ease;
        border-radius: inherit;
    }}

    .stTabs [data-baseweb="tab"]:hover {{
        color: var(--text-primary) !important;
        background: {'rgba(212, 175, 55, 0.08)' if is_dark else 'rgba(184, 134, 11, 0.06)'} !important;
        transform: translateY(-2px);
    }}

    .stTabs [data-baseweb="tab-highlight"],
    .stTabs [data-baseweb="tab-border"] {{
        display: none !important;
    }}

    .stTabs [aria-selected="true"] {{
        background: var(--gradient-gold) !important;
        color: {'#09090B' if is_dark else '#FFFFFF'} !important;
        font-weight: 700 !important;
        box-shadow: 0 8px 24px var(--accent-glow),
                    0 2px 8px var(--accent-glow-strong),
                    inset 0 1px 0 rgba(255, 255, 255, 0.25);
        transform: translateY(-2px) scale(1.02);
    }}

    .stTabs [aria-selected="true"] span,
    .stTabs [aria-selected="true"] p,
    .stTabs [aria-selected="true"] div {{
        color: {'#09090B' if is_dark else '#FFFFFF'} !important;
        font-weight: 700 !important;
        text-shadow: 0 1px 2px rgba(0, 0, 0, 0.1);
    }}

    .stTabs [aria-selected="true"]:hover {{
        transform: translateY(-3px) scale(1.03);
        box-shadow: 0 12px 32px var(--accent-glow-strong),
                    0 4px 12px var(--accent-glow),
                    inset 0 1px 0 rgba(255, 255, 255, 0.3);
    }}

    .stTabs [data-baseweb="tab-panel"] {{
        background: transparent !important;
        padding-top: 32px;
    }}

    /* ═══════════════════════════════════════════════════════════════════════════
       PREMIUM BUTTONS - Gold Gradient with Micro-interactions
       ═══════════════════════════════════════════════════════════════════════════ */
    .stButton > button {{
        background: var(--gradient-gold) !important;
        color: {'#09090B' if is_dark else '#FFFFFF'} !important;
        border: none !important;
        border-radius: 14px;
        padding: 16px 32px;
        font-family: 'Outfit', sans-serif !important;
        font-weight: 700;
        font-size: 13px;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        transition: all 0.4s cubic-bezier(0.34, 1.56, 0.64, 1);
        box-shadow: 0 4px 16px var(--accent-glow),
                    0 1px 3px rgba(0, 0, 0, 0.2),
                    inset 0 1px 0 rgba(255, 255, 255, 0.2);
        position: relative;
        overflow: hidden;
    }}

    .stButton > button::before {{
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.3), transparent);
        transition: left 0.6s ease;
    }}

    .stButton > button::after {{
        content: '';
        position: absolute;
        inset: 0;
        background: var(--gradient-gold-reverse);
        opacity: 0;
        transition: opacity 0.3s ease;
        border-radius: inherit;
    }}

    .stButton > button:hover {{
        transform: translateY(-3px) scale(1.02);
        box-shadow: 0 8px 28px var(--accent-glow-strong),
                    0 2px 6px rgba(0, 0, 0, 0.15),
                    inset 0 1px 0 rgba(255, 255, 255, 0.3);
    }}

    .stButton > button:hover::before {{
        left: 100%;
    }}

    .stButton > button:active {{
        transform: translateY(-1px) scale(0.98);
        box-shadow: 0 2px 8px var(--accent-glow),
                    inset 0 2px 4px rgba(0, 0, 0, 0.1);
        transition: all 0.1s ease;
    }}

    /* ─── Input Fields - Obsidian Glass ─────────────────────────────────────── */
    .stSelectbox [data-baseweb="select"] > div,
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input {{
        background: {'rgba(22, 22, 26, 0.8)' if is_dark else 'rgba(248, 247, 242, 0.9)'} !important;
        backdrop-filter: blur(12px);
        border: 1px solid var(--border) !important;
        border-radius: 14px !important;
        color: var(--text-primary) !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        padding: 16px 20px !important;
        font-size: 15px !important;
        font-weight: 500 !important;
        box-shadow: inset 0 2px 4px rgba(0, 0, 0, {'0.15' if is_dark else '0.04'});
    }}

    .stSelectbox [data-baseweb="select"] > div::placeholder,
    .stTextInput > div > div > input::placeholder {{
        color: var(--text-muted) !important;
        opacity: 0.7;
    }}

    .stSelectbox [data-baseweb="select"] > div:focus-within,
    .stTextInput > div > div > input:focus,
    .stNumberInput > div > div > input:focus {{
        border-color: var(--accent) !important;
        box-shadow: 0 0 0 3px var(--accent-glow),
                    inset 0 2px 4px rgba(0, 0, 0, {'0.1' if is_dark else '0.03'}) !important;
        background: var(--bg-elevated) !important;
        outline: none;
    }}

    .stTextArea textarea {{
        background: {'rgba(22, 22, 26, 0.8)' if is_dark else 'rgba(248, 247, 242, 0.9)'} !important;
        backdrop-filter: blur(12px);
        border: 1px solid var(--border) !important;
        border-radius: 14px !important;
        color: var(--text-primary) !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        padding: 16px 20px !important;
        font-size: 14px !important;
        box-shadow: inset 0 2px 4px rgba(0, 0, 0, {'0.15' if is_dark else '0.04'});
    }}

    .stTextArea textarea:focus {{
        border-color: var(--accent) !important;
        box-shadow: 0 0 0 3px var(--accent-glow),
                    inset 0 2px 4px rgba(0, 0, 0, {'0.1' if is_dark else '0.03'}) !important;
        outline: none;
    }}

    /* ═══════════════════════════════════════════════════════════════════════════
       PREMIUM FILE UPLOADER - Volcanic Glass Drop Zone
       ═══════════════════════════════════════════════════════════════════════════ */
    [data-testid="stFileUploader"] section {{
        background: {'rgba(17, 17, 20, 0.6)' if is_dark else 'rgba(255, 255, 255, 0.7)'} !important;
        backdrop-filter: blur(24px) saturate(150%);
        -webkit-backdrop-filter: blur(24px) saturate(150%);
        border: 2px dashed {'rgba(212, 175, 55, 0.4)' if is_dark else 'rgba(184, 134, 11, 0.35)'} !important;
        border-radius: 24px !important;
        transition: all 0.5s cubic-bezier(0.34, 1.56, 0.64, 1);
        padding: 40px 32px !important;
        position: relative;
        overflow: hidden;
    }}

    [data-testid="stFileUploader"] section::before {{
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: conic-gradient(from 0deg, transparent, var(--accent-glow), transparent, var(--accent-glow), transparent);
        opacity: 0;
        transition: opacity 0.5s ease;
        animation: borderRotate 4s linear infinite paused;
    }}

    @keyframes borderRotate {{
        100% {{ transform: rotate(360deg); }}
    }}

    [data-testid="stFileUploader"] section:hover {{
        border-color: var(--accent) !important;
        border-style: solid !important;
        background: {'rgba(212, 175, 55, 0.06)' if is_dark else 'rgba(184, 134, 11, 0.04)'} !important;
        box-shadow: 0 16px 48px var(--accent-glow),
                    0 0 0 1px {'rgba(212, 175, 55, 0.3)' if is_dark else 'rgba(184, 134, 11, 0.2)'},
                    inset 0 1px 0 {'rgba(255, 255, 255, 0.08)' if is_dark else 'rgba(255, 255, 255, 0.9)'};
        transform: translateY(-4px);
    }}

    [data-testid="stFileUploader"] section:hover::before {{
        opacity: 0.3;
        animation-play-state: running;
    }}

    /* ═══════════════════════════════════════════════════════════════════════════
       METRIC CARDS - Premium Data Display
       ═══════════════════════════════════════════════════════════════════════════ */
    [data-testid="stMetric"] {{
        background: {'rgba(17, 17, 20, 0.7)' if is_dark else 'rgba(255, 255, 255, 0.85)'} !important;
        backdrop-filter: blur(24px) saturate(180%);
        -webkit-backdrop-filter: blur(24px) saturate(180%);
        border: 1px solid {'rgba(212, 175, 55, 0.15)' if is_dark else 'rgba(184, 134, 11, 0.12)'} !important;
        border-radius: 20px !important;
        padding: 28px 24px !important;
        box-shadow: 0 8px 32px rgba(0, 0, 0, {'0.3' if is_dark else '0.08'}),
                    inset 0 1px 0 {'rgba(255, 255, 255, 0.05)' if is_dark else 'rgba(255, 255, 255, 0.9)'};
        transition: all 0.4s cubic-bezier(0.34, 1.56, 0.64, 1);
        position: relative;
        overflow: hidden;
    }}

    [data-testid="stMetric"]::before {{
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: var(--gradient-gold);
        opacity: 0.7;
    }}

    [data-testid="stMetric"]::after {{
        content: '';
        position: absolute;
        top: 0;
        right: 0;
        width: 100px;
        height: 100px;
        background: radial-gradient(circle, var(--accent-glow) 0%, transparent 70%);
        opacity: 0;
        transition: opacity 0.4s ease;
    }}

    [data-testid="stMetric"]:hover {{
        transform: translateY(-6px);
        border-color: {'rgba(212, 175, 55, 0.35)' if is_dark else 'rgba(184, 134, 11, 0.25)'} !important;
        box-shadow: 0 16px 48px rgba(0, 0, 0, {'0.4' if is_dark else '0.12'}),
                    0 0 0 1px var(--accent-glow),
                    inset 0 1px 0 {'rgba(255, 255, 255, 0.1)' if is_dark else 'rgba(255, 255, 255, 1)'};
    }}

    [data-testid="stMetric"]:hover::after {{
        opacity: 0.6;
    }}

    [data-testid="stMetricValue"] {{
        font-family: 'Syne', sans-serif !important;
        font-weight: 800 !important;
        font-size: 42px !important;
        letter-spacing: -0.04em !important;
        background: var(--gradient-gold);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        line-height: 1.1;
    }}

    [data-testid="stMetricLabel"] {{
        font-family: 'Outfit', sans-serif !important;
        color: var(--text-muted) !important;
        font-weight: 600 !important;
        font-size: 11px !important;
        letter-spacing: 0.12em !important;
        text-transform: uppercase;
        margin-top: 8px !important;
    }}

    /* ─── Expanders ─────────────────────────────────────────────────────────── */
    [data-testid="stExpander"] {{
        border: 1px solid var(--border) !important;
        border-radius: 16px !important;
        background: var(--bg-card) !important;
        overflow: hidden;
        transition: all 0.3s ease;
    }}

    [data-testid="stExpander"]:hover {{
        border-color: {'rgba(212, 175, 55, 0.3)' if is_dark else 'rgba(184, 134, 11, 0.2)'} !important;
    }}

    [data-testid="stExpander"] summary {{
        background: var(--bg-input) !important;
        padding: 16px 20px !important;
        transition: all 0.3s ease;
    }}

    [data-testid="stExpander"] summary:hover {{
        background: var(--bg-elevated) !important;
    }}

    [data-testid="stExpander"] summary span,
    [data-testid="stExpander"] summary p {{
        color: var(--text-primary) !important;
        font-weight: 600 !important;
    }}

    [data-testid="stExpander"] [data-testid="stExpanderDetails"] {{
        background: var(--bg-card) !important;
        padding: 20px !important;
        border-top: 1px solid var(--border);
    }}

    /* ═══════════════════════════════════════════════════════════════════════════
       HEADER BAR - Obsidian Glass Hero
       ═══════════════════════════════════════════════════════════════════════════ */
    .header-bar {{
        background: {'rgba(17, 17, 20, 0.75)' if is_dark else 'rgba(255, 255, 255, 0.92)'};
        backdrop-filter: blur(40px) saturate(200%);
        -webkit-backdrop-filter: blur(40px) saturate(200%);
        border: 1px solid {'rgba(212, 175, 55, 0.2)' if is_dark else 'rgba(184, 134, 11, 0.15)'};
        border-radius: 28px;
        padding: 36px 44px;
        margin-bottom: 40px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        box-shadow: 0 16px 48px rgba(0, 0, 0, {'0.4' if is_dark else '0.1'}),
                    0 0 0 1px {'rgba(255, 255, 255, 0.03)' if is_dark else 'transparent'},
                    inset 0 1px 0 {'rgba(255, 255, 255, 0.06)' if is_dark else 'rgba(255, 255, 255, 1)'};
        position: relative;
        overflow: hidden;
        animation: headerReveal 1s cubic-bezier(0.34, 1.56, 0.64, 1);
    }}

    @keyframes headerReveal {{
        from {{
            opacity: 0;
            transform: translateY(-30px) scale(0.98);
        }}
        to {{
            opacity: 1;
            transform: translateY(0) scale(1);
        }}
    }}

    .header-bar::before {{
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 5px;
        height: 100%;
        background: var(--gradient-gold);
        box-shadow: 0 0 30px var(--accent-glow-strong),
                    0 0 60px var(--accent-glow);
    }}

    .header-bar::after {{
        content: '';
        position: absolute;
        top: -100%;
        right: -20%;
        width: 500px;
        height: 300%;
        background: conic-gradient(from 45deg, transparent 0deg, var(--accent-glow) 90deg, transparent 180deg);
        opacity: 0.15;
        pointer-events: none;
        animation: headerGlow 8s linear infinite;
    }}

    @keyframes headerGlow {{
        100% {{ transform: rotate(360deg); }}
    }}

    .header-title {{
        font-family: 'Syne', sans-serif !important;
        font-size: 36px;
        font-weight: 800;
        letter-spacing: -0.04em;
        background: var(--gradient-gold);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        position: relative;
        z-index: 1;
        text-shadow: none;
    }}

    .header-subtitle {{
        font-family: 'Outfit', sans-serif !important;
        font-size: 14px;
        color: var(--text-secondary);
        margin-top: 8px;
        font-weight: 500;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        position: relative;
        z-index: 1;
    }}

    .header-badge {{
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 8px 16px;
        background: {'rgba(212, 175, 55, 0.15)' if is_dark else 'rgba(184, 134, 11, 0.1)'};
        border: 1px solid {'rgba(212, 175, 55, 0.3)' if is_dark else 'rgba(184, 134, 11, 0.2)'};
        border-radius: 100px;
        font-size: 11px;
        font-weight: 700;
        color: var(--accent);
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-left: 16px;
    }}

    /* ═══════════════════════════════════════════════════════════════════════════
       STAGE CARDS - Premium Section Headers
       ═══════════════════════════════════════════════════════════════════════════ */
    .stage-card {{
        background: {'rgba(17, 17, 20, 0.65)' if is_dark else 'rgba(255, 255, 255, 0.88)'};
        backdrop-filter: blur(32px) saturate(180%);
        -webkit-backdrop-filter: blur(32px) saturate(180%);
        border: 1px solid {'rgba(212, 175, 55, 0.12)' if is_dark else 'rgba(184, 134, 11, 0.1)'};
        border-radius: 24px;
        padding: 36px 40px;
        margin-bottom: 28px;
        transition: all 0.5s cubic-bezier(0.34, 1.56, 0.64, 1);
        box-shadow: 0 12px 40px rgba(0, 0, 0, {'0.3' if is_dark else '0.08'}),
                    inset 0 1px 0 {'rgba(255, 255, 255, 0.04)' if is_dark else 'rgba(255, 255, 255, 0.9)'};
        position: relative;
        overflow: hidden;
        animation: stageReveal 0.7s cubic-bezier(0.34, 1.56, 0.64, 1) backwards;
    }}

    @keyframes stageReveal {{
        from {{
            opacity: 0;
            transform: translateY(40px) scale(0.97);
        }}
        to {{
            opacity: 1;
            transform: translateY(0) scale(1);
        }}
    }}

    .stage-card::before {{
        content: '';
        position: absolute;
        top: 0;
        right: 0;
        width: 40%;
        height: 100%;
        background: radial-gradient(ellipse at 100% 0%, var(--accent-glow) 0%, transparent 70%);
        opacity: 0.4;
        pointer-events: none;
        transition: all 0.5s ease;
    }}

    .stage-card::after {{
        content: '';
        position: absolute;
        bottom: 0;
        left: 0;
        right: 0;
        height: 1px;
        background: linear-gradient(90deg, transparent, var(--accent), transparent);
        opacity: 0.3;
    }}

    .stage-card:hover {{
        border-color: {'rgba(212, 175, 55, 0.3)' if is_dark else 'rgba(184, 134, 11, 0.25)'};
        box-shadow: 0 20px 60px rgba(0, 0, 0, {'0.4' if is_dark else '0.12'}),
                    0 0 0 1px var(--accent-glow),
                    inset 0 1px 0 {'rgba(255, 255, 255, 0.08)' if is_dark else 'rgba(255, 255, 255, 1)'};
        transform: translateY(-4px);
    }}

    .stage-card:hover::before {{
        opacity: 0.7;
    }}

    .stage-number {{
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 54px;
        height: 54px;
        background: var(--gradient-gold);
        color: {'#09090B' if is_dark else '#FFFFFF'};
        border-radius: 16px;
        font-family: 'Syne', sans-serif !important;
        font-weight: 800;
        font-size: 22px;
        margin-right: 20px;
        box-shadow: 0 8px 24px var(--accent-glow-strong),
                    0 2px 8px var(--accent-glow),
                    inset 0 1px 0 rgba(255, 255, 255, 0.25);
        position: relative;
        z-index: 1;
        transition: all 0.4s cubic-bezier(0.34, 1.56, 0.64, 1);
    }}

    .stage-card:hover .stage-number {{
        transform: scale(1.08) rotate(-3deg);
        box-shadow: 0 12px 32px var(--accent-glow-strong),
                    0 4px 12px var(--accent-glow),
                    inset 0 1px 0 rgba(255, 255, 255, 0.35);
    }}

    .stage-title {{
        font-family: 'Syne', sans-serif !important;
        font-size: 26px;
        font-weight: 700;
        color: var(--text-primary);
        display: inline;
        letter-spacing: -0.03em;
        position: relative;
        z-index: 1;
    }}

    .stage-desc {{
        font-family: 'Outfit', sans-serif !important;
        font-size: 15px;
        color: var(--text-secondary);
        margin-top: 14px;
        margin-bottom: 24px;
        line-height: 1.7;
        font-weight: 400;
        position: relative;
        z-index: 1;
        max-width: 600px;
    }}

    /* ═══════════════════════════════════════════════════════════════════════════
       RESULT CARDS - Data Display Containers
       ═══════════════════════════════════════════════════════════════════════════ */
    .result-card {{
        background: {'rgba(17, 17, 20, 0.6)' if is_dark else 'rgba(255, 255, 255, 0.85)'};
        backdrop-filter: blur(20px) saturate(160%);
        -webkit-backdrop-filter: blur(20px) saturate(160%);
        border: 1px solid {'rgba(39, 39, 42, 0.8)' if is_dark else 'rgba(0, 0, 0, 0.06)'};
        border-radius: 20px;
        padding: 28px;
        margin: 16px 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, {'0.25' if is_dark else '0.06'}),
                    inset 0 1px 0 {'rgba(255, 255, 255, 0.03)' if is_dark else 'rgba(255, 255, 255, 0.9)'};
        animation: cardReveal 0.5s cubic-bezier(0.34, 1.56, 0.64, 1) backwards;
        transition: all 0.35s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
    }}

    .result-card::before {{
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 1px;
        background: linear-gradient(90deg, transparent, {'rgba(212, 175, 55, 0.3)' if is_dark else 'rgba(184, 134, 11, 0.2)'}, transparent);
    }}

    .result-card:hover {{
        transform: translateY(-3px);
        border-color: {'rgba(212, 175, 55, 0.2)' if is_dark else 'rgba(184, 134, 11, 0.15)'};
        box-shadow: 0 16px 48px rgba(0, 0, 0, {'0.35' if is_dark else '0.1'}),
                    0 0 0 1px {'rgba(212, 175, 55, 0.1)' if is_dark else 'transparent'},
                    inset 0 1px 0 {'rgba(255, 255, 255, 0.05)' if is_dark else 'rgba(255, 255, 255, 1)'};
    }}

    @keyframes cardReveal {{
        from {{
            opacity: 0;
            transform: translateY(24px) scale(0.98);
        }}
        to {{
            opacity: 1;
            transform: translateY(0) scale(1);
        }}
    }}

    /* ═══════════════════════════════════════════════════════════════════════════
       BADGES - Premium Status Indicators
       ═══════════════════════════════════════════════════════════════════════════ */
    .badge {{
        display: inline-flex;
        align-items: center;
        padding: 8px 16px;
        border-radius: 10px;
        font-family: 'Outfit', sans-serif !important;
        font-size: 11px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        box-shadow: 0 2px 8px rgba(0, 0, 0, {'0.2' if is_dark else '0.08'}),
                    inset 0 1px 0 rgba(255, 255, 255, 0.1);
        transition: all 0.3s cubic-bezier(0.34, 1.56, 0.64, 1);
        position: relative;
        overflow: hidden;
    }}

    .badge::before {{
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent);
        transition: left 0.5s ease;
    }}

    .badge:hover {{
        transform: translateY(-2px) scale(1.02);
    }}

    .badge:hover::before {{
        left: 100%;
    }}

    .badge-success {{
        background: {'rgba(34, 197, 94, 0.15)' if is_dark else 'rgba(22, 163, 74, 0.12)'};
        color: {'#4ADE80' if is_dark else '#16A34A'};
        border: 1px solid {'rgba(34, 197, 94, 0.35)' if is_dark else 'rgba(22, 163, 74, 0.25)'};
        box-shadow: 0 0 16px {'rgba(34, 197, 94, 0.15)' if is_dark else 'rgba(22, 163, 74, 0.1)'};
    }}

    .badge-danger {{
        background: {'rgba(239, 68, 68, 0.15)' if is_dark else 'rgba(220, 38, 38, 0.12)'};
        color: {'#F87171' if is_dark else '#DC2626'};
        border: 1px solid {'rgba(239, 68, 68, 0.35)' if is_dark else 'rgba(220, 38, 38, 0.25)'};
        box-shadow: 0 0 16px {'rgba(239, 68, 68, 0.15)' if is_dark else 'rgba(220, 38, 38, 0.1)'};
    }}

    .badge-warning {{
        background: {'rgba(245, 158, 11, 0.15)' if is_dark else 'rgba(217, 119, 6, 0.12)'};
        color: {'#FBBF24' if is_dark else '#D97706'};
        border: 1px solid {'rgba(245, 158, 11, 0.35)' if is_dark else 'rgba(217, 119, 6, 0.25)'};
        box-shadow: 0 0 16px {'rgba(245, 158, 11, 0.15)' if is_dark else 'rgba(217, 119, 6, 0.1)'};
    }}

    .badge-info {{
        background: {'rgba(56, 189, 248, 0.15)' if is_dark else 'rgba(2, 132, 199, 0.12)'};
        color: {'#7DD3FC' if is_dark else '#0284C7'};
        border: 1px solid {'rgba(56, 189, 248, 0.35)' if is_dark else 'rgba(2, 132, 199, 0.25)'};
        box-shadow: 0 0 16px {'rgba(56, 189, 248, 0.15)' if is_dark else 'rgba(2, 132, 199, 0.1)'};
    }}

    .badge-gold {{
        background: {'rgba(212, 175, 55, 0.2)' if is_dark else 'rgba(184, 134, 11, 0.15)'};
        color: var(--accent);
        border: 1px solid {'rgba(212, 175, 55, 0.4)' if is_dark else 'rgba(184, 134, 11, 0.3)'};
        box-shadow: 0 0 16px var(--accent-glow);
    }}

    /* ─── Stat Grid ─────────────────────────────────────────────────────────── */
    .stat-grid {{
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 20px;
        margin: 24px 0;
    }}

    @media (max-width: 900px) {{
        .stat-grid {{
            grid-template-columns: repeat(2, 1fr);
        }}
    }}

    .stat-card {{
        background: {'rgba(17, 17, 20, 0.6)' if is_dark else 'rgba(255, 255, 255, 0.85)'};
        backdrop-filter: blur(16px);
        border: 1px solid var(--border);
        border-radius: 18px;
        padding: 24px;
        text-align: center;
        transition: all 0.35s cubic-bezier(0.34, 1.56, 0.64, 1);
        position: relative;
        overflow: hidden;
    }}

    .stat-card::before {{
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 2px;
        background: var(--gradient-gold);
        opacity: 0.5;
    }}

    .stat-card:hover {{
        transform: translateY(-4px);
        border-color: {'rgba(212, 175, 55, 0.25)' if is_dark else 'rgba(184, 134, 11, 0.2)'};
        box-shadow: 0 12px 32px rgba(0, 0, 0, {'0.3' if is_dark else '0.1'}),
                    0 0 0 1px var(--accent-glow);
    }}

    .stat-value {{
        font-family: 'Syne', sans-serif !important;
        font-size: 36px;
        font-weight: 800;
        letter-spacing: -0.03em;
        background: var(--gradient-gold);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }}

    .stat-label {{
        font-family: 'Outfit', sans-serif !important;
        font-size: 11px;
        font-weight: 600;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-top: 8px;
    }}

    /* ─── Data Rows ─────────────────────────────────────────────────────────── */
    .data-row {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 14px 0;
        border-bottom: 1px solid {'rgba(39, 39, 42, 0.6)' if is_dark else 'rgba(0, 0, 0, 0.06)'};
        transition: all 0.2s ease;
    }}

    .data-row:last-child {{
        border-bottom: none;
    }}

    .data-row:hover {{
        padding-left: 8px;
        background: {'rgba(212, 175, 55, 0.03)' if is_dark else 'rgba(184, 134, 11, 0.02)'};
        border-radius: 8px;
        margin: 0 -8px;
        padding-right: 8px;
    }}

    .data-label {{
        font-family: 'Outfit', sans-serif !important;
        font-weight: 500;
        color: var(--text-muted);
        font-size: 13px;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }}

    .data-value {{
        font-family: 'JetBrains Mono', monospace !important;
        font-weight: 600;
        color: var(--text-primary);
        font-size: 14px;
    }}

    /* ─── Progress Bars ─────────────────────────────────────────────────────── */
    .progress-bar {{
        height: 8px;
        background: {'rgba(39, 39, 42, 0.8)' if is_dark else 'rgba(0, 0, 0, 0.08)'};
        border-radius: 100px;
        overflow: hidden;
        margin: 12px 0;
        position: relative;
    }}

    .progress-bar::before {{
        content: '';
        position: absolute;
        inset: 0;
        background: linear-gradient(90deg, transparent 0%, rgba(255, 255, 255, 0.1) 50%, transparent 100%);
        animation: progressShine 2s ease-in-out infinite;
    }}

    @keyframes progressShine {{
        0% {{ transform: translateX(-100%); }}
        100% {{ transform: translateX(100%); }}
    }}

    .progress-fill {{
        height: 100%;
        border-radius: 100px;
        transition: width 0.8s cubic-bezier(0.34, 1.56, 0.64, 1);
        position: relative;
    }}

    .progress-fill::after {{
        content: '';
        position: absolute;
        top: 0;
        right: 0;
        width: 20px;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.4));
        border-radius: 100px;
    }}

    /* ─── Empty State ───────────────────────────────────────────────────────── */
    .empty-state {{
        text-align: center;
        padding: 80px 40px;
        color: var(--text-muted);
        background: {'rgba(17, 17, 20, 0.4)' if is_dark else 'rgba(255, 255, 255, 0.6)'};
        border-radius: 24px;
        border: 1px dashed var(--border);
    }}

    .empty-icon {{
        font-size: 56px;
        margin-bottom: 20px;
        opacity: 0.4;
        filter: grayscale(0.5);
    }}

    .empty-title {{
        font-family: 'Syne', sans-serif !important;
        font-size: 20px;
        font-weight: 700;
        color: var(--text-secondary);
        margin-bottom: 12px;
        letter-spacing: -0.02em;
    }}

    .empty-hint {{
        font-family: 'Outfit', sans-serif !important;
        font-size: 14px;
        color: var(--text-muted);
        max-width: 300px;
        margin: 0 auto;
        line-height: 1.6;
    }}

    /* ─── Theme Toggle ──────────────────────────────────────────────────────── */
    .theme-toggle {{
        background: {'rgba(17, 17, 20, 0.8)' if is_dark else 'rgba(248, 247, 242, 0.9)'};
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 12px 20px;
        font-family: 'Outfit', sans-serif !important;
        font-size: 13px;
        font-weight: 600;
        color: var(--text-secondary);
        cursor: pointer;
        transition: all 0.3s cubic-bezier(0.34, 1.56, 0.64, 1);
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }}

    .theme-toggle:hover {{
        background: var(--gradient-gold);
        color: {'#09090B' if is_dark else '#FFFFFF'};
        border-color: var(--accent);
        transform: translateY(-2px);
        box-shadow: 0 8px 24px var(--accent-glow);
    }}

    /* ─── Hide Default Streamlit Elements ───────────────────────────────────── */
    .stDeployButton {{ display: none !important; }}

    /* ─── Premium Scrollbar ─────────────────────────────────────────────────── */
    ::-webkit-scrollbar {{
        width: 10px;
        height: 10px;
    }}

    ::-webkit-scrollbar-track {{
        background: {'rgba(9, 9, 11, 0.5)' if is_dark else 'rgba(0, 0, 0, 0.03)'};
        border-radius: 100px;
    }}

    ::-webkit-scrollbar-thumb {{
        background: {'rgba(212, 175, 55, 0.3)' if is_dark else 'rgba(184, 134, 11, 0.25)'};
        border-radius: 100px;
        border: 2px solid transparent;
        background-clip: content-box;
        transition: background 0.2s ease;
    }}

    ::-webkit-scrollbar-thumb:hover {{
        background: {'rgba(212, 175, 55, 0.6)' if is_dark else 'rgba(184, 134, 11, 0.5)'};
        border: 2px solid transparent;
        background-clip: content-box;
    }}

    ::-webkit-scrollbar-corner {{
        background: transparent;
    }}

    /* ─── Table / DataFrame Styling ─────────────────────────────────────────── */
    .dataframe {{
        background: var(--bg-card) !important;
        border: 1px solid var(--border) !important;
        border-radius: 16px !important;
        overflow: hidden;
    }}

    .dataframe th {{
        background: {'rgba(212, 175, 55, 0.08)' if is_dark else 'rgba(184, 134, 11, 0.06)'} !important;
        color: var(--text-primary) !important;
        font-family: 'Outfit', sans-serif !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        font-size: 11px !important;
        letter-spacing: 0.08em;
        padding: 16px !important;
    }}

    .dataframe td {{
        color: var(--text-primary) !important;
        background: var(--bg-card) !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 13px !important;
        padding: 14px 16px !important;
        border-bottom: 1px solid var(--border-light) !important;
    }}

    .dataframe tr:hover td {{
        background: {'rgba(212, 175, 55, 0.04)' if is_dark else 'rgba(184, 134, 11, 0.03)'} !important;
    }}

    [data-testid="stDataFrame"] canvas + div {{
        background: var(--bg-card) !important;
    }}

    /* ─── File Uploader Text ────────────────────────────────────────────────── */
    [data-testid="stFileUploader"] section div,
    [data-testid="stFileUploader"] section span,
    [data-testid="stFileUploader"] section small,
    [data-testid="stFileUploader"] section p {{
        color: var(--text-muted) !important;
        font-family: 'Outfit', sans-serif !important;
    }}

    [data-testid="stFileUploader"] section button {{
        color: var(--accent) !important;
        font-weight: 600 !important;
    }}

    [data-testid="stFileUploader"] [data-testid="stMarkdown"] {{
        color: var(--text-primary) !important;
    }}

    /* ─── Sidebar ───────────────────────────────────────────────────────────── */
    [data-testid="stSidebar"] {{
        background: {'rgba(12, 12, 15, 0.95)' if is_dark else 'rgba(253, 252, 247, 0.98)'} !important;
        border-right: 1px solid var(--border) !important;
    }}

    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] label {{
        color: var(--text-primary) !important;
    }}

    /* ─── Dividers ──────────────────────────────────────────────────────────── */
    hr {{
        border: none !important;
        height: 1px !important;
        background: linear-gradient(90deg, transparent, var(--border), transparent) !important;
        margin: 32px 0 !important;
    }}

    /* ─── Image Display ─────────────────────────────────────────────────────── */
    .stImage {{
        border-radius: 16px;
        overflow: hidden;
        box-shadow: 0 8px 32px rgba(0, 0, 0, {'0.4' if is_dark else '0.12'});
    }}

    .stImage img {{
        border-radius: 16px;
    }}

    /* ─── Column Gaps ───────────────────────────────────────────────────────── */
    [data-testid="stHorizontalBlock"] {{
        gap: 24px !important;
    }}

    /* ─── Loading Animation Override ────────────────────────────────────────── */
    .stSpinner {{
        color: var(--accent) !important;
    }}

    /* ─── Animation Utilities ───────────────────────────────────────────────── */
    @keyframes shimmer {{
        0% {{ background-position: -200% 0; }}
        100% {{ background-position: 200% 0; }}
    }}

    @keyframes pulse {{
        0%, 100% {{ opacity: 1; }}
        50% {{ opacity: 0.6; }}
    }}

    @keyframes float {{
        0%, 100% {{ transform: translateY(0); }}
        50% {{ transform: translateY(-8px); }}
    }}

    .shimmer {{
        background: linear-gradient(90deg, var(--bg-card) 0%, var(--bg-elevated) 50%, var(--bg-card) 100%);
        background-size: 200% 100%;
        animation: shimmer 1.5s ease-in-out infinite;
    }}

    /* ─── Special Gold Accent Classes ───────────────────────────────────────── */
    .gold-text {{
        background: var(--gradient-gold);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }}

    .gold-border {{
        border: 1px solid var(--accent) !important;
        box-shadow: 0 0 20px var(--accent-glow);
    }}

    .gold-glow {{
        box-shadow: 0 0 40px var(--accent-glow-strong),
                    0 0 80px var(--accent-glow);
    }}

    /* ─── Responsive Adjustments ────────────────────────────────────────────── */
    @media (max-width: 768px) {{
        .header-bar {{
            padding: 24px !important;
            border-radius: 20px;
        }}

        .header-title {{
            font-size: 24px !important;
        }}

        .stage-card {{
            padding: 24px !important;
        }}

        .stage-title {{
            font-size: 20px !important;
        }}

        .stat-grid {{
            grid-template-columns: repeat(2, 1fr) !important;
            gap: 12px;
        }}
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
    theme_icon = "☀️" if is_dark else "🌙"
    theme_text = "Light" if is_dark else "Dark"

    col1, col2 = st.columns([6, 1])
    with col1:
        st.markdown(f"""
        <div class="header-bar">
            <div>
                <div style="display: flex; align-items: center; gap: 16px;">
                    <div class="header-title">Hallmark QC</div>
                    <span class="header-badge">
                        <span style="font-size: 10px;">✦</span> Premium
                    </span>
                </div>
                <div class="header-subtitle">AI-Powered Quality Control • BIS Certified Validation</div>
            </div>
            <div style="position: relative; z-index: 2;">
                <div style="display: flex; align-items: center; gap: 12px;">
                    <div style="text-align: right;">
                        <div style="font-size: 11px; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.1em; font-weight: 600;">System Status</div>
                        <div style="font-size: 13px; color: var(--success); font-weight: 700; display: flex; align-items: center; gap: 6px; justify-content: flex-end;">
                            <span style="width: 8px; height: 8px; background: var(--success); border-radius: 50%; animation: pulse 2s ease-in-out infinite;"></span>
                            Online
                        </div>
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        if st.button(f"{theme_icon} {theme_text}", key="theme_toggle", use_container_width=True):
            toggle_theme()
            st.rerun()


def render_stage1():
    """Stage 1: Upload batch data (CSV/Excel)."""
    st.markdown("""
    <div class="stage-card" style="animation-delay: 0.1s;">
        <div style="display: flex; align-items: flex-start; gap: 20px;">
            <span class="stage-number">1</span>
            <div>
                <span class="stage-title">Upload Batch Data</span>
                <div class="stage-desc">Import your CSV or Excel file containing tag IDs and expected HUIDs for batch validation processing.</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
        <div style="margin-bottom: 16px;">
            <div style="font-size: 13px; font-weight: 600; color: var(--text-secondary); text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 12px;">
                📁 Import File
            </div>
        </div>
        """, unsafe_allow_html=True)

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
        st.markdown("""
        <div style="font-size: 13px; font-weight: 600; color: var(--text-secondary); text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 16px;">
            📋 Recent Batches
        </div>
        """, unsafe_allow_html=True)
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
    <div class="stage-card" style="animation-delay: 0.15s;">
        <div style="display: flex; align-items: flex-start; gap: 20px;">
            <span class="stage-number">2</span>
            <div>
                <span class="stage-title">Process Images</span>
                <div class="stage-desc">Upload hallmark images for AI-powered OCR analysis. Supports single or bulk image processing with automatic tag ID matching.</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("""
        <div style="font-size: 13px; font-weight: 600; color: var(--text-secondary); text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 16px;">
            📷 Single Image Upload
        </div>
        """, unsafe_allow_html=True)

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
        st.markdown("""
        <div style="font-size: 13px; font-weight: 600; color: var(--text-secondary); text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 8px;">
            📦 Bulk Image Upload
        </div>
        <div style="font-size: 12px; color: var(--text-muted); margin-bottom: 16px;">
            Upload multiple images. Filename should match tag ID (e.g., TAG001.jpg)
        </div>
        """, unsafe_allow_html=True)

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
    <div class="stage-card" style="animation-delay: 0.2s;">
        <div style="display: flex; align-items: flex-start; gap: 20px;">
            <span class="stage-number">3</span>
            <div>
                <span class="stage-title">View Results</span>
                <div class="stage-desc">Search, analyze, and export OCR validation results. Filter by tag ID or batch for detailed quality insights.</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("""
        <div style="font-size: 13px; font-weight: 600; color: var(--text-secondary); text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 16px;">
            🔎 Search
        </div>
        """, unsafe_allow_html=True)

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
        st.markdown("""
        <div style="font-size: 13px; font-weight: 600; color: var(--text-secondary); text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 16px;">
            📈 Results
        </div>
        """, unsafe_allow_html=True)

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
                image_bytes = fetch_image_bytes(result['image_url'])
                if image_bytes:
                    st.image(image_bytes, caption="Uploaded Image", use_container_width=True)

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
    <div class="stage-card" style="animation-delay: 0.25s;">
        <div style="display: flex; align-items: flex-start; gap: 20px;">
            <span class="stage-number" style="background: linear-gradient(135deg, #3B82F6 0%, #1D4ED8 100%); box-shadow: 0 8px 24px rgba(59, 130, 246, 0.35);">E</span>
            <div>
                <span class="stage-title">ERP Integration Monitor</span>
                <div class="stage-desc">Real-time monitoring dashboard for ERP system integration. Manage manual reviews, track statistics, and process pending items.</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Statistics section
    st.markdown("""
    <div style="font-size: 13px; font-weight: 600; color: var(--text-secondary); text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 16px;">
        📊 Today's Statistics
    </div>
    """, unsafe_allow_html=True)
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
    st.markdown("""
    <div style="font-size: 13px; font-weight: 600; color: var(--text-secondary); text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 16px;">
        👁️ Manual Review Queue
    </div>
    """, unsafe_allow_html=True)

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
                                    image_bytes = fetch_image_bytes(item["image_url"])
                                    if image_bytes:
                                        st.image(image_bytes, caption="Hallmark Image", use_container_width=True)
                                    else:
                                        st.caption("Image not available")
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
    st.markdown("""
    <div style="font-size: 13px; font-weight: 600; color: var(--text-secondary); text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 8px;">
        🧪 Quick Test: Upload & Process
    </div>
    <div style="font-size: 12px; color: var(--text-muted); margin-bottom: 16px;">
        Test the ERP integration by uploading an image directly
    </div>
    """, unsafe_allow_html=True)

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
        st.markdown("""
        <div class="result-card" style="border-left: 4px solid var(--warning); margin-bottom: 24px;">
            <div style="display: flex; align-items: center; gap: 12px;">
                <span style="font-size: 24px;">⚠️</span>
                <div>
                    <div style="font-weight: 700; color: var(--text-primary); margin-bottom: 4px;">API Connection Issue</div>
                    <div style="font-size: 13px; color: var(--text-muted);">Unable to connect to the backend server. Please ensure the API is running on port 8000.</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Main tabs with premium labels
    tab1, tab2, tab3, tab4 = st.tabs([
        "📤  Upload Data",
        "🔍  Process Images",
        "📊  View Results",
        "⚡  ERP Monitor"
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
