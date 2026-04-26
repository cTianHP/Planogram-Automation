import streamlit as st

def apply_custom_theme():
    st.markdown("""
    <style>

    /* ===== GLOBAL ===== */
    html, body, [class*="css"] {
        font-family: 'Segoe UI', sans-serif;
        background-color: #f9f9f9;
    }

    /* ===== HEADER ===== */
    h1 {
        color: #E53935;
        font-weight: 700;
    }

    h2, h3 {
        color: #333333;
    }

    /* ===== SECTION TITLE ===== */
    .stSubheader {
        color: #E53935 !important;
        font-weight: bold;
    }

    /* ===== METRIC CARD ===== */
    div[data-testid="metric-container"] {
        background: white;
        border-radius: 12px;
        padding: 15px;
        border-left: 5px solid #E53935;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }

    div[data-testid="metric-container"] label {
        color: #666 !important;
        font-weight: 600;
    }

    div[data-testid="metric-container"] div {
        color: #111;
        font-size: 20px;
        font-weight: bold;
    }

    /* ===== BUTTON ===== */
    .stButton > button {
        background-color: #E53935;
        color: white;
        border-radius: 8px;
        border: none;
        font-weight: bold;
    }

    .stButton > button:hover {
        background-color: #c62828;
        color: white;
    }

    /* ===== SELECTBOX ===== */
    div[data-baseweb="select"] {
        background-color: white;
        border-radius: 8px;
        border: 1px solid #ddd;
    }

    /* ===== INPUT ===== */
    input {
        border-radius: 8px !important;
    }

    /* ===== DATAFRAME ===== */
    .stDataFrame {
        background: white;
        border-radius: 10px;
        border: 1px solid #eee;
    }

    /* ===== INFO BOX ===== */
    .stAlert {
        border-left: 5px solid #FFC107;
        border-radius: 10px;
    }

    /* ===== HIGHLIGHT TEXT ===== */
    strong {
        color: #E53935;
    }

    /* ===== SCROLLBAR ===== */
    ::-webkit-scrollbar {
        width: 8px;
    }

    ::-webkit-scrollbar-thumb {
        background: #E53935;
        border-radius: 10px;
    }
    
    /* 🔥 EXPANDER TITLE SIZE */
    div[data-testid="stExpander"] summary {
        font-size: 20px !important;
        font-weight: bold;
        color: #E53935;
    }

    /* 🔥 OPTIONAL: ICON SIZE */
    div[data-testid="stExpander"] summary svg {
        transform: scale(1.2);
    }

    /* 🔥 SPACING BIAR LEBIH LEGA */
    div[data-testid="stExpander"] {
        margin-bottom: 10px;
    }

    </style>
    """, unsafe_allow_html=True)