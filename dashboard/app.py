"""
dashboard/app.py
================
Main Streamlit entry point for the Hybrid NIDS dashboard.

Run with:
    streamlit run dashboard/app.py

Pages:
    Detection  — upload CSV and run detection
    Analysis   — score distributions & model stats
    Chat       — RAG agent chat interface
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make src/ importable when running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

st.set_page_config(
    page_title="Hybrid NIDS",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ------------------------------------------------------------------
# Session state defaults
# ------------------------------------------------------------------
if "detection_results" not in st.session_state:
    st.session_state.detection_results = None
if "selected_row" not in st.session_state:
    st.session_state.selected_row = None
if "agent" not in st.session_state:
    st.session_state.agent = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# ------------------------------------------------------------------
# Sidebar navigation
# ------------------------------------------------------------------
st.sidebar.title("Hybrid NIDS")
st.sidebar.caption("Autoencoder + Isolation Forest + Agreement Layer")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    ["Detection", "Analysis", "Chat with Agent"],
    index=0,
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Models:** Autoencoder + Isolation Forest + XGBoost Arbitrator\n\n"
    "**Agreement Layer:** Tiered confidence arbitration\n\n"
    "**RAG Agent:** LangChain + ChromaDB + Sentence Transformers"
)


# ------------------------------------------------------------------
# Page routing
# ------------------------------------------------------------------
if page == "Detection":
    from dashboard.pages.detection import render
    render()

elif page == "Analysis":
    from dashboard.pages.analysis import render
    render()

elif page == "Chat with Agent":
    from dashboard.pages.chat import render
    render()
