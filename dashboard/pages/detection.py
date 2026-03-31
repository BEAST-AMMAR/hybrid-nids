"""
dashboard/pages/detection.py
=============================
Detection page: upload CSV, run detection, show results.
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd
import streamlit as st


@st.cache_resource(show_spinner="Loading NIDS models...")
def load_pipeline():
    from src.pipeline import NIDSPipeline
    try:
        return NIDSPipeline(), None
    except Exception as e:
        return None, str(e)


def render() -> None:
    st.title("Network Intrusion Detection")
    st.markdown("Upload a feature CSV or use the NSL-KDD test data to run detection.")

    pipeline, err = load_pipeline()

    if err:
        st.error(f"Failed to load models: {err}")
        st.info("Run `python scripts/train_all.py` first to train and save models.")
        return

    # ------------------------------------------------------------------
    # Data source selection
    # ------------------------------------------------------------------
    col1, col2 = st.columns([2, 1])
    with col1:
        source = st.radio(
            "Data source",
            ["Use NSL-KDD test data (.npy)", "Upload CSV"],
            horizontal=True,
        )

    with col2:
        max_rows = st.number_input("Max rows to analyse", min_value=100, max_value=25000,
                                   value=1000, step=100)

    X = None
    y_true = None

    if source == "Use NSL-KDD test data (.npy)":
        try:
            from src.preprocessing import load_preprocessed
            data = load_preprocessed()
            X = data["X_test"][:max_rows]
            y_true = data["y_test"][:max_rows]
            st.success(f"Loaded {len(X):,} test samples from notebooks/X_test_scaled.npy")
        except Exception as e:
            st.error(f"Could not load preprocessed data: {e}")
            return

    else:
        uploaded = st.file_uploader("Upload scaled feature CSV (no label column)", type=["csv"])
        if uploaded is None:
            st.info("Upload a CSV file to continue. Columns should match the 116 scaled features.")
            return
        df = pd.read_csv(uploaded).head(max_rows)
        X = df.values.astype(np.float32)
        st.success(f"Loaded {len(X):,} rows, {X.shape[1]} features.")

    # ------------------------------------------------------------------
    # Run detection
    # ------------------------------------------------------------------
    if st.button("Run Detection", type="primary", use_container_width=True):
        with st.spinner("Running Hybrid NIDS detection..."):
            results_df = pipeline.detect(X)
            st.session_state.detection_results = results_df
            if y_true is not None:
                st.session_state.y_true = y_true
            else:
                st.session_state.y_true = None

    results_df = st.session_state.get("detection_results")
    if results_df is None:
        return

    # ------------------------------------------------------------------
    # Summary metrics
    # ------------------------------------------------------------------
    summary = pipeline.summarize(results_df)
    st.markdown("---")
    st.subheader("Detection Summary")

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Total Samples",  f"{summary['total']:,}")
    m2.metric("Attacks Detected", f"{summary['attacks']:,}",
              delta=f"{summary['attack_rate']*100:.1f}%")
    m3.metric("Normal Traffic", f"{summary['normals']:,}")
    m4.metric("Avg Hybrid Score", f"{summary['avg_hybrid_score']:.3f}")
    tier2 = summary['tier_breakdown'].get(2, 0)
    m5.metric("Tier 2 (Uncertain)", f"{tier2:,}",
              delta=f"{tier2/summary['total']*100:.1f}%",
              delta_color="inverse")

    # Accuracy if labels available
    y_true = st.session_state.get("y_true")
    if y_true is not None:
        from sklearn.metrics import accuracy_score, classification_report
        acc = accuracy_score(y_true, results_df["final_pred"].values)
        st.metric("Accuracy", f"{acc*100:.2f}%")

    # ------------------------------------------------------------------
    # Results table
    # ------------------------------------------------------------------
    st.markdown("---")
    st.subheader("Detection Results")

    display_df = results_df.copy()
    if y_true is not None:
        display_df.insert(0, "true_label", y_true)
        display_df.insert(1, "correct", (display_df["true_label"] == display_df["final_pred"]).map({True: "✓", False: "✗"}))

    # Color coding helper
    def highlight_row(row):
        if row.get("final_pred") == 1 and row.get("confidence") == "HIGH":
            return ["background-color: #ffcccc"] * len(row)
        elif row.get("final_pred") == 1 and row.get("confidence") in ("MEDIUM", "LOW"):
            return ["background-color: #fff3cd"] * len(row)
        elif row.get("confidence") == "HIGH":
            return ["background-color: #d4edda"] * len(row)
        return [""] * len(row)

    st.dataframe(
        display_df.style.apply(highlight_row, axis=1),
        height=400,
        use_container_width=True,
    )

    # Download button
    csv_data = results_df.to_csv(index=False)
    st.download_button(
        "Download Results CSV",
        data=csv_data,
        file_name="nids_detection_results.csv",
        mime="text/csv",
    )

    # ------------------------------------------------------------------
    # Row selection for Chat context
    # ------------------------------------------------------------------
    st.markdown("---")
    st.subheader("Investigate a Detection")
    row_idx = st.number_input(
        "Select row index to investigate",
        min_value=0, max_value=len(results_df) - 1, value=0
    )
    selected = results_df.iloc[row_idx].to_dict()
    st.session_state.selected_row = selected

    col_a, col_b = st.columns(2)
    with col_a:
        pred_label = "ATTACK" if selected["final_pred"] == 1 else "NORMAL"
        colour = "red" if selected["final_pred"] == 1 else "green"
        st.markdown(
            f"**Prediction:** :{colour}[{pred_label}]  \n"
            f"**Confidence:** {selected['confidence']}  \n"
            f"**Tier Used:** {selected['tier_used']}  \n"
            f"**Agreement Score:** {selected['agreement_score']:.2f}"
        )
    with col_b:
        st.markdown(
            f"**Hybrid Score:** {selected['hybrid_score']:.4f}  \n"
            f"**AE Score:** {selected['ae_score']:.4f}  \n"
            f"**IF Score:** {selected['if_score']:.4f}  \n"
            f"**XGB Pred:** {selected.get('xgb_pred', 'N/A')}"
        )

    if st.button("Explain this Detection with AI Agent", use_container_width=True):
        st.session_state.explain_detection = selected
        st.info("Switch to the 'Chat with Agent' page to see the explanation.")
