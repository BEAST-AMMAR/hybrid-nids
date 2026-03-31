"""
dashboard/pages/analysis.py
============================
Analysis page: score distributions, confusion matrix, tier breakdown.
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd
import streamlit as st


def render() -> None:
    st.title("Model Analysis")

    results_df: pd.DataFrame | None = st.session_state.get("detection_results")
    y_true: np.ndarray | None = st.session_state.get("y_true")

    if results_df is None:
        st.info("No detection results yet. Go to the **Detection** page and run detection first.")
        return

    try:
        import plotly.express as px
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        st.error("plotly is required: pip install plotly")
        return

    n = len(results_df)
    st.caption(f"Analysing {n:,} samples")

    # ------------------------------------------------------------------
    # Score distributions
    # ------------------------------------------------------------------
    st.subheader("Score Distributions")

    tab1, tab2, tab3 = st.tabs(["Hybrid Score", "AE vs IF Scores", "Score Scatter"])

    with tab1:
        fig = px.histogram(
            results_df,
            x="hybrid_score",
            nbins=50,
            color="final_pred",
            color_discrete_map={0: "#28a745", 1: "#dc3545"},
            labels={"hybrid_score": "Hybrid Score", "final_pred": "Prediction"},
            title="Hybrid Score Distribution (0=Normal, 1=Attack)",
            barmode="overlay",
            opacity=0.7,
        )
        fig.add_vline(x=0.25, line_dash="dash", line_color="orange",
                     annotation_text="Low thresh (0.25)")
        fig.add_vline(x=0.75, line_dash="dash", line_color="red",
                     annotation_text="High thresh (0.75)")
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        fig2 = make_subplots(rows=1, cols=2,
                             subplot_titles=("AE Score", "IF Score"))
        for pred_val, color, name in [(0, "#28a745", "Normal"), (1, "#dc3545", "Attack")]:
            sub = results_df[results_df["final_pred"] == pred_val]
            fig2.add_trace(go.Histogram(x=sub["ae_score"], name=f"{name}",
                                        marker_color=color, opacity=0.7,
                                        nbinsx=40), row=1, col=1)
            fig2.add_trace(go.Histogram(x=sub["if_score"], name=f"{name}",
                                        marker_color=color, opacity=0.7,
                                        nbinsx=40, showlegend=False), row=1, col=2)
        fig2.update_layout(barmode="overlay", title="AE and IF Score Distributions")
        st.plotly_chart(fig2, use_container_width=True)

    with tab3:
        sample = results_df.sample(min(2000, n), random_state=42)
        fig3 = px.scatter(
            sample,
            x="ae_score",
            y="if_score",
            color=sample["final_pred"].map({0: "Normal", 1: "Attack"}),
            color_discrete_map={"Normal": "#28a745", "Attack": "#dc3545"},
            opacity=0.6,
            title="AE Score vs IF Score (sample of 2000)",
            labels={"x": "AE Score", "y": "IF Score"},
        )
        st.plotly_chart(fig3, use_container_width=True)

    # ------------------------------------------------------------------
    # Tier & Confidence breakdown
    # ------------------------------------------------------------------
    st.subheader("Agreement Layer Breakdown")
    col1, col2 = st.columns(2)

    with col1:
        tier_data = results_df["tier_used"].value_counts().reset_index()
        tier_data.columns = ["tier", "count"]
        tier_data["tier"] = tier_data["tier"].map({1: "Tier 1 (High Confidence)", 2: "Tier 2 (Uncertain)"})
        fig4 = px.pie(tier_data, values="count", names="tier",
                      title="Decisions by Tier",
                      color_discrete_sequence=["#007bff", "#ffc107"])
        st.plotly_chart(fig4, use_container_width=True)

    with col2:
        conf_data = results_df["confidence"].value_counts().reset_index()
        conf_data.columns = ["confidence", "count"]
        fig5 = px.pie(conf_data, values="count", names="confidence",
                      title="Confidence Level Distribution",
                      color_discrete_map={"HIGH": "#28a745", "MEDIUM": "#ffc107", "LOW": "#dc3545"})
        st.plotly_chart(fig5, use_container_width=True)

    # Agreement score distribution (Tier 2 only)
    tier2_df = results_df[results_df["tier_used"] == 2]
    if len(tier2_df) > 0:
        st.subheader("Tier 2 Agreement Score Distribution")
        fig6 = px.histogram(
            tier2_df, x="agreement_score", nbins=20,
            color="final_pred",
            color_discrete_map={0: "#28a745", 1: "#dc3545"},
            title=f"Agreement Score for {len(tier2_df):,} Tier 2 (uncertain) samples",
        )
        st.plotly_chart(fig6, use_container_width=True)

    # ------------------------------------------------------------------
    # Confusion matrix (if labels available)
    # ------------------------------------------------------------------
    if y_true is not None:
        st.subheader("Confusion Matrix")
        from sklearn.metrics import confusion_matrix, classification_report

        y_pred = results_df["final_pred"].values
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()

        fig7 = go.Figure(data=go.Heatmap(
            z=[[tn, fp], [fn, tp]],
            x=["Predicted Normal", "Predicted Attack"],
            y=["Actual Normal", "Actual Attack"],
            colorscale="Blues",
            text=[[f"TN={tn:,}", f"FP={fp:,}"], [f"FN={fn:,}", f"TP={tp:,}"]],
            texttemplate="%{text}",
            showscale=False,
        ))
        fig7.update_layout(title="Confusion Matrix")
        st.plotly_chart(fig7, use_container_width=True)

        # Classification report
        report = classification_report(
            y_true, y_pred,
            target_names=["Normal", "Attack"],
            output_dict=True,
        )
        report_df = pd.DataFrame(report).T.round(3)
        st.dataframe(report_df, use_container_width=True)

    # ------------------------------------------------------------------
    # Model explanations
    # ------------------------------------------------------------------
    st.subheader("About the Models")
    with st.expander("How does the hybrid scoring work?"):
        st.markdown(
            "**Hybrid Score = 0.6 × AE Score + 0.4 × IF Score**\n\n"
            "- **Autoencoder (AE)**: Trained only on normal traffic. "
            "High reconstruction error → anomalous.\n"
            "- **Isolation Forest (IF)**: Isolates outliers in feature space. "
            "Shorter isolation path → more anomalous.\n\n"
            "**Agreement Layer:**\n"
            "- **Tier 1** (clear-cut): Score > 0.75 → attack, Score < 0.25 → normal\n"
            "- **Tier 2** (borderline): Majority vote between AE, IF, and XGBoost arbitrator"
        )
