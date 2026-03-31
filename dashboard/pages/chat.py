"""
dashboard/pages/chat.py
=======================
Chat page: RAG agent interface for explaining detections and answering questions.
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st


@st.cache_resource(show_spinner="Initialising NIDS AI Agent...")
def load_agent():
    from src.rag.agent import NIDSAgent
    try:
        agent = NIDSAgent()
        return agent, None
    except Exception as e:
        return None, str(e)


def render() -> None:
    st.title("NIDS AI Security Agent")
    st.markdown(
        "Ask questions about detected attacks, MITRE ATT&CK mappings, "
        "response steps, or have the agent explain a specific detection."
    )

    # Agent status
    agent, err = load_agent()

    if err:
        st.error(f"Failed to load agent: {err}")
        st.markdown(
            "**Troubleshooting:**\n"
            "- Ensure your LLM backend is configured in `.env`\n"
            "- For Ollama: run `ollama pull llama3.2` and start the server\n"
            "- For OpenAI/Anthropic: set the API key in `.env`"
        )
        return

    kb_stats = agent.kb_stats()
    st.caption(
        f"Knowledge base: {kb_stats['total_documents']} documents | "
        f"Model: {kb_stats['embedding_model']} | "
        f"LLM: configured via .env"
    )

    # ------------------------------------------------------------------
    # Auto-explain detection if triggered from Detection page
    # ------------------------------------------------------------------
    explain_ctx = st.session_state.pop("explain_detection", None)
    if explain_ctx is not None:
        with st.spinner("Generating AI explanation..."):
            explanation = agent.explain_detection(explain_ctx)
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": f"**Detection Analysis (auto-generated):**\n\n{explanation}",
        })

    # ------------------------------------------------------------------
    # Quick action buttons
    # ------------------------------------------------------------------
    st.subheader("Quick Actions")
    q_cols = st.columns(4)
    quick_questions = [
        ("What is a neptune attack?", "neptune"),
        ("Explain buffer overflow U2R", "buffer_overflow U2R"),
        ("DoS vs Probe difference?", "DoS vs Probe"),
        ("How does the agreement layer work?", "agreement layer"),
    ]
    for i, (label, query) in enumerate(quick_questions):
        if q_cols[i].button(label, use_container_width=True):
            st.session_state.chat_history.append({"role": "user", "content": label})
            with st.spinner("Thinking..."):
                response = agent.chat(query)
            st.session_state.chat_history.append({"role": "assistant", "content": response})

    # Selected detection context button
    selected = st.session_state.get("selected_row")
    if selected:
        pred_label = "ATTACK" if selected.get("final_pred") == 1 else "NORMAL"
        if st.button(
            f"Explain selected row (pred={pred_label}, score={selected.get('hybrid_score', 0):.3f})",
            use_container_width=True, type="secondary"
        ):
            with st.spinner("Analysing detection..."):
                explanation = agent.explain_detection(selected)
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": f"**Selected Row Analysis:**\n\n{explanation}",
            })

    # ------------------------------------------------------------------
    # Chat history display
    # ------------------------------------------------------------------
    st.subheader("Conversation")
    chat_container = st.container(height=500)

    with chat_container:
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    # ------------------------------------------------------------------
    # Chat input
    # ------------------------------------------------------------------
    if user_input := st.chat_input("Ask the NIDS agent anything..."):
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        with st.spinner("Thinking..."):
            ctx = st.session_state.get("selected_row")
            response = agent.chat(user_input, detection_context=ctx)

        st.session_state.chat_history.append({"role": "assistant", "content": response})
        st.rerun()

    # ------------------------------------------------------------------
    # Clear chat button
    # ------------------------------------------------------------------
    col1, col2 = st.columns([4, 1])
    with col2:
        if st.button("Clear Chat", type="secondary"):
            st.session_state.chat_history = []
            st.rerun()
