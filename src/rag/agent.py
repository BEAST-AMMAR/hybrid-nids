"""
src/rag/agent.py
================
Agentic RAG assistant for the Hybrid NIDS.

Supports three configurable LLM backends (via .env):
  LLM_BACKEND=ollama     → ChatOllama  (local, no API key)
  LLM_BACKEND=openai     → ChatOpenAI
  LLM_BACKEND=anthropic  → ChatAnthropic

Important notes for local models (qwen3, deepseek-r1, etc.):
  Models that emit <think>...</think> "reasoning" blocks are handled
  automatically — the think blocks are stripped before the ReAct parser
  sees the output, so tool use works correctly.

The agent uses LangChain's ReAct framework with four tools:
  - search_knowledge_base
  - get_attack_info
  - get_mitre_mapping
  - get_recommendations

Usage:
    from src.rag.agent import NIDSAgent
    agent = NIDSAgent()
    response = agent.chat("What is a neptune attack?")
    response = agent.explain_detection(detection_result_dict)
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_core.messages import AIMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda

from src.rag.knowledge_base import KnowledgeBase
from src.rag.tools import NIDS_TOOLS, set_knowledge_base
from src.rag.prompts import SYSTEM_PROMPT, DETECTION_ANALYSIS_PROMPT


# Load .env from project root
_root = Path(__file__).resolve().parent.parent.parent
load_dotenv(_root / ".env")

# Strips <think>...</think> blocks emitted by qwen3, deepseek-r1, etc.
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def _strip_think(msg: Any) -> Any:
    """Post-process LLM output: strip think tags, return clean AIMessage."""
    if hasattr(msg, "content"):
        clean = _THINK_RE.sub("", msg.content).lstrip()
        return AIMessage(content=clean)
    return msg


# ------------------------------------------------------------------
# Hardcoded ReAct prompt — no LangChain Hub pull (avoids network hang)
# ------------------------------------------------------------------
REACT_PROMPT = PromptTemplate.from_template(
    "Answer the following question as best you can. "
    "You have access to the following tools:\n\n"
    "{tools}\n\n"
    "Use the following format EXACTLY:\n\n"
    "Question: the input question you must answer\n"
    "Thought: you should always think about what to do\n"
    "Action: the action to take, should be one of [{tool_names}]\n"
    "Action Input: the input to the action\n"
    "Observation: the result of the action\n"
    "... (this Thought/Action/Action Input/Observation can repeat up to 4 times)\n"
    "Thought: I now know the final answer\n"
    "Final Answer: the final answer to the original input question\n\n"
    "Begin!\n\n"
    "Question: {input}\n"
    "Thought:{agent_scratchpad}"
)


# ------------------------------------------------------------------
# Agent
# ------------------------------------------------------------------
class NIDSAgent:
    """
    Agentic RAG assistant for network intrusion detection.

    Parameters
    ----------
    knowledge_base : KnowledgeBase instance (built or loaded)
    auto_build_kb  : if True and KB is empty, auto-build on first use
    """

    def __init__(
        self,
        knowledge_base: KnowledgeBase | None = None,
        auto_build_kb: bool = True,
    ):
        self.kb = knowledge_base or KnowledgeBase()
        if auto_build_kb:
            self.kb.build()

        set_knowledge_base(self.kb)
        self._raw_llm, self.llm = self._build_backends()
        self._agent_executor = self._build_agent()

    def _build_backends(self):
        """Return (raw_llm, think_stripped_llm)."""
        backend = os.getenv("LLM_BACKEND", "ollama").lower()

        if backend == "ollama":
            from langchain_ollama import ChatOllama
            raw = ChatOllama(
                base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
                model=os.getenv("OLLAMA_MODEL", "llama3.2"),
                temperature=0.1,
            )
        elif backend == "openai":
            from langchain_openai import ChatOpenAI
            raw = ChatOpenAI(
                model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                api_key=os.getenv("OPENAI_API_KEY"),
                temperature=0.1,
            )
        elif backend == "anthropic":
            from langchain_anthropic import ChatAnthropic
            raw = ChatAnthropic(
                model=os.getenv("ANTHROPIC_MODEL", "claude-haiku-3-5"),
                api_key=os.getenv("ANTHROPIC_API_KEY"),
                temperature=0.1,
            )
        else:
            raise ValueError(f"Unknown LLM_BACKEND='{backend}'.")

        # Pipe through the think-tag stripper — safe no-op for models that don't think
        stripped = raw | RunnableLambda(_strip_think)
        return raw, stripped

    # ------------------------------------------------------------------
    # Agent construction
    # ------------------------------------------------------------------
    def _build_agent(self):
        from langchain.agents import create_react_agent, AgentExecutor

        agent = create_react_agent(
            llm=self.llm,
            tools=NIDS_TOOLS,
            prompt=REACT_PROMPT,
        )
        return AgentExecutor(
            agent=agent,
            tools=NIDS_TOOLS,
            verbose=False,
            max_iterations=6,
            handle_parsing_errors=True,
            return_intermediate_steps=False,
        )

    # ------------------------------------------------------------------
    # Chat
    # ------------------------------------------------------------------
    def chat(
        self,
        message: str,
        detection_context: dict | None = None,
    ) -> str:
        """
        Send a message and get a response from the NIDS agent.

        Parameters
        ----------
        message           : user query
        detection_context : optional detection result dict for context injection
        """
        full_input = f"{SYSTEM_PROMPT}\n\n"
        if detection_context:
            full_input += f"DETECTION CONTEXT:\n{self._format_context(detection_context)}\n\n"
        full_input += f"USER QUESTION: {message}"

        try:
            result = self._agent_executor.invoke({"input": full_input})
            return result.get("output", str(result))
        except Exception:
            return self._rag_answer(message, detection_context)

    def explain_detection(self, detection_result: dict) -> str:
        """Auto-generate explanation for a detection result dict."""
        ctx = self._format_context(detection_result)
        prompt = DETECTION_ANALYSIS_PROMPT.format(detection_context=ctx)
        return self.chat(prompt)

    # ------------------------------------------------------------------
    # Fallback: direct RAG-augmented LLM answer (no tool loop)
    # ------------------------------------------------------------------
    def _rag_answer(self, message: str, detection_context: dict | None = None) -> str:
        from langchain_core.messages import SystemMessage, HumanMessage

        docs = self.kb.search(message, k=3)
        context_block = "\n\n".join(d["text"] for d in docs)

        msgs = [SystemMessage(content=SYSTEM_PROMPT)]
        if context_block:
            msgs.append(HumanMessage(content=f"Relevant knowledge:\n{context_block}"))
        if detection_context:
            msgs.append(HumanMessage(
                content=f"Detection context:\n{self._format_context(detection_context)}"
            ))
        msgs.append(HumanMessage(content=message))

        raw = self._raw_llm.invoke(msgs)
        return _THINK_RE.sub("", raw.content).strip()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _format_context(ctx: dict) -> str:
        return "\n".join(f"  {k}: {v}" for k, v in ctx.items() if v != -1)

    def rebuild_knowledge_base(self) -> None:
        self.kb.build(force_rebuild=True)

    def kb_stats(self) -> dict:
        return self.kb.stats()
