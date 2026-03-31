"""
src/rag/tools.py
================
LangChain tools for the NIDS RAG agent.

Tool 1: SearchKnowledgeBase   — semantic RAG retrieval
Tool 2: GetAttackInfo         — direct attack type lookup
Tool 3: GetMITREMapping       — MITRE tactic/technique lookup
Tool 4: GetRecommendations    — incident response recommendations
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from langchain.tools import tool

from src.utils import get_project_root

if TYPE_CHECKING:
    from src.rag.knowledge_base import KnowledgeBase


# We use a module-level kb reference that the agent injects before use
_kb: "KnowledgeBase | None" = None


def set_knowledge_base(kb: "KnowledgeBase") -> None:
    """Inject the KnowledgeBase instance into this module."""
    global _kb
    _kb = kb


def _get_kb() -> "KnowledgeBase":
    if _kb is None:
        raise RuntimeError("KnowledgeBase not initialised. Call set_knowledge_base() first.")
    return _kb


# ------------------------------------------------------------------
# Tool 1: Semantic search over the full knowledge base
# ------------------------------------------------------------------
@tool
def search_knowledge_base(query: str) -> str:
    """
    Search the NIDS knowledge base for information about network attacks,
    security concepts, or model behaviour.
    Use this for general questions about attack types, MITRE ATT&CK,
    detection methodology, and incident response.
    Input: a natural language search query string.
    """
    kb = _get_kb()
    results = kb.search(query, k=4)
    if not results:
        return "No relevant information found for your query."

    output_parts = []
    for r in results:
        meta = r["metadata"]
        doc_type = meta.get("type", "")
        header = f"[{doc_type.upper()}]"
        if "attack" in meta:
            header += f" {meta['attack']}"
        output_parts.append(f"{header}\n{r['text']}")

    return "\n\n---\n\n".join(output_parts)


# ------------------------------------------------------------------
# Tool 2: Direct attack info lookup by name
# ------------------------------------------------------------------
@tool
def get_attack_info(attack_name: str) -> str:
    """
    Get detailed information about a specific NSL-KDD attack type by name.
    Use this when you know the exact attack name (e.g., 'neptune', 'buffer_overflow',
    'guess_passwd', 'ipsweep'). Returns description, MITRE mapping, indicators,
    and response steps.
    Input: the attack name (lowercase, e.g. 'neptune').
    """
    kb = _get_kb()
    result = kb.get_attack_info(attack_name)
    if result is None:
        return f"No information found for attack '{attack_name}'. Try search_knowledge_base."
    return result["text"]


# ------------------------------------------------------------------
# Tool 3: MITRE ATT&CK category mapping
# ------------------------------------------------------------------
@tool
def get_mitre_mapping(query: str) -> str:
    """
    Look up MITRE ATT&CK tactic and technique mappings for network attacks.
    Use this to find which MITRE tactics/techniques correspond to a given
    attack category or behaviour.
    Input: attack category name or tactic name (e.g., 'DoS', 'Probe', 'privilege escalation').
    """
    mapping_path = get_project_root() / "config" / "mitre_mapping.json"
    if not mapping_path.exists():
        return "MITRE mapping file not found."

    with open(mapping_path, "r") as f:
        mapping = json.load(f)

    query_lower = query.lower()
    results = []

    # Check categories
    for category, cat_data in mapping.get("categories", {}).items():
        if (query_lower in category.lower() or
                query_lower in cat_data.get("description", "").lower() or
                query_lower in cat_data.get("mitre_tactic", "").lower()):
            results.append(
                f"Category: {category}\n"
                f"MITRE Tactic: {cat_data.get('mitre_tactic')} ({cat_data.get('mitre_tactic_id')})\n"
                f"Description: {cat_data.get('description')}\n"
                f"Attacks: {', '.join(cat_data.get('attacks', {}).keys())}"
            )
            continue

        # Check individual attacks within category
        for attack_name, attack in cat_data.get("attacks", {}).items():
            if query_lower in attack_name.lower():
                sub = attack.get("subtechnique") or ""
                sub_id = attack.get("subtechnique_id") or ""
                entry = (
                    f"Attack: {attack_name} (Category: {category})\n"
                    f"MITRE Tactic: {cat_data.get('mitre_tactic')} ({cat_data.get('mitre_tactic_id')})\n"
                    f"MITRE Technique: {attack.get('mitre_technique')} ({attack.get('mitre_technique_id')})\n"
                )
                if sub:
                    entry += f"Sub-technique: {sub} ({sub_id})\n"
                entry += f"Severity: {attack.get('severity', 'unknown')}"
                results.append(entry)

    if not results:
        return f"No MITRE mapping found for '{query}'. Try a different term."

    return "\n\n".join(results)


# ------------------------------------------------------------------
# Tool 4: Incident response recommendations
# ------------------------------------------------------------------
@tool
def get_recommendations(attack_type: str) -> str:
    """
    Get specific incident response and remediation recommendations for a
    given attack type or category. Returns actionable steps to contain,
    investigate, and remediate the threat.
    Input: attack name or category (e.g., 'neptune', 'DoS', 'guess_passwd', 'rootkit').
    """
    mapping_path = get_project_root() / "config" / "mitre_mapping.json"
    if not mapping_path.exists():
        return "MITRE mapping file not found."

    with open(mapping_path, "r") as f:
        mapping = json.load(f)

    query_lower = attack_type.lower()
    results = []

    for category, cat_data in mapping.get("categories", {}).items():
        # Match by category
        if query_lower == category.lower():
            for atk_name, attack in cat_data.get("attacks", {}).items():
                steps = attack.get("response", [])
                if steps:
                    results.append(
                        f"[{atk_name}] Severity: {attack.get('severity', 'unknown')}\n"
                        + "\n".join(f"  - {s}" for s in steps)
                    )
            break

        # Match by attack name
        for attack_name, attack in cat_data.get("attacks", {}).items():
            if query_lower == attack_name.lower() or query_lower in attack_name.lower():
                steps = attack.get("response", [])
                results.append(
                    f"[{attack_name}] (Category: {category}) "
                    f"Severity: {attack.get('severity', 'unknown')}\n"
                    "Recommended Response Steps:\n"
                    + "\n".join(f"  {i+1}. {s}" for i, s in enumerate(steps))
                )

    if not results:
        # Fall back to semantic search
        kb = _get_kb()
        docs = kb.search(f"{attack_type} response steps recommendations", k=2)
        if docs:
            return "\n\n".join(d["text"] for d in docs)
        return f"No specific recommendations found for '{attack_type}'."

    return "\n\n".join(results)


# ------------------------------------------------------------------
# Export tool list
# ------------------------------------------------------------------
NIDS_TOOLS = [
    search_knowledge_base,
    get_attack_info,
    get_mitre_mapping,
    get_recommendations,
]
