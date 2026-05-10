import json
import os
from datetime import date
from typing import List, Dict, Any, Tuple


def load_knowledge(path: str) -> List[Dict[str, Any]]:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def normalize_entries(entries: List[Dict[str, Any]], source: str = None) -> List[Dict[str, Any]]:
    today = date.today().isoformat()
    normalized = []
    for i, e in enumerate(entries, start=1):
        ent = dict(e)
        if 'id' not in ent:
            key = ent.get('attack_type') or ent.get('name') or f'entry_{i}'
            slug = ''.join(c if c.isalnum() else '_' for c in key.lower())
            ent['id'] = f'kb_{i}_{slug}'
        if 'source' not in ent:
            ent['source'] = source or 'knowledge_base.json'
        if 'last_updated' not in ent:
            ent['last_updated'] = today
        normalized.append(ent)
    return normalized


def entry_text(entry: Dict[str, Any]) -> str:
    parts = []
    for k in ('name', 'attack_type', 'description'):
        v = entry.get(k)
        if v:
            parts.append(str(v))
    indicators = entry.get('indicators') or []
    parts.extend(indicators)
    feats = entry.get('feature_explanation') or {}
    parts.extend(feats.values())
    return '\n'.join(parts).lower()


class SimpleRAG:
    """Simple RAG-style retriever using keyword overlap scoring.

    This is intentionally lightweight and dependency-free so it works
    without additional packages. It normalizes entries, builds an
    in-memory index and scores entries by query token occurrences.
    """

    def __init__(self, knowledge_path: str):
        self.path = os.path.abspath(knowledge_path)
        self.entries = normalize_entries(load_knowledge(self.path), source=os.path.basename(self.path))
        self.index = [entry_text(e) for e in self.entries]

    def query(self, q: str, top_k: int = 3) -> List[Tuple[Dict[str, Any], float]]:
        q_tokens = [t for t in q.lower().split() if t]
        scores = []
        for ent, text in zip(self.entries, self.index):
            score = sum(text.count(tok) for tok in q_tokens)
            if score > 0:
                scores.append((ent, float(score)))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Simple RAG retriever for knowledge JSON')
    parser.add_argument('--kb', default=os.path.join(os.path.dirname(__file__), '..', '..', 'knowledge_base.json'))
    parser.add_argument('--top', type=int, default=3)
    args = parser.parse_args()

    kb_path = os.path.abspath(args.kb)
    if not os.path.exists(kb_path):
        print('Knowledge file not found:', kb_path)
        raise SystemExit(1)

    rag = SimpleRAG(kb_path)
    print(f'Loaded {len(rag.entries)} entries from {kb_path}')
    try:
        while True:
            q = input('\nQuery> ').strip()
            if not q:
                continue
            results = rag.query(q, top_k=args.top)
            if not results:
                print('No matches found.')
                continue
            for ent, score in results:
                print(f"\n- id: {ent['id']}  score: {score}\n  name: {ent.get('name')}\n  type: {ent.get('attack_type')}\n  description: {ent.get('description')[:200]}")
    except (KeyboardInterrupt, EOFError):
        print('\nExiting')
