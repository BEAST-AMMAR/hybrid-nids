import os
import json
import numpy as np
from typing import List, Dict, Any, Optional, Tuple

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

try:
    import openai
except Exception:
    openai = None

from sklearn.neighbors import NearestNeighbors
import joblib

from .rag_knowledge import load_knowledge, normalize_entries, entry_text


class KBIndex:
    def __init__(self, kb_path: str, index_dir: Optional[str] = None):
        self.kb_path = os.path.abspath(kb_path)
        self.index_dir = os.path.abspath(index_dir or os.path.join(os.path.dirname(self.kb_path), '..', 'kb_index'))
        os.makedirs(self.index_dir, exist_ok=True)
        self.metadata_path = os.path.join(self.index_dir, 'metadata.json')
        self.emb_path = os.path.join(self.index_dir, 'embeddings.npy')
        self.model_path = os.path.join(self.index_dir, 'nn.joblib')
        self.entries: List[Dict[str, Any]] = []
        self.embeddings: Optional[np.ndarray] = None
        self.nn: Optional[NearestNeighbors] = None

    def _get_embedder(self):
        if os.getenv('OPENAI_API_KEY') and openai is not None:
            return 'openai'
        if SentenceTransformer is not None:
            return 'sbert'
        raise RuntimeError('No embedding provider available: install sentence-transformers or set OPENAI_API_KEY and install openai')

    def _embed_texts(self, texts: List[str]) -> np.ndarray:
        provider = self._get_embedder()
        if provider == 'openai':
            model = os.getenv('OPENAI_EMBEDDING_MODEL', 'text-embedding-3-small')
            openai.api_key = os.getenv('OPENAI_API_KEY')
            res = openai.Embedding.create(model=model, input=texts)
            vectors = [r['embedding'] for r in res['data']]
            return np.array(vectors, dtype=np.float32)
        else:
            model_name = os.getenv('SBERT_MODEL', 'all-MiniLM-L6-v2')
            model = SentenceTransformer(model_name)
            vecs = model.encode(texts, show_progress_bar=False)
            return np.array(vecs, dtype=np.float32)

    def build(self, overwrite: bool = False) -> None:
        if os.path.exists(self.emb_path) and os.path.exists(self.metadata_path) and not overwrite:
            self.load()
            return

        raw = load_knowledge(self.kb_path)
        entries = normalize_entries(raw, source=os.path.basename(self.kb_path))
        contents = [entry_text(e) for e in entries]
        emb = self._embed_texts(contents)

        np.save(self.emb_path, emb)
        with open(self.metadata_path, 'w', encoding='utf-8') as f:
            json.dump(entries, f, ensure_ascii=False, indent=2)

        nn = NearestNeighbors(n_neighbors=min(10, len(entries)), metric='cosine')
        nn.fit(emb)
        joblib.dump(nn, self.model_path)

        self.entries = entries
        self.embeddings = emb
        self.nn = nn

    def load(self) -> None:
        if not os.path.exists(self.emb_path) or not os.path.exists(self.metadata_path) or not os.path.exists(self.model_path):
            raise FileNotFoundError('Index artifacts missing; call build() first')
        self.embeddings = np.load(self.emb_path)
        with open(self.metadata_path, 'r', encoding='utf-8') as f:
            self.entries = json.load(f)
        self.nn = joblib.load(self.model_path)

    def query(self, q: str, top_k: int = 3) -> List[Tuple[Dict[str, Any], float]]:
        if self.nn is None:
            self.load()
        vec = self._embed_texts([q])
        dists, idxs = self.nn.kneighbors(vec, n_neighbors=min(top_k, len(self.entries)))
        results = []
        for dist, idx in zip(dists[0], idxs[0]):
            results.append((self.entries[int(idx)], float(1.0 - dist)))
        return results
