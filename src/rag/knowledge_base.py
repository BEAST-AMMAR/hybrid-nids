"""
src/rag/knowledge_base.py
=========================
ChromaDB-backed vector knowledge base for the NIDS RAG assistant.

Embeddings: sentence-transformers/all-MiniLM-L6-v2 (local, no API key)
Storage:    persistent ChromaDB collection in vector_store/

Usage:
    kb = KnowledgeBase()
    kb.build()                          # ingest all documents (once)
    results = kb.search("neptune DoS")  # retrieve top-k relevant chunks
"""
from __future__ import annotations

from pathlib import Path
from typing import List

from src.utils import get_project_root, load_config
from src.rag.documents import get_all_documents


COLLECTION_NAME = "nids_knowledge"


class KnowledgeBase:
    """
    Persistent ChromaDB knowledge base with sentence-transformer embeddings.

    Parameters
    ----------
    vector_store_dir : path to persist the ChromaDB store.
                       Defaults to <project_root>/vector_store
    config           : optional config dict override
    """

    def __init__(
        self,
        vector_store_dir: str | Path | None = None,
        config: dict | None = None,
    ):
        root = get_project_root()
        cfg  = config or load_config()
        rag_cfg = cfg.get("rag", {})

        self.vector_store_dir = Path(vector_store_dir) if vector_store_dir else root / "vector_store"
        self.vector_store_dir.mkdir(parents=True, exist_ok=True)

        self.top_k = rag_cfg.get("top_k", 5)
        self.embedding_model_name = rag_cfg.get("embedding_model", "all-MiniLM-L6-v2")

        self._client = None
        self._collection = None
        self._embedding_fn = None

    # ------------------------------------------------------------------
    # Lazy initialisation
    # ------------------------------------------------------------------
    def _init(self) -> None:
        if self._client is not None:
            return
        try:
            import chromadb
            from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
        except ImportError as e:
            raise ImportError(
                "chromadb and sentence-transformers are required: "
                "pip install chromadb sentence-transformers"
            ) from e

        self._embedding_fn = SentenceTransformerEmbeddingFunction(
            model_name=self.embedding_model_name
        )
        self._client = chromadb.PersistentClient(path=str(self.vector_store_dir))
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=self._embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )

    # ------------------------------------------------------------------
    # Build — ingest all documents
    # ------------------------------------------------------------------
    def build(self, force_rebuild: bool = False) -> None:
        """
        Ingest all knowledge base documents into ChromaDB.

        Parameters
        ----------
        force_rebuild : if True, delete and recreate the collection
        """
        self._init()
        import chromadb

        if force_rebuild:
            try:
                self._client.delete_collection(COLLECTION_NAME)
            except Exception:
                pass
            self._collection = self._client.get_or_create_collection(
                name=COLLECTION_NAME,
                embedding_function=self._embedding_fn,
                metadata={"hnsw:space": "cosine"},
            )

        existing_count = self._collection.count()
        documents = get_all_documents()

        if existing_count >= len(documents) and not force_rebuild:
            print(f"Knowledge base already has {existing_count} documents. "
                  f"Skipping build. Use force_rebuild=True to rebuild.")
            return

        print(f"Building knowledge base with {len(documents)} documents...")

        # Upsert in batches to avoid memory issues
        batch_size = 50
        for i in range(0, len(documents), batch_size):
            batch = documents[i: i + batch_size]
            self._collection.upsert(
                ids=[d["id"] for d in batch],
                documents=[d["text"] for d in batch],
                metadatas=[d["metadata"] for d in batch],
            )

        print(f"Knowledge base built: {self._collection.count()} documents indexed.")

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------
    def search(
        self,
        query: str,
        k: int | None = None,
        filter_type: str | None = None,
    ) -> List[dict]:
        """
        Retrieve top-k relevant documents for a query.

        Parameters
        ----------
        query       : natural language query string
        k           : number of results (defaults to self.top_k)
        filter_type : optional filter on metadata 'type' field
                      (e.g. 'attack_doc', 'mitre_tactic', 'security_concept', 'model_explain')

        Returns
        -------
        List of dicts with keys: id, text, metadata, distance
        """
        self._init()
        k = k or self.top_k

        where = {"type": filter_type} if filter_type else None
        query_kwargs = dict(
            query_texts=[query],
            n_results=min(k, self._collection.count()),
        )
        if where:
            query_kwargs["where"] = where

        results = self._collection.query(**query_kwargs)

        output = []
        for i in range(len(results["ids"][0])):
            output.append({
                "id":       results["ids"][0][i],
                "text":     results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i],
            })
        return output

    def get_attack_info(self, attack_name: str) -> dict | None:
        """
        Directly retrieve the document for a specific attack by name.
        Returns None if not found.
        """
        self._init()
        try:
            result = self._collection.get(
                ids=[f"attack_{attack_name.lower().replace('-', '_')}"]
            )
            if result["ids"]:
                return {
                    "id":       result["ids"][0],
                    "text":     result["documents"][0],
                    "metadata": result["metadatas"][0],
                }
        except Exception:
            pass
        # Fall back to semantic search
        results = self.search(f"{attack_name} attack NSL-KDD", k=1)
        return results[0] if results else None

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------
    def stats(self) -> dict:
        """Return stats about the knowledge base."""
        self._init()
        return {
            "total_documents": self._collection.count(),
            "collection_name": COLLECTION_NAME,
            "vector_store_dir": str(self.vector_store_dir),
            "embedding_model": self.embedding_model_name,
        }
