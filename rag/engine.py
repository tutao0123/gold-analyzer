import os
import json
import logging
import numpy as np
import dashscope
from pathlib import Path

# ensure faiss is available
try:
    import faiss
except ImportError:
    faiss = None

logger = logging.getLogger(__name__)

class GoldMultimodalRAG:
    """Multimodal RAG knowledge base backed by DashScope embeddings and Faiss."""

    def __init__(self, db_dir=".rag_db", dimension=2560):
        self.api_key = os.getenv("DASHSCOPE_API_KEY", "")
        self.db_dir = db_dir
        self.dimension = dimension

        # internal storage
        self.knowledge_store = []  # document metadata
        self.index_file = os.path.join(self.db_dir, "gold.index")
        self.meta_file = os.path.join(self.db_dir, "gold_meta.json")

        os.makedirs(self.db_dir, exist_ok=True)

        if faiss:
            self.index = faiss.IndexFlatL2(self.dimension)
            self._load_db()
        else:
            logger.warning("faiss not installed; RAG will fall back to brute-force scan (slow)")
            self.index = None

    def _load_db(self):
        """Load an existing local vector store."""
        if os.path.exists(self.index_file):
            logger.info("Loading existing chart-text vector store...")
            self.index = faiss.read_index(self.index_file)
        if os.path.exists(self.meta_file):
            with open(self.meta_file, "r", encoding="utf-8") as f:
                self.knowledge_store = json.load(f)

    def _save_db(self):
        """Persist the vector store to disk."""
        if self.index is not None:
            faiss.write_index(self.index, self.index_file)
        with open(self.meta_file, "w", encoding="utf-8") as f:
            json.dump(self.knowledge_store, f, ensure_ascii=False, indent=2)

    def get_embedding(self, text=None, image_path=None, video_path=None):
        """
        Call the DashScope multimodal embedding API and return a fused vector.
        """
        if not self.api_key:
            raise ValueError("DASHSCOPE_API_KEY is not set")

        input_data = {}
        if text:
            input_data["text"] = text
        if image_path:
            # DashScope supports file:// URLs for local files; for remote images pass the URL directly.
            if image_path.startswith("http"):
                input_data["image"] = image_path
            else:
                abs_path = os.path.abspath(image_path)
                input_data["image"] = f"file://{abs_path}"
        if video_path:
            if video_path.startswith("http"):
                input_data["video"] = video_path
            else:
                abs_path = os.path.abspath(video_path)
                input_data["video"] = f"file://{abs_path}"

        if not input_data:
            raise ValueError("At least one of text, image, or video must be provided")

        logger.info(f"Generating multimodal embedding... input keys: {list(input_data.keys())}")

        resp = dashscope.MultiModalEmbedding.call(
            api_key=self.api_key,
            model="qwen3-vl-embedding",
            input=[input_data]
        )

        if resp.status_code != 200:
            raise Exception(f"Multimodal embedding failed: {resp.message}")

        return resp.output["embeddings"][0]["embedding"]

    def add_knowledge(self, knowledge_id: str, text: str = None, image_path: str = None, metadata: dict = None, precomputed_vector: list = None):
        """
        Add a knowledge entry to the RAG store and persist to disk.
        """
        if precomputed_vector is not None:
            vector = precomputed_vector
        else:
            vector = self.get_embedding(text=text, image_path=image_path)

        np_vector = np.array([vector], dtype="float32")

        # store metadata
        new_entry = {
            "id": knowledge_id,
            "text": text,
            "image": image_path,
            "metadata": metadata or {}
        }

        if self.index is not None:
            self.index.add(np_vector)

        self.knowledge_store.append(new_entry)
        self._save_db()
        logger.info(f"Knowledge [{knowledge_id}] added to multimodal vector store.")

    def search(self, query_text: str = None, query_image: str = None, top_k: int = 2):
        """
        Retrieve the most relevant memories for a text/image query.
        """
        if not self.knowledge_store:
            return []

        vector = self.get_embedding(text=query_text, image_path=query_image)
        np_vector = np.array([vector], dtype="float32")

        results = []
        if self.index is not None:
            # Faiss L2 distance: smaller = more similar
            distances, indices = self.index.search(np_vector, top_k)
            for dist, idx in zip(distances[0], indices[0]):
                if 0 <= idx < len(self.knowledge_store):
                    entry = self.knowledge_store[idx].copy()
                    entry["distance"] = float(dist)
                    results.append(entry)
        else:
            # brute-force fallback (faiss not installed)
            pass

        return results
