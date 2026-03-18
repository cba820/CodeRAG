from typing import List, Optional

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm
from sentence_transformers import SentenceTransformer

from .models import RetrievedChunk
from .settings import Settings, get_settings


class CodeRetriever:
    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self.embedder = SentenceTransformer(self.settings.embedding_model)
        self.qdrant = QdrantClient(url=self.settings.qdrant_url)

    def search(self, query: str, top_k: Optional[int] = None) -> List[RetrievedChunk]:
        vector = self.embedder.encode(query, normalize_embeddings=True).tolist()
        results = self.qdrant.query_points(
            collection_name=self.settings.collection_name,
            query=vector,
            limit=top_k or self.settings.top_k,
            with_payload=True,
        )
        points = results.points if hasattr(results, 'points') else []
        chunks: List[RetrievedChunk] = []
        for point in points:
            payload = point.payload or {}
            chunks.append(
                RetrievedChunk(
                    text=payload.get('content', ''),
                    score=float(point.score),
                    payload=payload,
                )
            )
        return chunks

    @staticmethod
    def build_context(chunks: List[RetrievedChunk], max_chunks: int) -> str:
        selected = chunks[:max_chunks]
        blocks = []
        for idx, chunk in enumerate(selected, start=1):
            payload = chunk.payload
            header = (
                f"[{idx}] repo={payload.get('repo')} path={payload.get('path')} "
                f"lines={payload.get('start_line')}-{payload.get('end_line')} "
                f"language={payload.get('language')} score={chunk.score:.4f}"
            )
            blocks.append(f"{header}\n{chunk.text}")
        return '\n\n'.join(blocks)
