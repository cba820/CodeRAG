import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm
from sentence_transformers import SentenceTransformer

from .github_loader import GitHubRepoLoader
from .settings import Settings, get_settings
from .utils import chunk_text, detect_language, sha1_as_int, should_skip_path

logger = logging.getLogger(__name__)


class RepoIndexer:
    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self.embedder = SentenceTransformer(self.settings.embedding_model)
        self.embedding_size = self.embedder.get_sentence_embedding_dimension()
        self.qdrant = QdrantClient(url=self.settings.qdrant_url)
        self.github_loader = GitHubRepoLoader(self.settings)
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        collections = self.qdrant.get_collections().collections
        existing = {item.name for item in collections}
        if self.settings.collection_name in existing:
            return
        self.qdrant.create_collection(
            collection_name=self.settings.collection_name,
            vectors_config=qm.VectorParams(size=self.embedding_size, distance=qm.Distance.COSINE),
        )

    def recreate_collection(self) -> None:
        self.qdrant.recreate_collection(
            collection_name=self.settings.collection_name,
            vectors_config=qm.VectorParams(size=self.embedding_size, distance=qm.Distance.COSINE),
        )

    def delete_repo_points(self, repo_name: str) -> None:
        self.qdrant.delete(
            collection_name=self.settings.collection_name,
            points_selector=qm.FilterSelector(
                filter=qm.Filter(
                    must=[qm.FieldCondition(key='repo', match=qm.MatchValue(value=repo_name))]
                )
            ),
        )

    def _iter_local_repo_documents(self, repo_name: str) -> Iterable[Dict[str, Any]]:
        repo_root = Path(self.settings.repos_root) / repo_name
        if not repo_root.exists() or not repo_root.is_dir():
            raise FileNotFoundError(f'No existe el repositorio local: {repo_root}')

        for file_path in repo_root.rglob('*'):
            if not file_path.is_file():
                continue
            relative_path = file_path.relative_to(repo_root).as_posix()
            if should_skip_path(relative_path, self.settings.include_extensions, self.settings.exclude_dirs):
                continue
            try:
                content = file_path.read_text(encoding='utf-8', errors='ignore')
            except Exception as exc:
                logger.warning('No se pudo leer %s: %s', file_path, exc)
                continue
            if not content.strip():
                continue
            yield {
                'repo': repo_name,
                'path': relative_path,
                'content': content,
                'sha': None,
                'ref': None,
                'source_type': 'local',
            }

    def _build_points_for_documents(self, documents: Iterable[Dict[str, Any]]) -> List[qm.PointStruct]:
        points: List[qm.PointStruct] = []
        texts: List[str] = []
        metadata_batch: List[Dict[str, Any]] = []

        for doc in documents:
            language = detect_language(doc['path'])
            chunks = chunk_text(
                text=doc['content'],
                chunk_size=self.settings.chunk_size,
                chunk_overlap=self.settings.chunk_overlap,
                language=language,
            )
            for idx, chunk in enumerate(chunks):
                texts.append(chunk.text)
                metadata_batch.append(
                    {
                        'repo': doc['repo'],
                        'path': doc['path'],
                        'language': language,
                        'symbol': chunk.symbol,
                        'chunk_index': idx,
                        'start_line': chunk.start_line,
                        'end_line': chunk.end_line,
                        'sha': doc.get('sha'),
                        'ref': doc.get('ref'),
                        'source_type': doc.get('source_type', 'local'),
                        'github_url': doc.get('github_url'),
                        'content': chunk.text,
                    }
                )

        if not texts:
            return []

        vectors = self.embedder.encode(texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False)
        for payload, vector in zip(metadata_batch, vectors):
            point_id = sha1_as_int(f"{payload['repo']}::{payload['path']}::{payload['chunk_index']}::{payload['start_line']}::{payload['end_line']}")
            points.append(qm.PointStruct(id=point_id, vector=vector.tolist(), payload=payload))
        return points

    def _upsert_points(self, points: List[qm.PointStruct]) -> int:
        if not points:
            return 0
        self.qdrant.upsert(collection_name=self.settings.collection_name, points=points)
        return len(points)

    def reindex_local_repo(self, repo_name: str, clear_existing_repo_points: bool = True) -> Dict[str, Any]:
        if clear_existing_repo_points:
            self.delete_repo_points(repo_name)
        points = self._build_points_for_documents(self._iter_local_repo_documents(repo_name))
        count = self._upsert_points(points)
        return {'repo': repo_name, 'source': 'local', 'points_indexed': count}

    def reindex_github_repo(self, github_url: str, ref: Optional[str] = None, clear_existing_repo_points: bool = True) -> Dict[str, Any]:
        repo_payload = self.github_loader.load_repository(github_url, ref=ref)
        repo_name = repo_payload['repo']
        if clear_existing_repo_points:
            self.delete_repo_points(repo_name)
        points = self._build_points_for_documents(repo_payload['documents'])
        count = self._upsert_points(points)
        return {
            'repo': repo_name,
            'source': 'github',
            'points_indexed': count,
            'ref': repo_payload['ref'],
            'github_url': github_url,
        }

    def reindex_all(self, clear_collection: bool = True, local_repos: Optional[List[str]] = None, github_repos: Optional[List[str]] = None) -> Dict[str, Any]:
        if clear_collection:
            self.recreate_collection()

        results: List[Dict[str, Any]] = []
        repo_root = Path(self.settings.repos_root)

        selected_local = local_repos
        if selected_local is None and repo_root.exists():
            selected_local = [item.name for item in repo_root.iterdir() if item.is_dir()]
        selected_local = selected_local or []

        for repo_name in selected_local:
            results.append(self.reindex_local_repo(repo_name, clear_existing_repo_points=not clear_collection))

        for github_url in github_repos or []:
            results.append(self.reindex_github_repo(github_url, clear_existing_repo_points=not clear_collection))

        return {
            'collection_name': self.settings.collection_name,
            'clear_collection': clear_collection,
            'results': results,
            'total_repos': len(results),
            'total_points_indexed': sum(item['points_indexed'] for item in results),
        }
