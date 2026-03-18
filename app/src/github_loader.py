from typing import Any, Dict, List, Optional

import httpx

from .settings import Settings
from .utils import decode_github_blob_content, parse_github_repo_url, should_skip_path


class GitHubRepoLoader:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.headers = {
            'Accept': 'application/vnd.github+json',
            'X-GitHub-Api-Version': '2022-11-28',
        }
        if settings.github_token:
            self.headers['Authorization'] = f'Bearer {settings.github_token}'

    def _client(self) -> httpx.Client:
        return httpx.Client(base_url=self.settings.github_api_url, headers=self.headers, timeout=30.0)

    def _get_repo(self, client: httpx.Client, owner: str, repo: str) -> Dict[str, Any]:
        response = client.get(f'/repos/{owner}/{repo}')
        response.raise_for_status()
        return response.json()

    def _get_tree(self, client: httpx.Client, owner: str, repo: str, ref: str) -> Dict[str, Any]:
        response = client.get(f'/repos/{owner}/{repo}/git/trees/{ref}', params={'recursive': '1'})
        response.raise_for_status()
        return response.json()

    def _get_blob(self, client: httpx.Client, owner: str, repo: str, sha: str) -> Dict[str, Any]:
        response = client.get(f'/repos/{owner}/{repo}/git/blobs/{sha}')
        response.raise_for_status()
        return response.json()

    def load_repository(self, repo_url: str, ref: Optional[str] = None) -> Dict[str, Any]:
        owner, repo = parse_github_repo_url(repo_url)
        repo_key = f'{owner}/{repo}'

        with self._client() as client:
            repo_data = self._get_repo(client, owner, repo)
            selected_ref = ref or self.settings.default_github_branch or repo_data.get('default_branch') or 'main'
            tree_data = self._get_tree(client, owner, repo, selected_ref)

            documents: List[Dict[str, Any]] = []
            for node in tree_data.get('tree', []):
                if node.get('type') != 'blob':
                    continue
                path = node.get('path', '')
                if should_skip_path(path, self.settings.include_extensions, self.settings.exclude_dirs):
                    continue
                blob = self._get_blob(client, owner, repo, node['sha'])
                if blob.get('encoding') != 'base64':
                    continue
                content = decode_github_blob_content(blob.get('content', ''))
                if not content.strip():
                    continue
                documents.append(
                    {
                        'repo': repo_key,
                        'path': path,
                        'content': content,
                        'ref': selected_ref,
                        'sha': node['sha'],
                        'source_type': 'github',
                        'github_url': repo_url,
                    }
                )

        return {
            'repo': repo_key,
            'ref': selected_ref,
            'documents': documents,
            'source_type': 'github',
            'github_url': repo_url,
        }
