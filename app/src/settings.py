from functools import lru_cache
from typing import List

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8', extra='ignore')

    log_level: str = 'INFO'
    qdrant_url: str = 'http://localhost:6333'
    ollama_base_url: str = 'http://localhost:11434'
    repos_root: str = '/workspace/repos'
    collection_name: str = 'code_chunks'
    embedding_model: str = 'BAAI/bge-small-en-v1.5'
    chat_model: str = 'qwen2.5:3b'
    top_k: int = 8
    max_context_chunks: int = 6
    chunk_size: int = 1400
    chunk_overlap: int = 200
    include_extensions: List[str] = ['.py', '.ts', '.tsx', '.js', '.jsx', '.cs', '.go', '.java', '.kt', '.md', '.yaml', '.yml', '.json', '.sql', '.cshtml', '.html']
    exclude_dirs: List[str] = ['.git', 'node_modules', 'dist', 'build', '.next', '.venv', 'venv', 'bin', 'obj', 'target', '__pycache__']
    github_token: str = ''
    github_api_url: str = 'https://api.github.com'
    default_github_branch: str = 'master'

    @field_validator('include_extensions', 'exclude_dirs', mode='before')
    @classmethod
    def split_csv_values(cls, value):
        if isinstance(value, str):
            return [item.strip() for item in value.split(',') if item.strip()]
        return value


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
