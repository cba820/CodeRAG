import base64
import hashlib
import re
from pathlib import Path
from typing import Iterable, List, Optional, Tuple
from urllib.parse import urlparse


EXTENSION_LANGUAGE_MAP = {
    '.py': 'python',
    '.ts': 'typescript',
    '.tsx': 'typescript',
    '.js': 'javascript',
    '.jsx': 'javascript',
    '.cs': 'csharp',
    '.go': 'go',
    '.java': 'java',
    '.kt': 'kotlin',
    '.md': 'markdown',
    '.yaml': 'yaml',
    '.yml': 'yaml',
    '.json': 'json',
    '.sql': 'sql',
}


def sha1_as_int(value: str) -> int:
    digest = hashlib.sha1(value.encode('utf-8')).hexdigest()
    return int(digest[:16], 16)


def detect_language(path: str) -> str:
    return EXTENSION_LANGUAGE_MAP.get(Path(path).suffix.lower(), 'text')


def should_skip_path(path: str, include_extensions: List[str], exclude_dirs: List[str]) -> bool:
    normalized = path.replace('\\', '/')
    parts = normalized.split('/')
    if any(part in exclude_dirs for part in parts):
        return True
    if include_extensions:
        ext = Path(normalized).suffix.lower()
        return ext not in include_extensions
    return False


def extract_symbol_hint(text: str, language: str) -> Optional[str]:
    patterns = {
        'python': [r'^class\s+([A-Za-z_][A-Za-z0-9_]*)', r'^def\s+([A-Za-z_][A-Za-z0-9_]*)'],
        'javascript': [r'^class\s+([A-Za-z_][A-Za-z0-9_]*)', r'^function\s+([A-Za-z_][A-Za-z0-9_]*)', r'^const\s+([A-Za-z_][A-Za-z0-9_]*)\s*=\s*\('],
        'typescript': [r'^class\s+([A-Za-z_][A-Za-z0-9_]*)', r'^function\s+([A-Za-z_][A-Za-z0-9_]*)', r'^export\s+class\s+([A-Za-z_][A-Za-z0-9_]*)'],
        'csharp': [r'^(?:public|private|internal|protected)?\s*class\s+([A-Za-z_][A-Za-z0-9_]*)', r'^(?:public|private|internal|protected)?\s*(?:async\s+)?(?:static\s+)?[A-Za-z_<>,\[\]]+\s+([A-Za-z_][A-Za-z0-9_]*)\s*\('],
        'go': [r'^type\s+([A-Za-z_][A-Za-z0-9_]*)\s+struct', r'^func\s+([A-Za-z_][A-Za-z0-9_]*)\s*\('],
        'java': [r'^class\s+([A-Za-z_][A-Za-z0-9_]*)', r'^(?:public|private|protected)?\s*[A-Za-z_<>,\[\]]+\s+([A-Za-z_][A-Za-z0-9_]*)\s*\('],
        'kotlin': [r'^class\s+([A-Za-z_][A-Za-z0-9_]*)', r'^fun\s+([A-Za-z_][A-Za-z0-9_]*)\s*\('],
    }
    candidates = patterns.get(language, [])
    for line in text.splitlines():
        stripped = line.strip()
        for pattern in candidates:
            match = re.match(pattern, stripped)
            if match:
                return match.group(1)
    return None


class SimpleChunk:
    def __init__(self, text: str, start_line: int, end_line: int, symbol: Optional[str] = None):
        self.text = text
        self.start_line = start_line
        self.end_line = end_line
        self.symbol = symbol



def chunk_text(text: str, chunk_size: int, chunk_overlap: int, language: str) -> List[SimpleChunk]:
    lines = text.splitlines()
    if not lines:
        return []

    chunks: List[SimpleChunk] = []
    current_lines: List[str] = []
    current_start = 1
    current_chars = 0

    for idx, line in enumerate(lines, start=1):
        next_len = len(line) + 1
        if current_lines and current_chars + next_len > chunk_size:
            chunk_text_value = '\n'.join(current_lines).strip()
            if chunk_text_value:
                chunks.append(
                    SimpleChunk(
                        text=chunk_text_value,
                        start_line=current_start,
                        end_line=idx - 1,
                        symbol=extract_symbol_hint(chunk_text_value, language),
                    )
                )
            if chunk_overlap > 0:
                overlap_chars = 0
                overlap_lines: List[str] = []
                overlap_start = idx
                for back_idx in range(len(current_lines) - 1, -1, -1):
                    candidate = current_lines[back_idx]
                    overlap_chars += len(candidate) + 1
                    overlap_lines.insert(0, candidate)
                    overlap_start = current_start + back_idx
                    if overlap_chars >= chunk_overlap:
                        break
                current_lines = overlap_lines.copy()
                current_start = overlap_start
                current_chars = sum(len(item) + 1 for item in current_lines)
            else:
                current_lines = []
                current_start = idx
                current_chars = 0

        if not current_lines:
            current_start = idx
        current_lines.append(line)
        current_chars += next_len

    if current_lines:
        chunk_text_value = '\n'.join(current_lines).strip()
        if chunk_text_value:
            chunks.append(
                SimpleChunk(
                    text=chunk_text_value,
                    start_line=current_start,
                    end_line=len(lines),
                    symbol=extract_symbol_hint(chunk_text_value, language),
                )
            )

    return chunks



def decode_github_blob_content(base64_text: str) -> str:
    return base64.b64decode(base64_text).decode('utf-8', errors='ignore')



def parse_github_repo_url(repo_url: str) -> Tuple[str, str]:
    url = repo_url.strip()
    if url.endswith('.git'):
        url = url[:-4]

    if url.startswith('git@github.com:'):
        path = url.split(':', 1)[1]
        owner, repo = path.split('/', 1)
        return owner, repo

    parsed = urlparse(url)
    parts = [part for part in parsed.path.split('/') if part]
    if len(parts) < 2:
        raise ValueError(f'GitHub URL inválida: {repo_url}')
    return parts[0], parts[1]



def estimate_token_count(text: str) -> int:
    return max(1, len(text) // 4)
