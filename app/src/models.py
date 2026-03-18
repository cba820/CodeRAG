from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class ReindexRequest(BaseModel):
    clear_collection: bool = True
    local_repos: Optional[List[str]] = None
    github_repos: Optional[List[str]] = None


class ReindexRepoRequest(BaseModel):
    source: Literal['local', 'github'] = 'local'
    github_url: Optional[str] = None
    ref: Optional[str] = None
    clear_existing_repo_points: bool = True


class ChatMessage(BaseModel):
    role: Literal['system', 'user', 'assistant']
    content: str


class ChatCompletionRequest(BaseModel):
    model: Optional[str] = None
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.1
    max_tokens: Optional[int] = 1200
    stream: Optional[bool] = False
    metadata: Optional[Dict[str, Any]] = None


class ChoiceMessage(BaseModel):
    role: str = 'assistant'
    content: str


class ChatCompletionChoice(BaseModel):
    index: int
    message: ChoiceMessage
    finish_reason: str = 'stop'


class ChatCompletionUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = 'chat.completion'
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: ChatCompletionUsage


class RetrievedChunk(BaseModel):
    text: str
    score: float
    payload: Dict[str, Any] = Field(default_factory=dict)
