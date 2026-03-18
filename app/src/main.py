import logging
import time
import uuid
from typing import List, Optional, TypedDict
import json
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import ORJSONResponse
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.graph import END, StateGraph

import httpx

from .indexer import RepoIndexer
from .models import (
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionUsage,
    ChoiceMessage,
    ReindexRepoRequest,
    ReindexRequest,
    RetrievedChunk,
)
from .retriever import CodeRetriever
from .settings import get_settings
from .utils import estimate_token_count

settings = get_settings()
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(title='Code RAG API', default_response_class=ORJSONResponse)
indexer = RepoIndexer(settings)
retriever = CodeRetriever(settings)


class GraphState(TypedDict):
    question: str
    chunks: List[RetrievedChunk]
    context: str
    answer: str
    model: str
    temperature: float
    max_tokens: int


SYSTEM_PROMPT = """Eres un asistente técnico especializado en responder preguntas sobre repositorios privados.
Usa solo el contexto recuperado. Si el contexto no alcanza, dilo explícitamente.
Cuando puedas, menciona repo, ruta y líneas relevantes.
No inventes APIs, clases ni flujos.
"""

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def log_event(event: str, **kwargs):
    payload = {"event": event, "ts": now_iso(), **kwargs}
    logger.info(json.dumps(payload, ensure_ascii=False, default=str))


def summarize_messages(messages):
    summary = []
    for i, m in enumerate(messages):
        content = m.get("content", "")
        if isinstance(content, str):
            preview = content[:200]
            length = len(content)
        else:
            preview = str(content)[:200]
            length = len(str(content))
        summary.append(
            {
                "index": i,
                "role": m.get("role"),
                "content_preview": preview,
                "content_length": length,
            }
        )
    return summary


def retrieve_node(state: GraphState) -> GraphState:
    started = time.perf_counter()
    question = state["question"]

    log_event(
        "retrieve_start",
        question_preview=question[:300],
        question_length=len(question),
        max_context_chunks=settings.max_context_chunks,
    )

    chunks = retriever.search(question)
    context = retriever.build_context(chunks, settings.max_context_chunks)

    elapsed_ms = round((time.perf_counter() - started) * 1000, 2)

    chunk_summaries = []
    for c in chunks[:5]:
        chunk_summaries.append(
            {
                "repo": getattr(c, "repo", None),
                "path": getattr(c, "path", None),
                "score": getattr(c, "score", None),
                "start_line": getattr(c, "start_line", None),
                "end_line": getattr(c, "end_line", None),
            }
        )

    log_event(
        "retrieve_end",
        elapsed_ms=elapsed_ms,
        chunks_count=len(chunks),
        context_length=len(context),
        top_chunks=chunk_summaries,
    )

    return {**state, "chunks": chunks, "context": context}



def answer_node(state: GraphState) -> GraphState:
    started = time.perf_counter()

    log_event(
        "answer_start",
        model=state["model"],
        temperature=state["temperature"],
        max_tokens=state["max_tokens"],
        context_length=len(state["context"] or ""),
        question_preview=state["question"][:300],
    )

    llm = ChatOllama(
        model=state["model"],
        base_url=settings.ollama_base_url,
        temperature=state["temperature"],
        num_predict=state["max_tokens"],
    )

    human_prompt = (
        f"Pregunta del usuario:\n{state['question']}\n\n"
        f"Contexto recuperado:\n{state['context'] or 'Sin contexto'}\n\n"
        "Responde en español. Si hay contexto útil, cita las fuentes al final en una sección 'Fuentes'. "
        "No digas que no tienes acceso al repositorio si sí hay contexto recuperado."
    )

    log_event(
        "answer_prompt_ready",
        prompt_length=len(human_prompt),
        prompt_preview=human_prompt[:2000],
    )

    response = llm.invoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=human_prompt),
    ])

    answer = response.content if isinstance(response.content, str) else str(response.content)
    elapsed_ms = round((time.perf_counter() - started) * 1000, 2)

    log_event(
        "answer_end",
        elapsed_ms=elapsed_ms,
        answer_length=len(answer),
        answer_preview=answer[:1000],
    )

    return {**state, "answer": answer}


workflow = StateGraph(GraphState)
workflow.add_node('retrieve', retrieve_node)
workflow.add_node('answer', answer_node)
workflow.set_entry_point('retrieve')
workflow.add_edge('retrieve', 'answer')
workflow.add_edge('answer', END)
chat_graph = workflow.compile()


@app.get('/health')
def health():
    return {'ok': True, 'collection_name': settings.collection_name}


@app.post('/admin/reindex')
def reindex_all(request: Optional[ReindexRequest] = None):
    started = time.perf_counter()
    payload = request or ReindexRequest()

    log_event(
        "reindex_all_start",
        clear_collection=payload.clear_collection,
        local_repos=payload.local_repos,
        github_repos=payload.github_repos,
    )

    try:
        result = indexer.reindex_all(
            clear_collection=payload.clear_collection,
            local_repos=payload.local_repos,
            github_repos=payload.github_repos,
        )
        elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
        log_event("reindex_all_end", elapsed_ms=elapsed_ms, result=result)
        return result
    except Exception as exc:
        logger.exception('Error reindexando todo')
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post('/admin/reindex/{repo_name}')
def reindex_one(repo_name: str, request: Optional[ReindexRepoRequest] = None):
    started = time.perf_counter()
    payload = request or ReindexRepoRequest()

    log_event(
        "reindex_repo_start",
        repo_name=repo_name,
        source=payload.source,
        github_url=payload.github_url,
        ref=payload.ref,
        clear_existing_repo_points=payload.clear_existing_repo_points,
    )

    try:
        if payload.source == 'github':
            github_url = payload.github_url or repo_name
            result = indexer.reindex_github_repo(
                github_url=github_url,
                ref=payload.ref,
                clear_existing_repo_points=payload.clear_existing_repo_points,
            )
        else:
            result = indexer.reindex_local_repo(
                repo_name=repo_name,
                clear_existing_repo_points=payload.clear_existing_repo_points,
            )

        elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
        log_event("reindex_repo_end", repo_name=repo_name, elapsed_ms=elapsed_ms, result=result)
        return result

    except FileNotFoundError as exc:
        log_event("reindex_repo_not_found", repo_name=repo_name, error=str(exc))
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception('Error reindexando repo %s', repo_name)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    request_started = time.perf_counter()

    try:
        body = await request.json()
    except Exception as exc:
        logger.exception("No se pudo parsear el JSON del request")
        raise HTTPException(status_code=400, detail="Invalid JSON body") from exc

    messages = body.get("messages", [])
    requested_model = body.get("model", settings.chat_model)
    temperature = body.get("temperature", 0.1)
    max_tokens = body.get("max_tokens", 512)
    stream = body.get("stream", False)

    log_event(
        "chat_request_received",
        requested_model=requested_model,
        internal_model=settings.chat_model,
        stream=stream,
        messages_count=len(messages),
        messages_summary=summarize_messages(messages),
    )

    if not messages:
        raise HTTPException(status_code=400, detail="No messages provided")

    user_parts = []
    for m in messages:
        if m.get("role") == "user":
            content = m.get("content", "")
            if isinstance(content, str):
                user_parts.append(content)

    user_message = "\n".join(user_parts).strip()

    if not user_message:
        raise HTTPException(status_code=400, detail="No user message content provided")

    log_event(
        "chat_user_message_built",
        question_length=len(user_message),
        question_preview=user_message[:500],
        estimated_tokens=estimate_token_count(user_message),
    )

    graph_input: GraphState = {
        "question": user_message,
        "chunks": [],
        "context": "",
        "answer": "",
        "model": settings.chat_model,  # fuerza el modelo interno real
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    try:
        graph_started = time.perf_counter()
        result = chat_graph.invoke(graph_input)
        graph_elapsed_ms = round((time.perf_counter() - graph_started) * 1000, 2)

        answer = result.get("answer", "")
        chunks = result.get("chunks", [])
        context = result.get("context", "")

        total_elapsed_ms = round((time.perf_counter() - request_started) * 1000, 2)

        log_event(
            "chat_request_completed",
            graph_elapsed_ms=graph_elapsed_ms,
            total_elapsed_ms=total_elapsed_ms,
            chunks_count=len(chunks),
            context_length=len(context),
            answer_length=len(answer),
        )

        return {
            "id": f"chatcmpl-{uuid.uuid4()}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "rag-codebase",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": answer,
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": estimate_token_count(user_message + context),
                "completion_tokens": estimate_token_count(answer),
                "total_tokens": estimate_token_count(user_message + context + answer),
            },
        }

    except httpx.ReadTimeout:
        logger.exception("Timeout llamando a Ollama")
        raise HTTPException(
            status_code=504,
            detail="Ollama tardó demasiado en responder. Sube el timeout o usa un modelo más liviano.",
        )
    except Exception as exc:
        logger.exception("Error procesando /v1/chat/completions")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": settings.chat_model,
                "object": "model",
                "owned_by": "local"
            }
        ]
    }