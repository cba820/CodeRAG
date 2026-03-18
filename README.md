# CodeRAG

RAG (Retrieval-Augmented Generation) system for querying private code repositories using local LLMs, Qdrant and LangChain/LangGraph.

---

## 🚀 Overview

CodeRAG allows you to ask natural language questions about private repositories (code, business logic, architecture) and get contextual answers based on indexed source code.

It is fully self-hosted and runs locally using Docker.

---

## 🧠 Architecture

```
Open WebUI → FastAPI (RAG API) → Qdrant (vector DB)
                                → Ollama (LLM)
```

### Components

* **Open WebUI**: Chat interface
* **FastAPI (rag-api)**: Core RAG logic
* **LangGraph**: Workflow orchestration (retrieve → answer)
* **Qdrant**: Vector database for embeddings
* **Ollama**: Local LLM inference
* **Indexer**: Processes repos (local or GitHub)

---

## ⚙️ Features

### ✅ Implemented (Phase 1)

* RAG over local repositories
* RAG over GitHub repositories (via API + token)
* Vector search using Qdrant
* Local LLM inference via Ollama
* OpenAI-compatible API (`/v1/chat/completions`)
* Open WebUI integration
* Basic chunking + metadata
* Logging and observability

---

## 📂 Project Structure

```
rag-codebase/
├── docker-compose.yml
├── .env
├── repos/
├── app/
│   ├── Dockerfile
│   ├── requirements.txt
│   └── src/
│       ├── main.py
│       ├── indexer.py
│       ├── retriever.py
│       ├── settings.py
│       └── utils.py
```

---

## 🛠️ Setup

### 1. Clone repos

```bash
git clone <your-private-repo> repos/PortalEmpresas
```

### 2. Configure `.env`

```env
CHAT_MODEL=qwen2.5:3b
QDRANT_URL=http://qdrant:6333
OLLAMA_BASE_URL=http://ollama:11434
```

### 3. Start services

```bash
docker compose up -d --build
```

### 4. Pull model

```bash
docker exec -it ollama ollama pull qwen2.5:3b
```

---

## 📥 Indexing

### Local repo

```bash
curl -X POST http://localhost:8000/admin/reindex/PortalEmpresas
```

### GitHub repo

```bash
curl -X POST http://localhost:8000/admin/reindex \
  -H "Content-Type: application/json" \
  -d '{"github_repos": ["https://github.com/org/repo"]}'
```

---

## 🔍 Querying

Via Open WebUI or:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "rag-codebase",
    "messages": [
      {"role": "user", "content": "What does PortalEmpresas do?"}
    ]
  }'
```

---

## 🧪 Debugging

### Check Qdrant

```bash
curl http://localhost:6333/collections/code_chunks
```

### Logs

```bash
docker logs -f rag-api
```

Look for:

* retrieve_start / retrieve_end
* answer_start / answer_end

---

## ⚠️ Known Limitations (Phase 1)

* No reranking
* No hybrid search
* Basic chunking
* No repo filtering
* No streaming
* No incremental indexing

---

## 🧭 Roadmap

### Phase 2

* Better code chunking (Tree-sitter)
* Reranker (cross-encoder)
* Hybrid search (dense + sparse)
* Repo-aware filtering

### Phase 3

* Langfuse observability
* Streaming responses
* Multi-repo reasoning
* Incremental indexing

### Phase 4

* Agents (LangGraph)
* Code execution tools
* CI/CD integration

---

## 🧠 Learnings

This project demonstrates:

* RAG architecture
* Vector databases
* Local LLM serving
* Prompt engineering
* AI system design

---

## 📌 Tech Stack

* Python
* FastAPI
* LangChain / LangGraph
* Qdrant
* Ollama
* Docker

---

## 📈 Future Improvements

* Add embeddings optimization
* Add caching layer
* Improve latency
* Add authentication

---

## 👨‍💻 Author

Built as a hands-on project to learn production-grade RAG systems.
