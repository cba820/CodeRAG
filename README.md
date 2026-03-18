# Code RAG - Fase 1

## Estructura

- `docker-compose.yml`
- `.env.example`
- `repos/` → clona aquí repos locales si vas a indexar desde disco
- `app/` → API FastAPI

## Archivos de la API

- `app/Dockerfile`
- `app/requirements.txt`
- `app/src/main.py`
- `app/src/settings.py`
- `app/src/models.py`
- `app/src/utils.py`
- `app/src/github_loader.py`
- `app/src/indexer.py`
- `app/src/retriever.py`

## Arranque

1. Copia `.env.example` a `.env`
2. Crea la carpeta `repos/`
3. Levanta los servicios:

```bash
docker compose up -d --build
```

4. Descarga el modelo de chat en Ollama:

```bash
docker exec -it ollama ollama pull qwen3:8b
```

## Endpoints

### Health

```bash
curl http://localhost:8000/health
```

### Reindexar todo

Solo repos locales detectados en `./repos`:

```bash
curl -X POST http://localhost:8000/admin/reindex \
  -H 'Content-Type: application/json' \
  -d '{"clear_collection": true}'
```

Repos locales específicos y repos GitHub específicos:

```bash
curl -X POST http://localhost:8000/admin/reindex \
  -H 'Content-Type: application/json' \
  -d '{
    "clear_collection": true,
    "local_repos": ["repo-a", "repo-b"],
    "github_repos": [
      "https://github.com/tu-org/proyecto-a",
      "https://github.com/tu-org/proyecto-b"
    ]
  }'
```

### Reindexar un repo local

```bash
curl -X POST http://localhost:8000/admin/reindex/mi-repo-local \
  -H 'Content-Type: application/json' \
  -d '{"source": "local", "clear_existing_repo_points": true}'
```

### Reindexar un repo GitHub con token

```bash
curl -X POST http://localhost:8000/admin/reindex/tu-org/proyecto-a \
  -H 'Content-Type: application/json' \
  -d '{
    "source": "github",
    "github_url": "https://github.com/tu-org/proyecto-a",
    "ref": "main",
    "clear_existing_repo_points": true
  }'
```

### Chat OpenAI-compatible

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3:8b",
    "messages": [
      {"role": "user", "content": "¿Dónde se calcula la comisión en el proyecto payments-service?"}
    ],
    "temperature": 0.1,
    "max_tokens": 800
  }'
```

## Notas

- Para GitHub privado, define `GITHUB_TOKEN` en `.env`.
- Para Open WebUI, entra a `http://localhost:3000`.
- La API usa `SentenceTransformer` para embeddings y Qdrant para almacenamiento vectorial.
