FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 1. Instalar PyTorch para CPU primero (ESTO ES LO NUEVO)
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 2. Instalar el resto de dependencias
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]