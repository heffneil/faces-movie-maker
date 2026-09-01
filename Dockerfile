FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg espeak-ng \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU-only torch keeps the image a fraction of the CUDA build's size
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY pumpkin.py sprite.py ghost.py talk.py cartoon.py ./
COPY webapp/app.py webapp/index.html webapp/

# Kokoro model cache persists via the /cache volume
ENV HF_HOME=/cache/huggingface
ENV HOST=0.0.0.0

EXPOSE 5173
CMD ["python", "webapp/app.py"]
