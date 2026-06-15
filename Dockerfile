# GGUF serverless worker for SIQ-1-35B (qwen3_5_moe — only llama.cpp supports this arch)
FROM ghcr.io/ggml-org/llama.cpp:server-cuda
RUN apt-get update && apt-get install -y python3 python3-pip curl && rm -rf /var/lib/apt/lists/*
RUN pip3 install --no-cache-dir --break-system-packages runpod huggingface_hub requests
COPY handler.py /handler.py
ENV GGUF_REPO=AlexWortega/SIQ-1-35B \
    GGUF_FILE=gguf/SIQ-1-35B.Q4_K_M.gguf \
    N_CTX=16384 NGL=99
ENTRYPOINT ["python3", "/handler.py"]
