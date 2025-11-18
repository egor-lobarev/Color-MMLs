FROM python:3.12

RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user

WORKDIR /app

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        torch \
        torchvision \
        git+https://github.com/huggingface/transformers \
        accelerate \
        qwen-vl-utils[decord]==0.0.8 \
        fastapi \
        uvicorn[standard] \
        minio

COPY --chown=user . /app

USER user

ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

RUN python3 download_model.py

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
