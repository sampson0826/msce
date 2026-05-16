FROM nvidia/cuda:12.1-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y python3 python3-pip && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip3 install --no-cache-dir \
    torch==2.3.1 \
    transformers>=4.40 \
    fastapi>=0.110 \
    uvicorn>=0.27 \
    gradio>=4.0 \
    numpy>=1.24 \
    scikit-learn>=1.3 \
    accelerate>=0.28 \
    pydantic>=2.0

ENV HALLU_MODEL=Qwen/Qwen2.5-7B-Instruct
ENV HALLU_DEVICE=cuda
ENV HF_ENDPOINT=https://hf-mirror.com

COPY . /app/constraint_residual/

EXPOSE 8080 7860

CMD python3 -m constraint_residual.hallucination_predictor.api_server --port 8080
