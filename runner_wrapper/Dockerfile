# syntax=docker/dockerfile:1.7
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive \
    VIRTUAL_ENV=/opt/runner \
    PATH=/opt/runner/bin:${PATH} \
    TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9+PTX" \
    UV_LINK_MODE=copy

RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates git python3 python3-pip python3-venv && \
    rm -rf /var/lib/apt/lists/* && \
    python3 -m pip install --no-cache-dir uv==0.12.5 && \
    uv venv --python python3 /opt/runner

COPY runner_wrapper/requirements.txt /build/requirements.txt
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124 && \
    uv pip install -r /build/requirements.txt && \
    uv pip install --no-deps \
      "git+https://github.com/microsoft/MoGe.git@72fdee98935d93b69f7ce0a1369cfd6e2573fc2f" \
      "git+https://github.com/EasternJournalist/utils3d.git@c5daf6f6c244d251f252102d09e9b7bcef791a38" \
      "git+https://github.com/naver-ai/ZIM.git@392e613c8a3d395ce4dda3b51edb6dabc1fd0004" \
      "git+https://github.com/xinntao/Real-ESRGAN.git@a4abfb2979a7bbff3f69f58f58ae324608821e27"

FROM nvidia/cuda:12.4.1-base-ubuntu22.04 AS runtime
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates python3 libegl1 libgl1 libglib2.0-0 libgomp1 && \
    rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/runner /opt/runner
WORKDIR /app/model_repo
COPY hy3dworld /app/model_repo/hy3dworld
COPY demo_scenegen.py LICENSE NOTICE /app/model_repo/
COPY runner_wrapper /app/model_repo/runner_wrapper

ENV VIRTUAL_ENV=/opt/runner \
    PATH=/opt/runner/bin:${PATH} \
    PYTHONPATH=/app/model_repo \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9+PTX" \
    RUNNER_NAME=hunyuanworld-panorama \
    RUNNER_TYPE=generator \
    RUNNER_VERSION=0.1.0 \
    RUNNER_PORT=58090 \
    RUNNER_CONTRACT_VERSION=1 \
    RUNNER_IDLE_TIMEOUT_SECONDS=900 \
    RUNNER_ADAPTER=runner_wrapper.adapter:run_job \
    HF_HOME=/data/model_cache/hunyuanworld-1.0/huggingface \
    TORCH_HOME=/data/model_cache/hunyuanworld-1.0/torch \
    XDG_CACHE_HOME=/data/model_cache/hunyuanworld-1.0/xdg \
    HF_HUB_DISABLE_XET=1 \
    OMP_NUM_THREADS=8 \
    OPENBLAS_NUM_THREADS=8

RUN --network=none python -B -c "from demo_scenegen import HYworldDemo; from runner_wrapper.adapter import run_job; print(HYworldDemo.__name__, run_job.__name__)"
EXPOSE 58090
CMD ["python", "-m", "runner_wrapper.server"]
