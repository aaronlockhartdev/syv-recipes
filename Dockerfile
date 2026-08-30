# Pinned vLLM 0.27.1 with every patch in patches/ applied.
#
# vLLM 0.27.1 pins torch 2.13.0 (cu130), triton 3.7.1 and
# flashinfer-python 0.6.16.post3 itself, so requirements.txt only pins what
# the reference install resolved for the rest.
#
# Base is CUDA "base" + nvcc, not "devel": vLLM's wheels bring their own
# CUDA libraries, but Triton's launchers and FlashInfer's JIT need a
# C/CUDA compiler and dev headers (cudart, curand) at runtime. The
# torch.compile and JIT caches live in the /cache volume, so first-start
# compilation happens once.
FROM nvidia/cuda:13.0.1-base-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
      python3.12 python3.12-venv python3.12-dev \
      cuda-nvcc-13-0 cuda-cudart-dev-13-0 libcurand-dev-13-0 \
      build-essential patch curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# uv owns all package management
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

WORKDIR /app
COPY requirements.txt .
RUN uv venv .venv --python /usr/bin/python3.12 \
    && uv pip install --python .venv/bin/python -r requirements.txt

# set -e + --batch: a hunk that fails (or was already applied) aborts the
# build instead of leaving a half-patched vLLM; compileall catches a patch
# that applies but leaves broken Python.
COPY patches/ patches/
RUN set -e; \
    SP=$(.venv/bin/python -c 'import vllm, os; print(os.path.dirname(vllm.__file__))'); \
    for p in patches/*.patch; do echo "== $p"; patch -p1 -d "$SP" --batch < "$p"; done; \
    .venv/bin/python -m compileall -q "$SP"

COPY docker/ docker/
COPY prepare/ prepare/
COPY recipes/ recipes/

# HOME is a volume: the HF hub cache (model prep -- bind-mount the host's
# own there, README: Docker), torch.compile, and the Triton/FlashInfer JIT
# caches. HF_HOME is pinned so the location cannot drift if HOME is
# overridden at run time.
RUN mkdir -p /cache /app/models && chmod 1777 /cache
ENV HOME=/cache HF_HOME=/cache/.cache/huggingface VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 HF_HUB_ENABLE_HF_TRANSFER=1
VOLUME ["/cache", "/app/models"]
EXPOSE 8080
ENTRYPOINT ["bash", "docker/entrypoint.sh"]
