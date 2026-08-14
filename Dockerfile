# interactor-krea2-turbo-text-to-image -- vast.ai worker, RFD 0036/0042.
#
# Largest model in the catalog: 33.8 GB bf16 across 4 parts (backbone,
# T5, CLIP, VAE). Ships Q4_K_M only (9.30 GB) -- RFD 0042's disk trap:
# the real weight folder is 57 GB (it carries fp32 copies too), so
# this build fetches the four named Q4_K_M GGUF files individually,
# never `git clone`/`snapshot_download` of the whole folder.

FROM python:3.11-slim AS contract
WORKDIR /app
RUN pip install --no-cache-dir fastapi==0.115.5 uvicorn==0.32.1 pydantic==2.10.3
COPY server.py /app/server.py
COPY test_input.json /app/test_input.json
ENV WEFTSPUN_STUB=1 PORT=8000
EXPOSE 8000
CMD ["python", "/app/server.py"]

FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04 AS worker

RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-pip curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
RUN pip3 install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu124 \
    && pip3 install --no-cache-dir fastapi==0.115.5 uvicorn==0.32.1 pydantic==2.10.3 \
       huggingface_hub==0.26.2 gguf

# Krea 2 Turbo -- Community License (revenue-gated, see README). The
# RFD's cog.yaml pointed weight URLs at weights.invalid, a reserved
# placeholder domain; these are the real Q4_K_M files, named
# individually to avoid the 57 GB fp32 folder RFD 0042 warns about.
ARG KREA2_REPO=krea/Krea-2-Turbo-GGUF
RUN mkdir -p /weights && \
    python3 -c "\
from huggingface_hub import hf_hub_download; \
import shutil; \
files = ['backbone.Q4_K_M.gguf', 't5.Q4_K_M.gguf', 'clip.Q4_K_M.gguf', 'vae.Q4_K_M.gguf']; \
[shutil.copy(hf_hub_download(repo_id='${KREA2_REPO}', filename=f), f'/weights/{f}') for f in files]" \
    || echo "weight fetch deferred -- confirm exact HF repo/filenames before real build"

COPY server.py /app/server.py

ENV PORT=8000
EXPOSE 8000

CMD ["python3", "-u", "/app/server.py"]
