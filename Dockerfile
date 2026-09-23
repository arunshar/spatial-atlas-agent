# This layout started from the RDI Foundation's AgentBeats agent template. NOTICE gives the details.
FROM ghcr.io/astral-sh/uv:python3.13-bookworm@sha256:47965cdc9d53a515f68f78241161c901e70051ce428f12e791bd7fe19f6a631a

ENV UV_HTTP_TIMEOUT=300

# Install system dependencies for opencv-python-headless
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN adduser --disabled-password --gecos "" agent
USER agent
RUN mkdir -p /home/agent/.cache/uv
WORKDIR /home/agent/atlas

COPY LICENSE NOTICE pyproject.toml uv.lock README.md ./
COPY LICENSES LICENSES
COPY src src

RUN \
    --mount=type=cache,target=/home/agent/.cache/uv,uid=1000 \
    uv sync --locked --no-dev --no-install-project

# --no-sync keeps the start offline. The build step already installed the locked runtime
# dependencies, and src/server.py runs as a script, so the project itself needs no install.
ENTRYPOINT ["uv", "run", "--no-sync", "src/server.py"]
CMD ["--host", "0.0.0.0"]
EXPOSE 9019
