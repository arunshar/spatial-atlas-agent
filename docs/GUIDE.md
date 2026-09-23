# Spatial Atlas Guide

This guide shows how to install Spatial Atlas, run its tests, start the Agent-to-Agent (A2A) server on your own machine, and check that it answers. Every step in Sections 1 to 7 runs locally. [ARCHITECTURE.md](ARCHITECTURE.md) describes what happens inside the server, and [EXPLAINED.md](EXPLAINED.md) explains the ideas and the evidence boundary.

## Before you start

You need the tools in this table.

| Tool | Version | Notes |
| --- | --- | --- |
| Python | 3.12 or 3.13 | The project requires at least 3.12 and below 3.14. The `.python-version` file selects 3.13. |
| `uv` | A recent release | It installs the locked dependencies. The [uv documentation](https://docs.astral.sh/uv/) explains how to install it. |
| `git` | Any | It fetches the repository. |
| Docker | Any recent release | You need it only for the container steps in Section 8. |

The agent needs an API key for a model provider before it can answer a task. The default model tiers use OpenAI models, so `OPENAI_API_KEY` is enough. The server can start and serve its agent card without a key.

## 1. Get the code

Clone the repository and run every later command from its root directory.

```bash
git clone https://github.com/arunshar/spatial-atlas-agent.git
cd spatial-atlas-agent
```

## 2. Install the dependencies

```bash
uv sync --frozen --extra test
```

This command builds a local virtual environment from `uv.lock` and adds the test dependencies. The `--frozen` flag installs exactly what `uv.lock` pins. You can leave out `--extra test` when you only want to run the server.

## 3. Add a model key

```bash
cp sample.env .env
```

Open `.env` and put your key after `OPENAI_API_KEY=`. The server reads `.env` when it starts. Keep `.env` out of version control, because it holds secrets.

You can point any tier at another provider that LiteLLM supports. Each value needs a provider prefix, as in `openai/gpt-4.1`.

| Variable | Default | What the tier does |
| --- | --- | --- |
| `ATLAS_FAST_MODEL` | `openai/gpt-4.1-mini` | It grades the FieldWork answer and extracts object phrases for the metric bridge. |
| `ATLAS_STANDARD_MODEL` | `openai/gpt-4.1` | It analyzes MLE-Bench (machine learning engineering benchmark) competitions. |
| `ATLAS_STRONG_MODEL` | `openai/gpt-4.1` | It extracts the scene, writes the answer and the refinement, and writes MLE pipeline code. |
| `ATLAS_VISION_MODEL` | `openai/gpt-4.1` | It describes images and video frames. |

## 4. Run the tests

```bash
uv run pytest
```

The default options skip tests marked `e2e` or `gpu`. Those tests need a SpatialClaw GPU tool server and benchmark data, and this repository includes neither. A passing run shows that the covered code behaves as the tests expect. It says nothing about accuracy on any benchmark.

## 5. Start the server on your own machine

```bash
uv run src/server.py --host 127.0.0.1 --port 9019
```

Startup prints the resolved model tiers and the two skills, and then Uvicorn starts listening. The command keeps running until you stop it. On a loopback address such as `127.0.0.1`, only your own machine can reach the server, so it may run without a bearer token.

When startup stops with a message about a model tier, check that each `ATLAS_*_MODEL` value is set and carries a provider prefix.

## 6. Check that the server answers

Open a second terminal and fetch the agent card.

```bash
curl -s http://127.0.0.1:9019/.well-known/agent-card.json | python3 -m json.tool
```

The reply is a JSON agent card named `Spatial Atlas` with two skills, `fieldwork-research` and `ml-engineering`. Opening `http://127.0.0.1:9019/` in a browser shows a short landing page. The agent card describes the interface, and it does not show that any task has run.

Run the smoke client against the local server to send one real task. This step calls your model provider, so it needs the key from Section 3.

```bash
uv run python eval_smoke.py
```

The client posts a synthetic FieldWork question and exits with code 0 when the task completes with a usable text answer. It exits with code 1 when the task ends in any other state. Add `--image path/to/image.jpg` to attach an image as well. The default target is `http://127.0.0.1:9019/`, and `--url` points the client at another address. The smoke client sends no bearer token, so use it only against a loopback server that has no token set.

## 7. Stop the server

Press Ctrl+C in the terminal that runs the server. The task store lives in memory, so stopping the server discards every task.

## 8. Run on a non-loopback address

A server that listens on any other address, such as `0.0.0.0`, must have a bearer token of at least 32 characters. Startup stops without one.

Generate a token and add it to `.env`.

```bash
python3 -c "import secrets; print(secrets.token_urlsafe(32))"
```

```text
ATLAS_BEARER_TOKEN=<TOKEN>
```

Each request that is not GET, HEAD, or OPTIONS must then send the header `Authorization: Bearer <TOKEN>`. You can confirm that the check works by sending a POST without the header. The command should print 401.

```bash
curl -s -o /dev/null -w "%{http_code}\n" -X POST http://127.0.0.1:9019/
```

Three more settings control admission and startup.

| Variable | Default | Effect |
| --- | --- | --- |
| `ATLAS_MAX_REQUEST_BYTES` | 64 MiB | The server answers 413 for a larger body. |
| `ATLAS_MAX_CONCURRENT_REQUESTS` | 4 | The server answers 503 when this many non-read-only requests are already active. |
| `ATLAS_ALLOW_UNAUTHENTICATED_PUBLIC` | Disabled | This test-only override lets a non-loopback server start without a token. Do not use it for a deployment. |

### Run in Docker

The Dockerfile starts the server on `0.0.0.0`, so the container needs `ATLAS_BEARER_TOKEN` in `.env`.

```bash
docker build -t spatial-atlas-agent .
docker run --rm -p 9019:9019 --env-file .env -e PUBLIC_URL=http://127.0.0.1:9019/ spatial-atlas-agent
```

Fetch the agent card with the `curl` command from Section 6. The image copies `src/`, the project metadata, the LICENSE and NOTICE files, the license texts in `LICENSES/`, and the locked runtime dependencies. It does not include the tests or the evaluation entry.

### Hosted deployment

The repository also includes `deploy_to_hf.sh`, which copies the committed files at `HEAD` into a Hugging Face Space. Commit your changes before you run it, because it skips uncommitted edits and untracked files. The script reads the Space address from `HF_SPACE_REPO`, and it stops when that variable is unset or holds credentials. Your Space needs `OPENAI_API_KEY` as a secret. It also needs `ATLAS_BEARER_TOKEN` as a secret, because the Space binds to a non-loopback address.

## 9. Keep MLE execution off unless you have an isolated worker

The MLE handler refuses to execute generated code by default. A task routed to it fails with a sanitized message and a reference code, and that failure is the expected behavior.

Execution needs both `ATLAS_ENABLE_MLEBENCH_CODE_EXECUTION=true` and `ATLAS_TRUSTED_ISOLATED_WORKER=true`, and startup then also requires the bearer token. Set these flags only inside an isolated, disposable worker that you control. The executor limits time, output size, and environment variables, but it is not a complete security sandbox.

`ATLAS_ALLOW_DUMMY_SUBMISSION=true` is a separate switch. It allows a placeholder submission after every real attempt fails, so leave it unset unless you need that behavior.

## 10. Optional: the metric perception engine

Setting `ATLAS_FIELDWORK_ENGINE=metric` asks the FieldWork handler to build its scene from SAM3 masks and a Depth-Anything-3 point map. This engine needs two external pieces that this repository does not include.

1. The `spatial_agent` package from NVIDIA's SpatialClaw project must be importable in the same environment. NVIDIA licenses SpatialClaw under the NVIDIA Source Code License-NC, which restricts it to non-commercial use. The [NOTICE](../NOTICE) file gives the details.
2. A SpatialClaw GPU tool server must be running. NVIDIA's public SpatialClaw finds the server through the registry file `logs/gpu_server.json`. SpatialClaw polls for about four hours when that file lists no live server. My local SpatialClaw checkout also reads a `SPATIALCLAW_GPU_SERVER_URL` override that pins one server address, so a call to an unreachable address fails within seconds. NVIDIA's public release does not read this variable.

Without the package, the server logs `metric backend unavailable` and falls back to the default scene graph for that request. The first metric request can wait about four hours before it fails and falls back when NVIDIA's public package is installed and no GPU server is live. This fallback is intended for the A2A path. The evaluation entry turns it off.

## 11. Optional: the evaluation entry

`eval_bench.py` loads a benchmark through the benchmark factory in a SpatialClaw checkout. It calls the FieldWork handler directly on each row and writes journals to an output directory. `eval_bench.py` needs a SpatialClaw checkout (`--spatialclaw-root` or `SPATIALCLAW_ROOT`), the benchmark data (`--data-root`), and a model key. The metric arms also need the GPU tool server. The strict QSpatial arms need a `qspatial_gap` benchmark loader that I wrote inside a local SpatialClaw checkout. NVIDIA's public SpatialClaw does not include this loader. The loader extends SpatialClaw's benchmark classes, so it falls under NVIDIA's Source Code License-NC. This repository does not include the loader for that reason. The journals and `run_manifest.json` record absolute paths from your machine, so replace those paths before you share any output file.

The strict QSpatial checks need the digests of private control artifacts, which the operator supplies through environment variables. `ATLAS_QSPATIAL_GAP_SLICE_SHA256` covers the frozen gap slice. The shuffled-image arm also needs the control mapping file and the two variables `ATLAS_QSPATIAL_CONTROL_MAPPING_SHA256` and `ATLAS_QSPATIAL_IMAGE_MANIFEST_SHA256`. When a digest is unset, the matching check fails closed. None of these artifacts ships with this repository.

A minimal scene-graph invocation looks like this.

```bash
uv run python eval_bench.py \
  --benchmark <benchmark-key> \
  --data-root <path-to-data> \
  --engine scenegraph \
  --subsample 5 \
  --output-dir work_dir/scenegraph \
  --spatialclaw-root <path-to-SpatialClaw>
```

[ARCHITECTURE.md](ARCHITECTURE.md#51-the-four-frozen-run-modes) lists the flags for the four run modes. Running them does not reproduce the V37 run. That run used a combined orchestration in a separate private execution environment, and this repository does not include that orchestration.

## Troubleshooting

| What you see | What it means and what to do |
| --- | --- |
| Startup stops with "A non-loopback server requires ATLAS_BEARER_TOKEN" | The server binds to a non-loopback address without a token. Set a token of at least 32 characters, or bind to `127.0.0.1`. |
| Startup stops with "ATLAS_BEARER_TOKEN must contain at least 32 characters" | The token is too short. Generate a new one with the command in Section 8. |
| Startup stops with "Enabled MLE code execution requires isolated-worker attestation and ATLAS_BEARER_TOKEN" | MLE execution is on without its other requirements. Unset the flag unless you run inside an isolated worker. |
| Startup stops with a message about a model tier | A tier value is empty or has no provider prefix. Use a value such as `openai/gpt-4.1`. |
| A POST returns 401 | The request has no bearer header, or the header holds the wrong token. |
| A POST returns 413 | The body is larger than `ATLAS_MAX_REQUEST_BYTES`. |
| A POST returns 503 | Every request slot is busy. Retry later, or raise `ATLAS_MAX_CONCURRENT_REQUESTS`. |
| An ML task fails with a reference code | The MLE handler refuses execution by default, so this failure is expected. |
| `eval_smoke.py` prints a reference code | The task failed on the server. Search the server's terminal output for that code to see the exception type. A missing or invalid model key is one common cause. |
| The log says `metric backend unavailable` | The SpatialClaw package or GPU service is missing, so the server used the scene graph. |
| `Address already in use` | Another process holds port 9019. Pass `--port 9020` to the server. |
| Tests fail with import errors | Run the tests from the repository root with `uv run pytest`. |
