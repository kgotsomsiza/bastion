# FrugalRouter

Track 1 agent for the AMD Developer Hackathon: ACT II.

FrugalRouter is a token-efficient general-purpose AI agent. It handles the official Track 1 task format, uses only models from `ALLOWED_MODELS`, sends all Fireworks calls through `FIREWORKS_BASE_URL`, and writes the required `/output/results.json`.

## Official Track 1 Contract

The submitted container must:

1. Read `/input/tasks.json` on startup.
2. Write `/output/results.json` before exiting.
3. Read `FIREWORKS_API_KEY`, `FIREWORKS_BASE_URL`, and `ALLOWED_MODELS` from the environment.
4. Use only allowed Fireworks models.
5. Exit with code 0 on success.

Expected input:

```json
[
  { "task_id": "t1", "prompt": "Summarise the following text in one sentence: ..." }
]
```

Expected output:

```json
[
  { "task_id": "t1", "answer": "..." }
]
```

## Strategy

```text
task -> classify category -> deterministic shortcut if very safe
     -> choose allowed Fireworks model
     -> concise prompt through FIREWORKS_BASE_URL
     -> write official answer JSON
     -> log route/model/token metadata separately
```

The agent covers the eight Track 1 categories: factual Q&A, math reasoning, sentiment, summarization, NER, code debugging, logic puzzles, and code generation.

## Quick Start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
python -m pytest
```

Local dry run without Fireworks:

```powershell
python -m frugalrouter.cli --tasks data/sample_tasks.json --output outputs/sample_debug.jsonl --no-remote --debug-output
```

Local run with Fireworks:

```powershell
$env:ALLOWED_MODELS="minimax-m3,kimi-k2p7-code,gemma-4-31b-it,gemma-4-26b-a4b-it,gemma-4-31b-it-nvfp4"
$env:FIREWORKS_BASE_URL="https://api.fireworks.ai/inference/v1"
python -m frugalrouter.cli --tasks data/sample_tasks.json --output outputs/sample_results.json
python -m frugalrouter.report --log logs/decisions.jsonl
```

## Docker

Build a linux/amd64 image:

```powershell
docker buildx build --platform linux/amd64 -t frugalrouter:latest .
```

Local container smoke test:

```powershell
New-Item -ItemType Directory -Force -Path input,output | Out-Null
Copy-Item data/sample_tasks.json input/tasks.json
docker run --rm `
  -e FIREWORKS_API_KEY=$env:FIREWORKS_API_KEY `
  -e FIREWORKS_BASE_URL="https://api.fireworks.ai/inference/v1" `
  -e ALLOWED_MODELS="minimax-m3,kimi-k2p7-code,gemma-4-31b-it,gemma-4-26b-a4b-it,gemma-4-31b-it-nvfp4" `
  -v ${PWD}/input:/input `
  -v ${PWD}/output:/output `
  frugalrouter:latest
```

For submission, push a public linux/amd64 image to Docker Hub or GitHub Container Registry.

## Project Layout

- `frugalrouter/cli.py` - official Track 1 runner.
- `frugalrouter/router.py` - route decision logic.
- `frugalrouter/task_classifier.py` - category classifier.
- `frugalrouter/model_policy.py` - allowed-model selection.
- `frugalrouter/providers/local.py` - deterministic zero-token shortcuts.
- `frugalrouter/providers/fireworks.py` - Fireworks judging-proxy client.
- `frugalrouter/decision_log.py` - JSONL decision logging.
- `frugalrouter/report.py` - summary report for route decisions.
- `config/models.json` - model routing and token caps.
- `docs/track1_battle_plan.md` - solo-builder plan for Track 1.

## Troubleshooting

If Docker Desktop dies at startup with `remove ...: The file cannot be accessed by the system`, run:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/fix_docker.ps1
```

If the Twitch web player buffers or fails, launch the lablab.ai Twitch stream through VLC:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/watch_lablab_twitch.ps1
```
