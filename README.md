# FrugalRouter

Track 1 prep scaffold for the AMD Developer Hackathon: ACT II.

FrugalRouter is a token-efficient hybrid routing agent. It tries to answer with a local strategy first, verifies confidence locally, and only spends Fireworks API tokens when the answer is risky.

## Strategy

```text
task -> classify -> local draft -> local verifier -> confidence
     -> submit local answer if safe
     -> call Fireworks if risky
     -> log route, confidence, tokens, latency, and fallback reason
```

The kickoff rules will reveal the real task format and allowed models. Until then, this repo gives you the boring-but-important plumbing: Docker, env config, provider abstraction, decision logs, and a quick sample runner.

## Quick Start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
python -m frugalrouter.cli --tasks data/sample_tasks.jsonl --output outputs/sample_results.jsonl
```

Set your Fireworks key when you want to test remote fallback:

```powershell
$env:FIREWORKS_API_KEY="fw_your_key_here"
python -m frugalrouter.cli --tasks data/remote_smoke_task.jsonl --output outputs/remote_smoke_results.jsonl --allow-remote
python -m frugalrouter.report --log logs/decisions.jsonl
```

## Docker

```powershell
docker build -t frugalrouter .
docker run --rm -v ${PWD}/outputs:/app/outputs frugalrouter
```

With Fireworks:

```powershell
docker run --rm `
  -e FIREWORKS_API_KEY=$env:FIREWORKS_API_KEY `
  -e FRUGAL_ALLOW_REMOTE=1 `
  -v ${PWD}/outputs:/app/outputs `
  frugalrouter
```

## Project Layout

- `frugalrouter/cli.py` - JSONL task runner.
- `frugalrouter/router.py` - route decision logic.
- `frugalrouter/providers/local.py` - local answer baseline.
- `frugalrouter/providers/fireworks.py` - Fireworks API provider.
- `frugalrouter/evaluation/verifier.py` - local confidence checks.
- `frugalrouter/decision_log.py` - JSONL decision logging.
- `frugalrouter/report.py` - summary report for route decisions.
- `config/models.json` - model and routing configuration.
- `docs/pre_kickoff_checklist.md` - what to sort before kickoff.
- `docs/track1_battle_plan.md` - solo-builder plan for Track 1.

## Kickoff Swap Points

When the organizers reveal the tasks:

1. Update `Task` fields in `frugalrouter/types.py` if needed.
2. Update `LocalProvider.answer()` for the task-specific cheap baseline.
3. Update `LocalVerifier.score()` with the task-specific validation rules.
4. Update `config/models.json` with the revealed Fireworks model IDs.
5. Run batches and tune `remote_threshold`.

## Troubleshooting

If Docker Desktop dies at startup with `remove ...: The file cannot be accessed by the system` (stale Unix-socket files after the Windows AF_UNIX regression), run:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/fix_docker.ps1
```

## Notes

The Fireworks model in `config/models.json` is a placeholder for prep. Replace it with the model IDs revealed at kickoff or a model you confirm in the Fireworks dashboard.

## Kickoff Stream Helper

If the Twitch web player buffers or fails, launch the lablab.ai Twitch stream through VLC:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/watch_lablab_twitch.ps1
```
