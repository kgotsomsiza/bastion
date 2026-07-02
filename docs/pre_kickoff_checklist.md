# Pre-Kickoff Checklist

## Access

- lablab.ai enrollment approved.
- AMD AI Developer Program account active.
- AMD Developer Community Discord joined.
- lablab.ai Discord joined.
- Fireworks account created.
- Fireworks API key created and stored.
- `FIREWORKS_API_KEY` tested in a fresh terminal.
- AMD Developer Cloud access checked when credits arrive.

## Repo

- Public GitHub repository created.
- README includes setup and usage instructions.
- Docker build succeeds.
- Local sample run succeeds.
- Remote Fireworks smoke test succeeds.
- Decision logs are written to `logs/decisions.jsonl`.
- Outputs are written to `outputs/*.jsonl`.

## Kickoff Capture

- Final Track 1 task schema.
- Allowed local models.
- Allowed Fireworks models.
- Accuracy threshold.
- Scoring command or submission harness.
- Container runtime constraints.
- Submission deadline in local timezone.

## First Hour After Kickoff

1. Copy the revealed task examples into `data/`.
2. Update `frugalrouter/types.py` if the schema differs.
3. Add task-specific local heuristics.
4. Add task-specific verifier rules.
5. Run a 20-task batch with remote disabled.
6. Run the same batch with remote enabled.
7. Tune `remote_threshold` based on failures.

