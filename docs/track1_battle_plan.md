# Track 1 Battle Plan

## North Star

Win by spending remote tokens only when local confidence is too low to survive the accuracy threshold.

## Pre-Kickoff Build

- Keep the repo small and containerized.
- Make local runs fast.
- Log every route decision.
- Make threshold tuning a config change, not a code rewrite.
- Prepare a Fireworks smoke test, but avoid burning credits.

## Kickoff Questions To Answer

- What exact task schema is scored?
- What is the accuracy threshold?
- Are outputs exact-match, semantic-judged, or unit-tested?
- Which local models are allowed?
- Which Fireworks models are allowed?
- Are local model tokens truly counted as zero in all cases?
- Is there a hidden test set?
- How is failure handled: zero score, partial credit, or retry?

## Winning Loop

1. Run the local-only baseline.
2. Tag failures by type.
3. Add cheap deterministic rules for common failures.
4. Add local verification checks.
5. Enable remote fallback only for risky buckets.
6. Compare token spend against accuracy.
7. Tune `remote_threshold`.
8. Repeat in small batches.

## Error Taxonomy

- Format failures: wrong JSON, too verbose, wrong label, missing field.
- Knowledge failures: factual question not covered locally.
- Reasoning failures: math, code, multi-step logic.
- Ambiguity failures: prompt underspecified or multiple valid answers.
- Overcall failures: remote used when local would have passed.
- Undercall failures: local used when remote was needed.

## Submission Story

The README and video should emphasize that FrugalRouter is not just a wrapper around an API. It is a cost-aware agent with local drafting, local verification, remote escalation, and decision logging.

