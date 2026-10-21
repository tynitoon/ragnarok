---
name: ragnarok-review
description: Use at a scientific milestone of the Ragnarok project or before trusting new agent or evaluation code (before freezing a protocol, after a surprising result): an adversarial multi-agent review run with the Workflow tool.
---

# Adversarial review for Ragnarok

Standing preference of the owner: several independent agents at major milestones, dissent over
consent. Reviews found real defects on 21 Oct 2026 (an explorer using a different death signal than
the validated one, duplicated roots, fold leakage, a per-seed criterion missing, an input defect).

## Shape (Workflow tool, pipeline, <= ~10 agents)
1. **Review**: 3-4 lenses run in parallel, each with a schema `{findings:[{title,file,line,severity:
   invalidates|major|minor,evidence,failure_scenario,fix}],notes}`. Useful lenses:
   - correctness of the learning code (state restored per branch: tracker pos/miss, previous views,
     HUD reader, vanish count; ties; labels; frames accounting);
   - evaluation fairness vs the frozen protocol (same live conditions for arms and baselines, seeds,
     criterion exactly as written, skipped jobs, stale outputs);
   - oracle leakage (trace every input and label back to screen bytes; RAM/true score/lives only in
     evaluation and diagnostics);
   - for offline benchmarks: fold leakage, yardstick, input coverage.
2. **Verify**: one skeptic per lens tries to REFUTE each finding against the code (default real=false).
3. **Synthesize**: re-check the surviving findings, classify (a) invalidates running measurements,
   (b) fix before the next experiment, (c) minor; give exact code changes.

## Context every agent needs (write it in the script, they do not see this conversation)
- Paths of the files and worktree under review; what the change is meant to do; the protocol file.
- HARD RULE: read and grep only; never run python, emulators or torch (memory is reserved for Tavern,
  see skill `ragnarok-run`); never edit files.
- The honesty rules (skill `ragnarok-protocol`).

## Practicalities
- Build prompts with plain string concatenation: backticks inside template literals break the
  script parser.
- Results: read `<transcript dir>/journal.jsonl` (`type == "result"` lines); the notification text is
  truncated. Write long syntheses to the scratchpad and read them with the Read tool.
- Apply (a) and (b) fixes before freezing; record what was not fixed and why in the protocol.
