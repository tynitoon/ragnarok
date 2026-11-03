"""v85 arm registry (docs/ARC4_V85_PROTOCOL_EN.md Part B item 7). Evaluator side.

An arm is a module `ragnarok/evaluation/arms_v85/<arm>.py` (lower-case name), found by name at run time,
so a new arm is a new file and neither this package nor scripts/v85_job.py is edited. This package
imports no arm and no instrument: an arm's closure (§13.4) holds only what that arm imports.

Arm module interface:
- `build(ctx) -> dict`, with any of
    steer     injected steering object of Explorer2s (pick_cell, pick_process, on_put, on_iter,
              to_state, from_state);
    q5        injected Q5 process object (first_slot, next_slot, __len__, to_state, from_state); the
              bandit gains its fifth arm once the library is non-empty (Explorer2s);
    chooser   chooser(ex, env, e) -> (cell, process, first slot); its state is checkpointed through
              to_state()/from_state(d) when it has them;
    kwargs    further Explorer2s keyword arguments, e.g. after=o.after (called after each executed
              rollout) or script=<ALE actions of expansion 0's rollout> (D);
    explorer  a factory(body, budget=, seed=, steer=, q5=, chooser=, boundary=, **kwargs) -> an Explorer2s
              (subclass); default Explorer2s;
  and any private keys the arm keeps for its captions. Mutable arm state must live in steer or q5 (the
  explorer's to_state), in the chooser or a kwargs object with to_state/from_state (checkpointed by the
  job), or in the explorer; anything else is rebuilt by `build` on resume, so `build` is deterministic.
- optional `ABORTS`: a tuple of exception classes that end a run as aborted (O_4-dice's DiceMismatch, A.7):
  the job writes the run JSON with `aborted` = the message and the state reached (counted, not retried).
- optional `captions(hooks, run) -> dict` (JSON-serialisable), called after the run for the run JSON's
  "instrument" field; `run` = dict(ex=explorer, env=SenseEnv85, trace=Trace85, ctx=Ctx). A
  `captions(hooks)` with one parameter is also accepted.
"""
import dataclasses
import importlib
import inspect
import re
from typing import Any, Callable, Optional

HOOKS = ("steer", "q5", "chooser", "explorer", "kwargs")


@dataclasses.dataclass
class Ctx:
    """What an arm may use to build its hooks (all evaluator-held; the agent sees only the hooks' effect)."""
    game: str                      # unit (ALE game name; evaluator side only)
    start: str                     # "A" or "B"
    seed: int                      # the explorer's agent seed of this run
    E: int                         # the start's emulator and entry seed E_s (9819 / 9820; 9800 in smoke runs)
    s1: dict                       # this start: blob, S, lives, step, context, record, actions (ALE indices
                                   # above s1, uint8), cum (per-step cum above s1)
    other_s1: Optional[dict]       # the other start's s1 dict, None if that start is invalid
    body: Any                      # this start's cached Body (hash-checked); the explorer uses this object
    ladder: list                   # [M_0..M_10]
    rungs: list                    # counted rungs of this run (K_{g,s} after F1, else K_g)
    k_first: int                   # min of `rungs`
    k_ref: int                     # max K_g: the evaluator stop M_{k_ref}
    cap: int                       # billed frames from the clock start
    units: list                    # the units of this stage in draw order (derangements, e.g. K~mac)
    paths: dict                    # 'v85' output root, 's0' start dir, 'starts' starts.json, 'out' run JSON
    starts: dict                   # the parsed starts.json
    load_s1: Callable              # load_s1(game, start) -> s1 dict or None (hash-checked)
    load_body: Callable            # load_body(game, start) -> a fresh Body (hash-checked)
    smoke: bool = False


def load(name):
    if not re.fullmatch(r"[a-z][a-z0-9_]*", name):
        raise ValueError(f"bad arm name {name!r}")
    return importlib.import_module(f"{__name__}.{name}")


def build(name, ctx):
    """(module, hooks) with every key of HOOKS present (None, or {} for kwargs, when not injected)."""
    mod = load(name)
    hooks = dict(dict.fromkeys(HOOKS), kwargs={})
    hooks.update(mod.build(ctx) or {})
    return mod, hooks


def stateful(hooks):
    """{name: object} of the hook objects the job checkpoints itself: the chooser and the kwargs values with
    to_state (steer and q5 are in the explorer's own state); each object once."""
    out, seen = {}, {id(hooks["steer"]), id(hooks["q5"])}
    for n, o in [("chooser", hooks["chooser"])] + sorted(hooks["kwargs"].items()):
        if o is not None and hasattr(o, "to_state") and id(o) not in seen:
            out[n] = o
            seen.add(id(o))
    return out


def captions(mod, hooks, run):
    fn = getattr(mod, "captions", None)
    if fn is None:
        return {}
    return fn(hooks, run) if len(inspect.signature(fn).parameters) >= 2 else fn(hooks)
