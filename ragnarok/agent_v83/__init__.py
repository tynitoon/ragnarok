"""ARC 4 v83 'promise map' agent (docs/ARC4_V83_PROTOCOL.md).

The agent sees screen bytes, the score's rises and the lives count (owner decision 2026-10-22), and acts
with buttons. It receives a senses object from the evaluator and never imports ragnarok.evaluation,
torch, the emulator or any earlier learner (tests/test_isolation_v83.py). No game name appears here.
"""
