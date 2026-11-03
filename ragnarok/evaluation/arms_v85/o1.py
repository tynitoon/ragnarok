"""O_1 (docs/ARC4_V85_PROTOCOL_EN.md A.7, G-build test 6): one candidate through the sandbox; byte-identical
to V2s apart from oracle frames and captions. Smoke and identity runs only, never a gate row."""
from ragnarok.evaluation.arms_v85 import o4
from ragnarok.evaluation.arms_v85.o4 import captions   # noqa: F401


def build(ctx):
    return o4.build(ctx, m=1)
