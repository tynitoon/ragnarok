"""Experimenter-only adjacent RAM history for a two-raw-frame observation.

Ms. Pac-Man decoding needs the RAM before each source RGB frame, hence three
snapshots. This capture hook is never installed in an agent RPC interface.
Observed RAM/frame discontinuities are rejected. Native hidden state mutation
cannot be detected here; the separately sealed live capture has no restore path.
"""
import numpy as np


class _TripleRAMTap:
    def __init__(self,native):
        self.native=native
        initial=np.asarray(native.getRAM())
        if initial.shape!=(128,) or initial.dtype!=np.uint8:
            raise ValueError('Expected emulator uint8 RAM[128]')
        frame=int(native.getEpisodeFrameNumber())
        if frame!=0:
            raise ValueError('Capture RAM history must begin at a fresh cold frame zero')
        self.before_previous=initial.copy()
        self.previous=initial.copy()
        self.current=initial.copy()
        self.frame_numbers=(frame,frame,frame)
        self.recorded_actions=0

    def __getattr__(self,name):
        return getattr(self.native,name)

    @property
    def cold_history_fallback(self):
        return self.recorded_actions<2

    def act(self,action):
        before=np.asarray(self.native.getRAM()).copy()
        frame=int(self.native.getEpisodeFrameNumber())
        if frame!=self.frame_numbers[-1] or not np.array_equal(before,self.current):
            raise RuntimeError('Native state changed outside the capture tap; do not bridge a reset or restore')
        value=self.native.act(action)
        after=np.asarray(self.native.getRAM())
        next_frame=int(self.native.getEpisodeFrameNumber())
        if next_frame!=frame+1 or after.shape!=(128,) or after.dtype!=np.uint8:
            raise RuntimeError('Native act must advance exactly one ordinary frame')
        self.before_previous=self.previous.copy()
        self.previous=before.copy()
        self.current=after.copy()
        self.frame_numbers=(self.frame_numbers[1],frame,next_frame)
        self.recorded_actions+=1
        return value
