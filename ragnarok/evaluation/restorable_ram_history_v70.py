"""Private three-RAM scorer history restored with the pixel engine's snapshot.

The sealed capture tap refuses unannounced restores. This new evaluator adapter
stores its exact adjacent history beside the full native/RGB/sticky snapshot;
the monotone charge ledger is still owned by the unchanged cold engine.
No history, state, reward or labels are added to any observation RPC.
"""
import numpy as np
from ragnarok.evaluation.triple_ram_tap_v70 import _TripleRAMTap
from ragnarok.environments.cold_emu_pool_v70 import _ColdPixelEngine


class _RestorableTripleRAMTap(_TripleRAMTap):
    def __init__(self,native,on_frame):
        if not callable(on_frame):raise ValueError('A private scorer callback is required')
        super().__init__(native)
        self._on_frame=on_frame

    def act(self,action):
        # The engine calls the scorer after paying and preserving the real act.
        # A failing decoder must not hide the frame already executed by ALE.
        return super().act(action)

    def export_history(self):
        if (int(self.native.getEpisodeFrameNumber())!=self.frame_numbers[-1]
                or not np.array_equal(self.native.getRAM(),self.current)):
            raise RuntimeError('Scorer history is not aligned with native state')
        return (self.before_previous.copy(),self.previous.copy(),self.current.copy(),
                tuple(self.frame_numbers),self.recorded_actions)

    def import_history(self,saved):
        before,previous,current,numbers,actions=saved
        if (any(not isinstance(a,np.ndarray) or a.shape!=(128,) or a.dtype!=np.uint8
                for a in (before,previous,current)) or type(actions) is not int or actions<0
                or numbers!=(max(0,actions-2),max(0,actions-1),actions)
                or int(self.native.getEpisodeFrameNumber())!=actions
                or not np.array_equal(self.native.getRAM(),current)
                or (actions==0 and not (np.array_equal(before,previous) and np.array_equal(previous,current)))
                or (actions==1 and not np.array_equal(before,previous))):
            raise ValueError('Saved scorer history does not match the restored native frame/RAM')
        self.before_previous,self.previous,self.current=(a.copy() for a in (before,previous,current))
        self.frame_numbers=tuple(numbers);self.recorded_actions=actions


class _ScorerHistoryColdEngine(_ColdPixelEngine):
    def __init__(self,config,on_frame):
        if (config.game not in ('ms_pacman','asteroids') or config.mode!='lab' or config.frame_skip!=4
                or not callable(on_frame)):
            raise ValueError('This private scorer engine requires a declared repeat4/lab game and callback')
        super().__init__(config)
        self.set_native_wrapper(lambda native:_RestorableTripleRAMTap(native,on_frame))

    def _advance(self,action,frames,*,experimental):
        """Same sticky/terminal/charge semantics; scoring occurs AFTER the paid act."""
        self._action(action)
        for _ in range(frames):
            if self._ale.game_over():break
            chosen=self._last_action if self._sticky_rng.random()<self.config.sticky_probability else action
            before=self._ale.getEpisodeFrameNumber()
            self._ale.act(self._actions[chosen])
            elapsed=self._ale.getEpisodeFrameNumber()-before
            if elapsed!=1:raise RuntimeError('Expected exactly one native frame')
            self._last_action=chosen
            if experimental:self._budget.record_experiment(1)
            else:self._budget.record_play(1,agent_steps=0)
            self._frames.append(self._ale.getScreenRGB().copy())
            self._ale._on_frame(self._ale)
        if experimental:self._budget.record_experiment(0,agent_steps=1)
        else:self._budget.record_play(0,agent_steps=1)
        return self.observe()

    def _capture(self):
        return (*super()._capture(),self._ale.export_history())

    def _restore(self,saved):
        # Native, RGB and sticky restoration still charges60 in the sealed base.
        # Neither the saved scorer history nor this method contains a ledger.
        if not isinstance(saved,tuple) or len(saved)!=5:
            raise ValueError('A complete native/pixel/sticky/scorer snapshot is required')
        super()._restore(saved[:4])
        self._ale.import_history(saved[4])
