"""CPU development instrument with a fixed hardware start for every episode.

This is a distinct convention from legacy ALE reset_game. It repeats a certified
deterministic construction, including format detection, and charges every load.
No RAM, reward, lives or terminal flag crosses the inherited pixel-only RPC.
"""
from collections import deque
import hashlib
from pathlib import Path
import random

from ragnarok.environments.emu_pool import EmulatorPool, _PixelEngine, _serve
from ragnarok.environments.frame_budget_v70 import BudgetExceeded, FrameBudget


BINARY_SHA256 = "b7e29d7c2662be5a7d49d9e7151637a7842547ecc149b2b2c6771b9e02159d7c"
SYSTEM_SEED = 4753849
CONVENTION = "fixed_hardware_cold_start_v1"
# Only the declared development ROMs. No confirmation games are opened here.
ROM_SPECS = {
    "pong": ("60e0ea3cbe0913d39803477945e9e5ec", 132),
    "breakout": ("f34f08e5eb96e500e851a80be3277a56", 132),
    "freeway": ("8e0ab801b1705a740b476b7f588c6d16", 132),
    "seaquest": ("240bfbac5163af4df5ae713985386f92", 128),
    "ms_pacman": ("87e79cd41ce136fd4f72cc6e2c161bee", 132),
    "asteroids": ("ccbd36746ed4525821a8083b0d6d2c2c", 132),
}


class _ColdPixelEngine(_PixelEngine):
    def __init__(self, config):
        import ale_py
        from ale_py.roms import get_rom_path

        if ale_py.__version__ != "0.12.1" or config.game not in ROM_SPECS:
            raise ValueError("Cold-start certification covers the six development ROMs on ALE0.12.1 only")
        if hashlib.sha256(Path(ale_py._ale_py.__file__).read_bytes()).hexdigest() != BINARY_SHA256:
            raise ValueError("ALE binary differs from the certified deterministic construction")
        self._rom_path = get_rom_path(config.game)
        md5, self._bootstrap_frames = ROM_SPECS[config.game]
        if hashlib.md5(self._rom_path.read_bytes()).hexdigest() != md5:
            raise ValueError("ROM differs from the cold-start certification")
        self.config = config
        self._budget = FrameBudget(experiment_fraction=config.experiment_fraction,
                                   max_charged_frames=config.max_charged_frames)
        self._sticky_rng = random.Random(config.seed)
        self._last_action = 0
        self._reset_frames_spent = self._loads = 0
        self._saves = {}
        self._native_wrapper = None
        self._cold_start(experimental=False)

    def _cold_start(self, *, experimental):
        import ale_py
        admit = self._budget.can_experiment if experimental else self._budget.can_play
        if not admit(self._bootstrap_frames):
            raise BudgetExceeded("Complete cold construction does not fit the charged budget")
        native = ale_py.ALEInterface()
        native.setInt("random_seed", self.config.seed)
        native.setInt("system_random_seed", SYSTEM_SEED)
        native.setInt("frame_skip", 1)
        native.setInt("max_num_frames_per_episode", self.config.max_episode_frames)
        native.setFloat("repeat_action_probability", 0.0)
        for name in ("color_averaging", "display_screen", "sound"):
            native.setBool(name, False)
        native.loadROM(self._rom_path)
        record = self._budget.record_experiment if experimental else self._budget.record_play
        record(self._bootstrap_frames, agent_steps=0)
        self._loads += 1
        self._reset_frames_spent += self._bootstrap_frames - 60
        if native.getFrameNumber() != 0 or native.getEpisodeFrameNumber() != 0:
            raise RuntimeError("Cold-start trace differs from certification; stop the instrument")
        first = native.getScreenRGB().copy()
        self._frames = deque([first.copy(), first], maxlen=2)
        self._actions = {a.value: a for a in native.getLegalActionSet()}
        self._last_action = 0
        self._ale = native if self._native_wrapper is None else self._native_wrapper(native)

    def set_native_wrapper(self, wrapper):
        """Experimenter-private hook, absent from RPC; survives native recreation."""
        if self._native_wrapper is not None:
            raise ValueError("A native wrapper is already installed")
        self._native_wrapper = wrapper
        self._ale = wrapper(self._ale)

    def step(self, action):
        self._action(action)
        terminal = self._ale.game_over()
        experimental = bool(self._saves)
        admit = self._budget.can_experiment if experimental else self._budget.can_play
        cost = self.config.frame_skip + (self._bootstrap_frames if terminal else 0)
        if not admit(cost):
            raise BudgetExceeded("Cold construction and complete repeat do not fit the charged budget")
        if terminal:
            self._cold_start(experimental=experimental)
        return self._advance(action, self.config.frame_skip, experimental=experimental)

    def accounting(self):
        report = self._budget.report()
        report.update(reset_emulated_frames=self._reset_frames_spent,
                      bootstrap_emulated_frames=self._bootstrap_frames,
                      format_detection_emulated_frames=60 * self._loads,
                      cold_loads=self._loads, reset_convention=CONVENTION,
                      system_random_seed=SYSTEM_SEED, ale_binary_sha256=BINARY_SHA256,
                      sticky_implementation="worker-python-random-per-frame",
                      sticky_probability=self.config.sticky_probability)
        return report


def _serve_cold(connection, config):
    _serve(connection, config, _ColdPixelEngine)


class ColdEmulatorPool(EmulatorPool):
    _worker_target = staticmethod(_serve_cold)
