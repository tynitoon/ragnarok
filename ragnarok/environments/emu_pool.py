"""ARC 4 CPU instrument: isolated ALE workers, pixel observations and opaque saves.

This is an instrument foundation, not gate 0 or an agent that has learned to see.
Only RGB pixels cross the observation interface. Rewards, RAM, lives and ALEState
stay inside workers. Accounting is a separate experimenter-side method.
"""
from collections import deque
from dataclasses import dataclass
import math
import hashlib
import multiprocessing as mp
import random
import secrets

import numpy as np

from ragnarok.environments.frame_budget_v70 import BudgetExceeded, FrameBudget, nonnegative_int


@dataclass(frozen=True)
class EmulatorConfig:
    game: str = "pong"
    seed: int = 0
    frame_skip: int = 4
    sticky_probability: float = 0.25
    mode: str = "lab"
    experiment_fraction: float = 0.2
    max_charged_frames: int = 100_000
    max_snapshots: int = 64
    max_episode_frames: int = 108_000

    def __post_init__(self):
        if not isinstance(self.game, str) or not self.game:
            raise ValueError("A ROM id is required")
        nonnegative_int(self.seed, "seed")
        if self.seed >= 2**31:
            raise ValueError("seed must fit ALE's signed 32-bit setting")
        for name in ("frame_skip", "max_snapshots"):
            if not 1 <= nonnegative_int(getattr(self, name), name) <= 64:
                raise ValueError(f"{name} must be in 1..64")
        if self.mode not in ("lab", "live"):
            raise ValueError("mode must be lab or live")
        if not 1 <= nonnegative_int(self.max_episode_frames, "max_episode_frames") <= 108_000:
            raise ValueError("max_episode_frames must be in 1..108000")
        if not math.isfinite(self.sticky_probability) or not 0 <= self.sticky_probability <= 1:
            raise ValueError("sticky_probability must be in [0, 1]")
        FrameBudget(experiment_fraction=self.experiment_fraction,
                    max_charged_frames=self.max_charged_frames)


class _PixelEngine:
    """Private, one-worker engine. No native state is returned to the caller."""
    def __init__(self, config):
        import ale_py
        from ale_py.roms import get_rom_path

        # Reset costs are ROM/version/mode-specific and absent from ALE's frame getters.
        # Only this audited development fixture is enabled at the first CPU milestone.
        if ale_py.__version__ != "0.12.1" or config.game != "pong":
            raise ValueError("Only Pong mode 0 on ALE 0.12.1 has an audited bootstrap cost here")
        rom_path = get_rom_path(config.game)
        if hashlib.md5(rom_path.read_bytes()).hexdigest() != "60e0ea3cbe0913d39803477945e9e5ec":
            raise ValueError("Pong ROM does not match the audited bootstrap fixture")
        self.config = config
        self._budget = FrameBudget(experiment_fraction=config.experiment_fraction,
                                   max_charged_frames=config.max_charged_frames)
        # Console AUTO-DETECT advances 60 frames before ALE's 72-frame reset.
        self._bootstrap_frames = 132
        if not self._budget.can_play(self._bootstrap_frames):
            raise BudgetExceeded("Initial Pong bootstrap requires 132 charged frames")
        self._sticky_rng = random.Random(config.seed)
        self._last_action = 0
        self._reset_frames_spent = 0
        self._ale = ale_py.ALEInterface()
        self._ale.setInt("random_seed", config.seed)
        # This is ALE 0.12.1's existing default; distinct from its action RNG.
        self._ale.setInt("system_random_seed", 4753849)
        self._ale.setInt("frame_skip", 1)  # wrapper counts and pools each real frame
        # ALEState omits native sticky action history. Own it, including its RNG, here.
        self._ale.setFloat("repeat_action_probability", 0.0)
        self._ale.setBool("color_averaging", False)
        self._ale.setInt("max_num_frames_per_episode", config.max_episode_frames)
        self._ale.setBool("display_screen", False)
        self._ale.setBool("sound", False)
        self._ale.loadROM(rom_path)
        self._budget.record_play(self._bootstrap_frames, agent_steps=0)
        self._reset_frames_spent = 72
        self._actions = {a.value: a for a in self._ale.getLegalActionSet()}
        first = self._ale.getScreenRGB().copy()
        self._frames = deque([first.copy(), first], maxlen=2)
        self._saves = {}

    def observe(self, raw=False):
        if type(raw) is not bool:
            raise ValueError("raw must be boolean")
        if raw or len(self._frames) == 1:
            return self._frames[-1].copy()
        return np.maximum(self._frames[-2], self._frames[-1])

    def _action(self, action):
        if type(action) is not int or action not in self._actions:
            raise ValueError("Action must be an integer from the controller's legal action set")

    def _lab_only(self):
        if self.config.mode != "lab":
            raise ValueError("Savestates and branching are disabled in live mode")

    def _capture(self):
        return (self._ale.cloneState(include_rng=True), tuple(frame.copy() for frame in self._frames),
                self._sticky_rng.getstate(), self._last_action)

    def _restore(self, saved):
        self._budget.record_experiment(0, restores=1)
        self._ale.restoreState(saved[0])
        self._frames = deque((frame.copy() for frame in saved[1]), maxlen=2)
        self._sticky_rng.setstate(saved[2])
        self._last_action = saved[3]

    def _advance(self, action, frames, *, experimental):
        self._action(action)
        for _ in range(frames):
            # Stop exactly at terminal; no free reset or repeated fictitious frames in a branch.
            if self._ale.game_over():
                break
            chosen = self._last_action if self._sticky_rng.random() < self.config.sticky_probability else action
            before = self._ale.getEpisodeFrameNumber()
            self._ale.act(self._actions[chosen])  # reward is intentionally discarded
            elapsed = self._ale.getEpisodeFrameNumber() - before
            if elapsed != 1:
                raise RuntimeError(f"Expected one real ALE frame, got {elapsed}; instrument must be reviewed")
            self._last_action = chosen
            if experimental:
                self._budget.record_experiment(1)
            else:
                self._budget.record_play(1, agent_steps=0)
            self._frames.append(self._ale.getScreenRGB().copy())
        if experimental:
            self._budget.record_experiment(0, agent_steps=1)
        else:
            self._budget.record_play(0, agent_steps=1)
        return self.observe()

    def step(self, action):
        self._action(action)
        reset_cost = 72 if self._ale.game_over() else 0
        # Explicit clone/restore opens a lab session: replay frames cannot masquerade as ordinary play.
        experimental = bool(self._saves)
        admit = self._budget.can_experiment if experimental else self._budget.can_play
        if not admit(self.config.frame_skip + reset_cost):
            raise BudgetExceeded("A complete action repeat does not fit the frame ceiling")
        if reset_cost:
            self._ale.reset_game()
            record = self._budget.record_experiment if experimental else self._budget.record_play
            record(reset_cost, agent_steps=0)
            self._reset_frames_spent += reset_cost
            first = self._ale.getScreenRGB().copy()
            self._frames = deque([first.copy(), first], maxlen=2)
            self._last_action = 0
        return self._advance(action, self.config.frame_skip, experimental=experimental)

    def clone(self):
        """Open a laboratory session; steps stay experimental until all tokens are released."""
        self._lab_only()
        if len(self._saves) >= self.config.max_snapshots:
            raise ValueError("Snapshot limit reached; release an old token first")
        token = secrets.token_hex(16)
        self._saves[token] = self._capture()
        return token

    def restore(self, token):
        self._lab_only()
        self._restore(self._saves[token])
        return self.observe()

    def release(self, token):
        self._lab_only()
        del self._saves[token]

    def branch_pair(self, action, control=0, frames=4):
        self._lab_only()
        self._action(action)
        self._action(control)
        if not 1 <= nonnegative_int(frames, "frames") <= 64:
            raise ValueError("Branch length must be in 1..64 emulated frames")
        if not self._budget.can_experiment(2 * frames, restores=2):
            raise BudgetExceeded("Complete branch pair, including two restores, does not fit")
        anchor = self._capture()
        try:
            intervention = self._advance(action, frames, experimental=True)
        finally:
            self._restore(anchor)
        try:
            control_pixels = self._advance(control, frames, experimental=True)
        finally:
            self._restore(anchor)
        return intervention, control_pixels

    def accounting(self):
        report = self._budget.report()
        report.update(reset_emulated_frames=self._reset_frames_spent,
                      bootstrap_emulated_frames=self._bootstrap_frames,
                      format_detection_emulated_frames=60,
                      system_random_seed=4753849,
                      sticky_implementation="worker-python-random-per-frame",
                      sticky_probability=self.config.sticky_probability)
        return report


def _serve(connection, config, engine_type=_PixelEngine):
    try:
        engine = engine_type(config)
        connection.send((True, engine.observe()))
        methods = {name: getattr(engine, name) for name in
                   ("observe", "step", "clone", "restore", "release", "branch_pair", "accounting")}
        while True:
            command, args = connection.recv()
            if command == "close":
                connection.send((True, None))
                break
            try:
                if command not in methods:
                    raise ValueError("Unknown emulator command")
                connection.send((True, methods[command](*args)))
            except Exception as exc:
                connection.send((False, (type(exc).__name__, str(exc))))
    except EOFError:
        pass
    except Exception as exc:
        try:
            connection.send((False, (type(exc).__name__, str(exc))))
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        connection.close()


class EmulatorPool:
    """A small spawn-worker pool. Calls send requests before waiting on workers.

    A future socket transport can implement the same closed command vocabulary.
    This version deliberately has no remote endpoint or ground-truth channel.
    """
    _worker_target = staticmethod(_serve)

    def __init__(self, configs, *, timeout=30.0):
        configs = list(configs)
        if not 1 <= len(configs) <= 16 or not all(isinstance(c, EmulatorConfig) for c in configs):
            raise ValueError("Supply 1..16 EmulatorConfig instances")
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be positive and finite")
        self._timeout = timeout
        self._connections, self._processes = [], []
        self._closed = False
        ctx = mp.get_context("spawn")
        try:
            for config in configs:
                parent, child = ctx.Pipe()
                process = ctx.Process(target=self._worker_target, args=(child, config), daemon=True)
                try:
                    process.start()
                except BaseException:
                    parent.close()
                    child.close()
                    raise
                child.close()
                self._connections.append(parent)
                self._processes.append(process)
            for connection in self._connections:
                self._receive(connection)
        except BaseException:
            self.close()
            raise

    def _receive(self, connection):
        if not connection.poll(self._timeout):
            raise TimeoutError("Emulator worker did not answer")
        ok, value = connection.recv()
        if not ok:
            name, message = value
            if name == "BudgetExceeded":
                raise BudgetExceeded(message)
            if name in ("ValueError", "KeyError"):
                raise ValueError(message)
            raise RuntimeError(f"Emulator worker {name}: {message}")
        return value

    def _one(self, index, command, *args):
        if self._closed:
            raise RuntimeError("Emulator pool is closed")
        if type(index) is not int or not 0 <= index < len(self._connections):
            raise IndexError("Invalid worker index")
        connection = self._connections[index]
        try:
            connection.send((command, args))
            return self._receive(connection)
        except (TimeoutError, EOFError, OSError, RuntimeError, KeyboardInterrupt) as exc:
            if isinstance(exc, BudgetExceeded):
                raise
            self.close()
            raise

    def observe(self, index=0, *, raw=False):
        return self._one(index, "observe", raw)

    def step(self, actions):
        if self._closed:
            raise RuntimeError("Emulator pool is closed")
        actions = list(actions)
        if len(actions) != len(self._connections):
            raise ValueError("Provide one controller action per worker")
        if any(type(action) is not int or not 0 <= action < 18 for action in actions):
            raise ValueError("Controller actions must be integers in 0..17")
        try:
            for connection, action in zip(self._connections, actions):
                connection.send(("step", (action,)))
        except (BrokenPipeError, EOFError, OSError, KeyboardInterrupt):
            self.close()
            raise
        values, errors = [], []
        # Drain all replies after a recoverable error so the next request cannot read an old reply.
        for connection in self._connections:
            try:
                values.append(self._receive(connection))
            except KeyboardInterrupt:
                self.close()
                raise
            except Exception as exc:
                errors.append(exc)
        if errors:
            if any(isinstance(exc, (TimeoutError, EOFError, OSError)) or
                   isinstance(exc, RuntimeError) and not isinstance(exc, BudgetExceeded) for exc in errors):
                self.close()
            raise errors[0]
        return values

    def clone(self, index=0):
        return self._one(index, "clone")

    def restore(self, token, index=0):
        return self._one(index, "restore", token)

    def release(self, token, index=0):
        return self._one(index, "release", token)

    def branch_pair(self, action, *, control=0, frames=4, index=0):
        return self._one(index, "branch_pair", action, control, frames)

    def accounting(self, index=0):
        return self._one(index, "accounting")

    def close(self):
        if self._closed:
            return
        self._closed = True
        for connection, process in zip(self._connections, self._processes):
            if process.is_alive():
                try:
                    connection.send(("close", ()))
                except (BrokenPipeError, EOFError, OSError):
                    pass
            connection.close()
        for process in self._processes:
            process.join(timeout=2)
            if process.is_alive():
                process.terminate()
                process.join(timeout=2)
            if process.is_alive():
                process.kill()
                process.join()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
