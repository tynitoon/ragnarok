from pathlib import Path

import numpy as np
import pytest

from scripts.mspacman_coverage_controller_v70 import CoverageController
from ragnarok.evaluation.pong_v70 import ObjectLabel
from ragnarok.evaluation import mspacman_ghosts_v70 as ghosts


class Targets:
    def step(self,rgb):
        return [ObjectLabel("pill",(48,49,4,2)),ObjectLabel("pill",(8,99,4,2))]


def test_virtual_start_cannot_skip_an_intermediate_ghost(monkeypatch):
    controller=CoverageController();controller.steps=8;controller.detector=Targets()
    controller.walls=np.zeros((171,160),dtype=bool)
    controller.graph={(10,50):[(20,50),(10,100)],(20,50):[(10,50),(30,50)],
                      (30,50):[(20,50),(50,50)],(50,50):[(30,50)],(10,100):[(10,50)]}
    monkeypatch.setattr(ghosts,"screen_ghosts",lambda rgb:[ObjectLabel("ghost",(16,45,8,10))])
    rgb=np.zeros((210,160,3),dtype=np.uint8);rgb[49:51,9:11]=(210,164,74)
    assert controller.action(rgb)==5  # Down50, rather than right40 through the ghost.


def recorded(number):
    folder=Path(__file__).resolve().parents[1]/"craft_v6_out/v70_mspacman_calibration/development"
    first=((number-1)//256)*256+1
    path=folder/f"frames_{first:05d}_{first+255:05d}.npz"
    if not path.is_file():pytest.skip("Recorded development fixture unavailable")
    with np.load(path,allow_pickle=False) as data:
        return data["rgb"][number-first].copy()


def test_recorded_stationary_mouth_animation_triggers_escape():
    controller=CoverageController();controller.steps=8
    controller.action(recorded(1))  # Learn the maze from past RGB only.
    for number in range(2881,2962,2):
        controller.action(recorded(number))
    assert controller.stationary>=12
    assert len(controller.recent_positions)==13


def test_sliding_window_does_not_mistake_cumulative_movement_for_jitter(monkeypatch):
    controller=CoverageController();controller.steps=8;controller.detector=Targets()
    controller.walls=np.zeros((171,160),dtype=bool)
    controller.graph={(10,50):[(50,50)],(50,50):[(10,50)]}
    monkeypatch.setattr(ghosts,"screen_ghosts",lambda rgb:[])
    for x in range(10,40):
        rgb=np.zeros((210,160,3),dtype=np.uint8);rgb[49:51,x-1:x+1]=(210,164,74)
        controller.action(rgb)
        assert controller.stationary<12
