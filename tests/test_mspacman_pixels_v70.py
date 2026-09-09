from pathlib import Path

import numpy as np
import pytest

from ragnarok.evaluation.mspacman_pixels_v70 import MsPacmanPillPixels, PILL_COLOUR


def image():
    rgb = np.zeros((210, 160, 3), dtype=np.uint8)
    rgb[:] = (0, 28, 136)
    return rgb


def boxes(objects):
    return {(o.kind, o.box) for o in objects}


@pytest.mark.parametrize("height,kind", [(2, "pill"), (7, "power_pill")])
def test_fragments_need_a_complete_past_or_current_observation(height, kind):
    detector = MsPacmanPillPixels()
    fragment = image()
    fragment[40, 20] = PILL_COLOUR
    fragment[40 + height - 1, 23] = PILL_COLOUR
    earlier = detector.step(fragment)
    assert earlier == []  # A correctly sized bounding box alone cannot seed.
    complete = image()
    complete[40:40 + height, 20:24] = PILL_COLOUR
    expected = {(kind, (20, 40, 4, height))}
    assert boxes(detector.step(complete)) == expected
    assert boxes(detector.step(fragment)) == expected
    assert detector.step(image()) == []  # Remembered geometry is not presence.
    assert boxes(detector.step(fragment)) == expected
    assert earlier == []  # Later discovery cannot rewrite the earlier result.
    assert MsPacmanPillPixels().step(fragment) == []


def test_power_pill_reduced_to_two_rows_never_creates_a_small_pill():
    detector = MsPacmanPillPixels()
    full = image()
    full[80:87, 30:34] = PILL_COLOUR
    detector.step(full)
    cut = image()
    cut[80:82, 30:34] = PILL_COLOUR
    cut[100:102, 60:64] = PILL_COLOUR
    expected = {("power_pill", (30, 80, 4, 2)), ("pill", (60, 100, 4, 2))}
    assert boxes(detector.step(cut)) == expected
    assert boxes(detector.step(cut)) == expected


def test_later_large_support_replaces_an_earlier_contained_small_anchor():
    detector = MsPacmanPillPixels()
    small = image()
    small[50:52, 20:24] = PILL_COLOUR
    assert boxes(detector.step(small)) == {("pill", (20, 50, 4, 2))}
    large = image()
    large[48:55, 20:24] = PILL_COLOUR
    assert boxes(detector.step(large)) == {("power_pill", (20, 48, 4, 7))}
    assert boxes(detector.step(small)) == {("power_pill", (20, 50, 4, 2))}


def test_occluded_wall_fragments_cannot_seed_a_pill_location():
    detector = MsPacmanPillPixels()
    wall = image()
    wall[60:65, 70:102] = PILL_COLOUR
    assert detector.step(wall) == []
    cut = image()
    # Translated shape from the central-wall witness at development frame312.
    # Some remaining components are 2x2, but none is a complete 4x2 rectangle.
    pattern = ["#......###", ".##.##..##", "..#..#..##",
               ".##.##..##", "........##"]
    for y, row in enumerate(pattern, 60):
        for x, value in enumerate(row, 70):
            if value == "#":
                cut[y, x] = PILL_COLOUR
    assert detector.step(cut) == []
    assert detector.step(wall) == []


def test_hud_and_other_sprite_colours_do_not_seed_pills():
    rgb = image()
    rgb[174:181, 20:24] = PILL_COLOUR
    rgb[40:47, 40:44] = (210, 164, 74)
    rgb[60:62, 60:64] = (214, 214, 214)
    assert MsPacmanPillPixels().step(rgb) == []


def development_frame(number):
    folder = Path(__file__).resolve().parents[1] / "craft_v6_out/v70_mspacman_calibration/development"
    first = ((number - 1) // 256) * 256 + 1
    path = folder / f"frames_{first:05d}_{first + 255:05d}.npz"
    if not path.is_file():
        pytest.skip("Optional recorded development fixture is unavailable")
    with np.load(path, allow_pickle=False) as archive:
        return archive["rgb"][number - first].copy()


def test_recorded_power_fragment_at_747_keeps_its_previously_seen_category():
    detector = MsPacmanPillPixels()
    first = detector.step(development_frame(1))
    assert {o.box[0] for o in first if o.kind == "power_pill"} == {8}
    ninth = detector.step(development_frame(9))
    assert {o.box[0] for o in ninth if o.kind == "power_pill"} == {148}
    objects = detector.step(development_frame(747))
    # The top two rows are a complete disconnected 4x2 component; lower
    # fragments still extend the same power pill's visible union to height7.
    assert ("power_pill", (148, 146, 4, 7)) in boxes(objects)
    assert not any(o.kind == "pill" and 148 <= o.box[0] < 152 and 146 <= o.box[1] < 153
                   for o in objects)
