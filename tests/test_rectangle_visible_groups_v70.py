"""Judge-only equivalence to the frozen raster, without native data or timing."""
import numpy as np
import pytest

from ragnarok.evaluation.scene_perception_v70 import SceneObject
from ragnarok.evaluation.rectangle_visible_groups_v70 import rectangle_visible_groups
from ragnarok.evaluation.visible_temporal_groups_v70 import REGISTRY, group_visible_sources


def test_edges_join_but_corners_do_not_and_memberships_are_exact():
    objects = [SceneObject('a', 'pill', (1, 1, 2, 2), False),
               SceneObject('b', 'pill', (3, 1, 2, 2), False),
               SceneObject('c', 'pill', (3, 3, 2, 2), False),
               SceneObject('corner', 'pill', (5, 5, 1, 1), False),
               SceneObject('overlap', 'pill', (2, 2, 2, 2), False)]
    groups = rectangle_visible_groups(objects, game='ms_pacman')
    assert groups == group_visible_sources(objects, game='ms_pacman')
    assert [group.members for group in groups] == [frozenset(('a', 'b', 'c', 'overlap')),
                                                  frozenset(('corner',))]
    assert [group.label.box for group in groups] == [(1, 1, 4, 4), (5, 5, 1, 1)]


def test_raster_order_uses_first_occupied_pixel_not_bounding_box_left():
    # Component A first appears at (8,1), although its box reaches left to x0
    # on row3. Separate B starts at (4,1), so raster order is B then A.
    objects = [SceneObject('a-top', 'pill', (8, 1, 1, 3), False),
               SceneObject('a-bottom', 'pill', (0, 3, 9, 1), False),
               SceneObject('b', 'pill', (4, 1, 1, 1), False)]
    groups = rectangle_visible_groups(objects, game='ms_pacman')
    assert groups == group_visible_sources(objects, game='ms_pacman')
    assert [group.members for group in groups] == [frozenset(('b',)), frozenset(('a-top', 'a-bottom'))]


def test_kind_order_duplicates_and_repeated_external_ids_are_preserved():
    objects = [SceneObject('same', 'pill', (1, 1, 2, 2), False),
               SceneObject('same', 'pill', (8, 1, 2, 2), False),
               SceneObject('ghost', 'ghost', (1, 1, 2, 2)),
               SceneObject('player', 'player', (1, 1, 2, 2))]
    objects += objects[:2]
    snapshot = list(objects)
    groups = rectangle_visible_groups(iter(objects), game='ms_pacman')
    assert groups == group_visible_sources(objects, game='ms_pacman')
    assert [group.label.kind for group in groups] == ['player', 'ghost', 'pill', 'pill']
    assert objects == snapshot
    assert rectangle_visible_groups([], game='asteroids') == []


@pytest.mark.parametrize('game', ['ms_pacman', 'asteroids'])
def test_seeded_rectangles_match_raster_exactly_including_input_permutations(game):
    rng = np.random.default_rng(19764)
    roi, kinds, _ = REGISTRY[game]
    for case in range(64):
        objects = []
        for index in range(int(rng.integers(1, 65))):
            # Clustered integer rectangles make edge/corner/overlap cases common.
            x, y = int(rng.integers(0, 24)), roi[1] + int(rng.integers(0, 24))
            w, h = int(rng.integers(1, 10)), int(rng.integers(1, 10))
            kind = kinds[int(rng.integers(len(kinds)))]
            objects.append(SceneObject(f'{case}:{index}', kind, (x, y, w, h), bool(index % 2)))
        objects += objects[:3]
        rng.shuffle(objects)
        assert rectangle_visible_groups(objects, game=game) == group_visible_sources(objects, game=game)


def test_complete_validation_rejects_bad_scene_rectangles():
    valid = SceneObject('valid', 'pill', (0, 0, 1, 1), False)
    invalid = [SceneObject('', 'pill', (0, 0, 1, 1)),
               SceneObject('bad', 'unknown', (0, 0, 1, 1)),
               SceneObject('bad', 'pill', (0, 0, 0, 1)),
               SceneObject('bad', 'pill', (-1, 0, 1, 1)),
               SceneObject('bad', 'pill', (159, 0, 2, 1)),
               SceneObject('bad', 'pill', (0., 0, 1, 1)), object()]
    for obj in invalid:
        for fn in (rectangle_visible_groups, group_visible_sources):
            with pytest.raises(ValueError):
                fn([valid, obj], game='ms_pacman')
