"""Exact external filled-rectangle components without rasterizing the ROI.

This is a judge-only candidate. No learner, collector or native worker uses it
implicitly. Four-connectivity and component scan order follow the sealed raster.
"""
from ragnarok.evaluation.pong_v70 import ObjectLabel
from ragnarok.evaluation.scene_perception_v70 import SceneObject
from ragnarok.evaluation.visible_temporal_groups_v70 import REGISTRY, VisibleEnvelopeGroup


def rectangle_visible_groups(objects, *, game):
    roi, kinds, _ = REGISTRY[game]
    objects = list(objects)
    # Same complete validation before grouping as the sealed raster definition.
    for obj in objects:
        if not isinstance(obj, SceneObject) or obj.kind not in kinds or not obj.instance_id:
            raise ValueError('Declared external source scene objects required')
        x, y, w, h = obj.box
        if (any(type(v) is not int for v in obj.box) or not roi[0] <= x < x + w <= roi[2]
                or not roi[1] <= y < y + h <= roi[3]):
            raise ValueError('Source rectangle outside its calibrated integer ROI')
    groups = []
    for kind in kinds:
        own = list(dict.fromkeys(obj for obj in objects if obj.kind == kind))
        rectangles = [(x, y, x + w, y + h) for x, y, w, h in (obj.box for obj in own)]
        parent = list(range(len(own)))

        def root(index):
            while parent[index] != index:
                parent[index] = parent[parent[index]]
                index = parent[index]
            return index

        active = []
        for index in sorted(range(len(own)), key=lambda i: (rectangles[i][1], rectangles[i][0])):
            left, top, right, bottom = rectangles[index]
            # Equal boundary y is retained: horizontal edges can be adjacent.
            active = [old for old in active if rectangles[old][3] >= top]
            for old in active:
                a, b, c, d = rectangles[old]
                overlap_x = min(right, c) - max(left, a)
                overlap_y = min(bottom, d) - max(top, b)
                # Positive overlap on one axis plus overlap/contact on the other
                # is exactly shared pixels OR a four-neighbour edge. A corner
                # with both overlaps zero is deliberately not connected.
                if ((overlap_x > 0 and overlap_y >= 0)
                        or (overlap_x >= 0 and overlap_y > 0)):
                    parent[root(index)] = root(old)
            active.append(index)
        components = {}
        for index, obj in enumerate(own):
            components.setdefault(root(index), []).append(index)
        ordered = []
        for indices in components.values():
            # Bounding-box left is insufficient for raster order: the leftmost
            # rectangle may begin below the component's first occupied row.
            first_pixel = min((rectangles[i][1], rectangles[i][0]) for i in indices)
            left = min(rectangles[i][0] for i in indices)
            top = min(rectangles[i][1] for i in indices)
            right = max(rectangles[i][2] for i in indices)
            bottom = max(rectangles[i][3] for i in indices)
            group = VisibleEnvelopeGroup(
                ObjectLabel(kind, (left, top, right - left, bottom - top)),
                frozenset(own[i].instance_id for i in indices),
            )
            ordered.append((first_pixel, group))
        groups.extend(group for _, group in sorted(ordered, key=lambda item: item[0]))
    return groups
