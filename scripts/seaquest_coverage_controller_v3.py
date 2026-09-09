"""Third prospective Seaquest fixture policy: RGB only, experimenter only.

Hold a central horizontal position and face incoming enemies while divers swim
into the player. This is a hand-written game-specific collection aid, never an
agent skill, teacher, reference extractor, or source of advanced entry states.
"""
import numpy as np
from scipy import ndimage

ACTIONS = {(0, 0): 1, (0, -1): 10, (1, 0): 11, (-1, 0): 12, (0, 1): 13,
           (1, -1): 14, (-1, -1): 15, (1, 1): 16, (-1, 1): 17}
ENEMY_COLOURS = ((92, 186, 92), (198, 108, 58), (160, 171, 79),
                 (72, 160, 72), (198, 89, 179), (170, 170, 170))


def components(mask, x=0, y=0):
    labels, _ = ndimage.label(mask, np.ones((3, 3), dtype=np.uint8))
    return [(x+xs.start, y+ys.start, xs.stop-xs.start, ys.stop-ys.start)
            for ys, xs in ndimage.find_objects(labels)]


def direction(delta, tolerance=3):
    return 1 if delta > tolerance else -1 if delta < -tolerance else 0


class CoverageController:
    def __init__(self):
        self.phase = "dive"

    def action(self, rgb):
        if rgb.shape != (210, 160, 3) or rgb.dtype != np.uint8:
            raise ValueError("The coverage controller accepts raw RGB only")
        area = rgb[33:165]
        friendly = np.all(area == (187, 187, 53), axis=2) | np.all(area == (236, 236, 236), axis=2)
        boxes = [b for b in components(friendly, y=33) if b[3] >= 2]
        if not boxes:
            self.phase = "dive"
            return 1
        x, y, w, h = max(boxes, key=lambda b: b[2]*b[3])
        if w > 16:
            # A contacting one-row missile must not move the estimated centre
            # or reverse the inferred facing of an otherwise stationary body.
            body = friendly[y-33:y-33+h, x:x+w].copy()
            for row in range(h):
                occupied = np.flatnonzero(body[row])
                if len(occupied) and occupied[-1]-occupied[0]+1 > 16:
                    body[row] = False
            yy, xx = np.where(body)
            if len(xx) and xx.max()-xx.min()+1 <= 16 and yy.max()-yy.min()+1 >= 6:
                x, w = x+int(xx.min()), int(xx.max()-xx.min()+1)
        px, py = x+w/2, y+h/2
        tips = np.flatnonzero(friendly[y-33, x:x+w])
        facing = direction(x+float(tips.mean())-px, 0) if len(tips) else 1
        oxygen = int(np.count_nonzero(np.all(rgb[172, 49:112] == (214, 214, 214), axis=1)))
        collected = sum(bool(np.any(np.all(rgb[178:187, 58+8*i:66+8*i] == (24, 26, 167), axis=2)))
                        for i in range(6))
        if y <= 46 and oxygen < 60:
            self.phase = "refill"
        if py > 58 and (collected >= 6 or oxygen <= 20):
            self.phase = "ascend"
        if self.phase == "refill":
            if oxygen >= 60 and collected < 6:
                self.phase = "dive"
            else:
                return 1
        if self.phase == "ascend":
            if y <= 46:
                self.phase = "refill"
                return 1
            patrol = components(np.all(rgb[40:57] == (170, 170, 170), axis=2), y=40)
            target_x = (32 if patrol[0][0]+patrol[0][2]/2 > 80 else 128) if patrol else px
            return ACTIONS[(direction(target_x-px) if py < 85 else 0, -1)]
        divers = []
        for top in (70, 94, 118, 142):
            mask = np.all(rgb[top:top+9] == (66, 72, 200), axis=2)
            divers.extend(b for b in components(mask, y=top) if b[2] > 2 and b[3] >= 6)
        if divers:
            target = min(divers, key=lambda b: abs(b[1]+b[3]/2-py)+.2*abs(b[0]+b[2]/2-px))
            target_y = target[1]+target[3]/2
        else:
            target_y = min(146, max(74, py))
        enemy_mask = np.zeros((104, 160), dtype=bool)
        for colour in ENEMY_COLOURS:
            enemy_mask |= np.all(rgb[61:165] == colour, axis=2)
        enemies = components(enemy_mask, y=61)
        dy = direction(target_y-py)
        # Do not cross an occupied lane immediately beside an enemy. Holding
        # position permits continuous fire to clear it first.
        crossing = [b for b in enemies if abs(b[0]+b[2]/2-px) < 24 and
                    min(py, target_y)-9 <= b[1]+b[3]/2 <= max(py, target_y)+9]
        if crossing:
            dy = 0
        nearby = [b for b in enemies if abs(b[1]+b[3]/2-py) <= 12]
        target_enemy = min(nearby, key=lambda b: abs(b[0]+b[2]/2-px)) if nearby else None
        if target_enemy is not None:
            required = direction(target_enemy[0]+target_enemy[2]/2-px, 0)
            dx = required if required != facing else 0
        else:
            dx = direction(80-px) if px < 48 or px > 112 else 0
        return ACTIONS[(dx, dy)]
