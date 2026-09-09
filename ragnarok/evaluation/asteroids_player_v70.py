"""Development-only Asteroids ship/explosion and missile render candidate.

RAM is external scoring input only. Empirical glyphs were inferred on every
archived 17511 frame, without ROM access. The RGB matcher takes pixels only.
This candidate is not a held-out calibration result.
"""
import numpy as np
from scipy import ndimage

from ragnarok.evaluation.asteroids_v70 import (
    ASTEROID_COLOURS, BLUE, PLAYER_COLOUR, SlottedLabel, x_position,
)

REVISION = 'asteroids_player_render_development_v1'
# Bit x is the pixel at coarse_x - 1 + x; every byte row lasts two lines.
SHIP_ROWS = (
    (8,8,28,28,62), (4,12,28,60,12), (2,12,60,24,8), (0,2,124,56,48),
    (32,56,63,56,32), (48,56,124,2,0), (8,24,60,12,2), (12,60,28,12,4),
    (124,56,56,16,16), (48,60,56,48,32), (16,24,60,48,64), (12,28,62,64,0),
    (4,28,252,28,4), (0,64,62,28,12), (64,48,60,24,16), (32,48,56,60,48),
)
EXPLOSION_ROWS = ((8,64,16,68,16), (16,8,1,32,69), (4,129,68,8,32))
SHIP_POINTER = (121,127,133,139,145,151,157,163,169,163,157,151,145,139,133,127)
EXPLOSION_POINTER = (175,181,187)


def _points(rows):
    return frozenset((x-1, 2*y+dy) for y, row in enumerate(rows)
                     for x in range(8) if row & (1 << x) for dy in range(2))


def _place(rows, x, y, fraction, wrap=False):
    points = set()
    for dx, dy in _points(rows):
        for shift in ((0,178) if wrap else (0,)):
            a, b = x+dx, y+dy+shift
            if 0 <= a < 160 and 18+fraction <= b < 195:
                points.add((a,b))
    return points


def player_ram_pixels(ram):
    """Predict red ship pixels, including explosion and both vertical pieces."""
    r = np.asarray(ram)
    if r.shape != (128,) or r.dtype != np.uint8:
        raise ValueError('Expected uint8 RAM[128]')
    if int(r[57]) & 1 or int(r[111]) == 224:
        return set()
    if int(r[110]) != 223:
        raise ValueError('Unknown player graphics bank')
    direction = int(r[60]) & 15
    pointer = int(r[109])
    logical_y = int(r[74]); logical_y -= 256 if logical_y >= 128 else 0
    fraction = int(r[107]) & 1
    if pointer in EXPLOSION_POINTER:
        rows = EXPLOSION_ROWS[EXPLOSION_POINTER.index(pointer)]
    else:
        expected = SHIP_POINTER[direction] - (5+logical_y if logical_y < 0 else 0)
        if pointer != expected:
            raise ValueError('Unknown player graphics pointer/direction')
        rows = SHIP_ROWS[direction]
    if not -5 <= logical_y <= 83:
        raise ValueError('Player vertical state outside observed development model')
    expected_y = logical_y + (89 if logical_y < 0 else 0)
    if int(r[111]) != expected_y:
        raise ValueError('Unknown logical/rendered player chronology')
    return _place(rows, x_position(r[73]), 18+2*logical_y+fraction,
                  fraction, wrap=logical_y < 0)


def _boxes(points):
    """One box per contiguous visible vertical band, not per explosion dot."""
    if not points:
        return []
    ys = sorted({y for x,y in points})
    bands = [[ys[0]]]
    for y in ys[1:]:
        if y == bands[-1][-1]+1: bands[-1].append(y)
        else: bands.append([y])
    result = []
    for band in bands:
        selected = [(x,y) for x,y in points if band[0] <= y <= band[-1]]
        xs = [x for x,y in selected]
        result.append((min(xs),band[0],max(xs)-min(xs)+1,len(band)))
    return result


def decode_player_missile_ram(ram):
    """Current RAM only; ship support can occult either missile colour."""
    ship = player_ram_pixels(ram)
    output = [SlottedLabel('player', b, f'player:{i}') for i,b in enumerate(_boxes(ship))]
    if int(ram[57]) & 1:
        return output
    for slot in range(2):
        raw_y = int(ram[86+slot])
        if raw_y == 224:
            continue
        if not 0 <= raw_y <= 88:
            raise ValueError('Unknown missile vertical state')
        x = (x_position(ram[83+slot])+1) % 160
        y = 17+2*raw_y
        points = {(x,y+dy) for dy in range(2) if 16 <= y+dy < 210} - ship
        for piece,b in enumerate(_boxes(points)):
            output.append(SlottedLabel('player_missile', b, f'missile:{slot}:{piece}'))
    return output


def _components(points):
    mask = np.zeros((210,160), dtype=bool)
    for x,y in points: mask[y,x] = True
    labels, _ = ndimage.label(mask, np.ones((3,3),dtype=np.uint8))
    return [{(int(x),int(y)) for y,x in np.argwhere(labels == k)}
            for k in range(1,int(labels.max())+1)]


def classify_red_pixels(points):
    """Independent shape matching: no RAM, history, counters or episode input.

    Full ship/explosion glyphs and their paired wrap pieces take priority over
    the nominal component split. Every remaining component is retained.
    """
    points = set(points)
    candidates = set()
    for rows in (*SHIP_ROWS, *EXPLOSION_ROWS):
        template = _points(rows)
        first = min(template, key=lambda p:(p[1],p[0]))
        for px,py in points:
            x,y = px-first[0],py-first[1]
            candidate = frozenset((x+dx,y+dy) for dx,dy in template)
            if candidate <= points:
                candidates.add(candidate)
        # The observed signed-y transition wraps exactly 178 lines. Pair the
        # top and bottom pieces before assigning any tiny piece to a missile.
        for logical_y in range(-5,0):
            for fraction in (0,1):
                template = _place(rows,0,18+2*logical_y+fraction,fraction,wrap=True)
                # Avoid clipping x at zero while building a translatable shape.
                template = {(x-8,y) for x,y in _place(rows,8,18+2*logical_y+fraction,fraction,wrap=True)}
                if not template: continue
                first = min(template,key=lambda p:(p[1],p[0]))
                for px,py in points:
                    if py != first[1]: continue
                    x = px-first[0]
                    candidate = frozenset((x+dx,dy) for dx,dy in template)
                    if candidate <= points:
                        candidates.add(candidate)
    players, remaining, matched = [], set(points), set()
    for candidate in sorted(candidates,key=lambda p:(-len(p),sorted(p))):
        if candidate <= remaining:
            players.extend(_boxes(candidate))
            remaining -= candidate
            matched |= candidate
    missiles, fallback = [], []
    for component in _components(remaining):
        b = _boxes(component)[0]
        if b[2] <= 2 and b[3] <= 3:
            missiles.append(b)
        else:
            players.append(b)
            fallback.append(b)
    return players, missiles, fallback


def screen_player_missile_objects(rgb):
    """RGB-only actor checker; blue asteroid-phase ambiguity remains explicit."""
    if rgb.shape != (210,160,3) or rgb.dtype != np.uint8:
        raise ValueError('Expected uint8 RGB[210,160,3]')
    red_mask = np.all(rgb == PLAYER_COLOUR,axis=2); red_mask[:16] = False
    y,x = np.nonzero(red_mask)
    player,missile,_ = classify_red_pixels(zip(map(int,x),map(int,y)))
    output = [SlottedLabel('player',b,f'pixel:player:{i}') for i,b in enumerate(player)]
    blue_mask = np.all(rgb == BLUE,axis=2); blue_mask[:16] = False
    y,x = np.nonzero(blue_mask)
    blues = _components(set(zip(map(int,x),map(int,y))))
    other_asteroid_colour = any(np.any(np.all(rgb[16:] == c,axis=2))
                               for c in ASTEROID_COLOURS if c != BLUE)
    asteroid_phase = other_asteroid_colour or any(
        _boxes(c)[0][2] > 2 or _boxes(c)[0][3] > 3 for c in blues)
    if not asteroid_phase:
        missile.extend(_boxes(c)[0] for c in blues)
    output.extend(SlottedLabel('player_missile',b,f'pixel:missile:{i}') for i,b in enumerate(missile))
    return output
