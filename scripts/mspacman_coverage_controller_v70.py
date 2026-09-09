"""Prospective RGB maze walker for scorer-only coverage, never an agent skill.

The controller seeks currently visible pills over a maze inferred from pink
wall components. It has no RAM, reward, emulator, snapshot or agent endpoint.
The first observed maze is its declared scope; all later failures are retained.
"""
import heapq

import numpy as np
from scipy import ndimage

from ragnarok.evaluation.mspacman_pixels_v70 import MsPacmanPillPixels, PILL_COLOUR


class CoverageController:
    def __init__(self):
        self.detector = MsPacmanPillPixels()
        self.graph = None
        self.walls = None
        self.visits = {}
        self.recent_positions = []
        self.stationary = 0
        self.steps = 0
        self.last_action = 1

    @staticmethod
    def _clear(walls, a, b):
        x,y = a
        xx,yy = b
        if x == xx:
            return not walls[min(y,yy):max(y,yy)+1,x].any()
        if y == yy:
            return not walls[y,min(x,xx):max(x,xx)+1].any()
        return False

    def _build(self, rgb, objects):
        pink = np.all(rgb[:171] == PILL_COLOUR, axis=2)
        labels,_ = ndimage.label(pink, np.ones((3,3),dtype=np.uint8))
        walls = np.zeros_like(pink)
        for label,sl in enumerate(ndimage.find_objects(labels),1):
            h,w = sl[0].stop-sl[0].start,sl[1].stop-sl[1].start
            if w > 4 or h > 7:
                walls[sl] |= labels[sl] == label
        full = [o.box for o in objects if o.kind == "pill" and o.box[2:] == (4,2)]
        xs,ys = sorted({x+2 for x,y,w,h in full}),sorted({y+1 for x,y,w,h in full})
        if len(xs) < 10 or len(ys) < 10:
            return
        nodes = {(x,y) for x in xs for y in ys if not walls[y,x]}
        graph = {p:[] for p in nodes}
        for y in ys:
            for x,xx in zip(xs,xs[1:]):
                if (x,y) in nodes and (xx,y) in nodes and self._clear(walls,(x,y),(xx,y)):
                    graph[(x,y)].append((xx,y));graph[(xx,y)].append((x,y))
        for x in xs:
            for y,yy in zip(ys,ys[1:]):
                if (x,y) in nodes and (x,yy) in nodes and self._clear(walls,(x,y),(x,yy)):
                    graph[(x,y)].append((x,yy));graph[(x,yy)].append((x,y))
        self.graph,self.walls = graph,walls

    def action(self, rgb):
        # Import only the pixel ghost checker, which has no ROM or RAM access.
        from ragnarok.evaluation.mspacman_ghosts_v70 import screen_ghosts
        rgb = np.asarray(rgb)
        if rgb.shape != (210,160,3) or rgb.dtype != np.uint8:
            raise ValueError("RGB must be uint8[210,160,3]")
        self.steps += 1
        objects = self.detector.step(rgb)
        if self.graph is None:
            self._build(rgb,objects)
        if self.steps <= 8:
            return 1
        ys,xs = np.where(np.all(rgb[:171] == (210,164,74),axis=2))
        if not len(xs) or not self.graph:
            return 1
        position = (int(round((int(xs.min())+int(xs.max())+1)/2)),
                    int(round((int(ys.min())+int(ys.max())+1)/2)))
        self.recent_positions = (self.recent_positions+[position])[-13:]
        span_x=max(p[0] for p in self.recent_positions)-min(p[0] for p in self.recent_positions)
        span_y=max(p[1] for p in self.recent_positions)-min(p[1] for p in self.recent_positions)
        self.stationary = len(self.recent_positions)-1 if max(span_x,span_y)<=2 else 0
        # Snap only the perpendicular axis by <=2 pixels, accounting for mouth
        # animation. A virtual start can connect along an unobstructed corridor.
        starts = []
        for node in self.graph:
            dx,dy = abs(node[0]-position[0]),abs(node[1]-position[1])
            if dx <= 2 and self._clear(self.walls,(node[0],position[1]),node):
                starts.append((dx+dy,node))
            elif dy <= 2 and self._clear(self.walls,(position[0],node[1]),node):
                starts.append((dx+dy,node))
        if not starts:
            return (2,3,5,4)[(self.steps//8)%4]
        # Connect the virtual start only to its immediately adjacent nodes.
        # Seeding every collinear node would bypass penalties along a corridor.
        nearby=[item for item in starts if item[0]<=2]
        if nearby:
            starts=[min(nearby)]
        else:
            nearest={}
            for item in starts:
                _,node=item
                dx,dy=node[0]-position[0],node[1]-position[1]
                direction=("right" if dx>0 else "left") if abs(dx)>2 else ("down" if dy>0 else "up")
                if direction not in nearest or item<nearest[direction]:
                    nearest[direction]=item
            starts=list(nearest.values())
        visible_targets = {(o.box[0]+o.box[2]/2,o.box[1]+o.box[3]/2) for o in objects}
        targets = {n for n in self.graph if any(abs(n[0]-x)<=2 and abs(n[1]-y)<=4
                                                for x,y in visible_targets)}
        ghosts = [(o.box[0]+o.box[2]/2,o.box[1]+o.box[3]/2) for o in screen_ghosts(rgb)]
        def hazard(node):
            if not ghosts:
                return 0
            distance = min(abs(node[0]-x)+abs(node[1]-y) for x,y in ghosts)
            return 1000 if distance < 12 else 100 if distance < 24 else 0
        queue=[];best={};first={}
        for distance,node in starts:
            cost=distance+hazard(node)
            if cost < best.get(node,float("inf")):
                best[node]=cost;first[node]=node;heapq.heappush(queue,(cost,node))
        goal=None
        while queue:
            cost,node=heapq.heappop(queue)
            if cost != best[node]:
                continue
            if node in targets and abs(node[0]-position[0])+abs(node[1]-position[1])>2:
                goal=node;break
            for nxt in self.graph[node]:
                candidate=cost+abs(nxt[0]-node[0])+abs(nxt[1]-node[1])+hazard(nxt)+min(self.visits.get(nxt,0),30)
                if candidate < best.get(nxt,float("inf")):
                    best[nxt]=candidate
                    first[nxt]=nxt if abs(first[node][0]-position[0])+abs(first[node][1]-position[1])<=2 else first[node]
                    heapq.heappush(queue,(candidate,nxt))
        if goal is None:
            goal=min(starts)[1]
            target=goal
        else:
            target=first[goal]
        self.visits[target]=self.visits.get(target,0)+1
        dx,dy=target[0]-position[0],target[1]-position[1]
        if abs(dx)>2 and abs(dy)<=2:
            action=3 if dx>0 else 4
        elif abs(dy)>2 and abs(dx)<=2:
            action=5 if dy>0 else 2
        else:
            action=self.last_action
        if self.stationary>=12:
            # Deterministic escape; never restart or discard an unsuccessful run.
            action=(2,3,5,4,1)[(self.steps//4)%5]
        self.last_action=action
        return action
