"""DeviceVecTechTree — a DATA-DRIVEN, PROCEDURAL tech-tree gridworld (v10).

A generalization of CraftWorld whose recipe graph is a SPEC, not hardcoded,
so we can generate RANDOM tech trees and test whether the developmental /
model-based agent works on worlds neither it nor we designed.

Items are RESOURCES (collected from a grid cell type, optionally gated by a
tool item) or CRAFTS (consume input items + require tool items). A generator
samples a random valid DAG. Batched + device-resident, same interface as the
other DeviceVec* envs.

Kept separate from craft_world.py — existing CraftWorld results are untouched.
"""

import math
import numpy as np
import torch

from ragnarok.infrastructure.device import DEVICE

WALL_PAD = 99          # sentinel cell id for out-of-bounds in the egocentric view


def gen_tree(seed, n_items=14, n_base_res=2, p_resource=0.35, max_inputs=3):
    """Sample a random valid tech-tree DAG. Items built in topological order
    (each item's prerequisites are a random subset of earlier items)."""
    rng = np.random.default_rng(seed)
    kind, cell, tool, inputs, tools = [], [], [], [], []
    for i in range(n_base_res):                       # base resources (no tool)
        kind.append("R"); cell.append(i + 1); tool.append(-1)
        inputs.append({}); tools.append([])
    next_cell = n_base_res + 1
    for i in range(n_base_res, n_items):
        existing = list(range(i))
        crafts = [j for j in existing if kind[j] == "C"]
        if crafts and rng.random() < p_resource:      # tool-gated resource
            kind.append("R"); cell.append(next_cell); next_cell += 1
            tool.append(int(rng.choice(crafts))); inputs.append({}); tools.append([])
        else:                                          # craft item
            kind.append("C"); cell.append(-1); tool.append(-1)
            k = int(rng.integers(1, min(max_inputs, len(existing)) + 1))
            ins = {int(x): 1 for x in rng.choice(existing, size=k, replace=False)}
            inputs.append(ins)
            tl = [int(rng.choice(crafts))] if (crafts and rng.random() < 0.4) else []
            tools.append(tl)
    # depth of each item (longest prerequisite chain) -> target = deepest
    depth = [0] * n_items
    for i in range(n_items):
        prereq = set(inputs[i]) | set(tools[i])
        if tool[i] >= 0:
            prereq.add(tool[i])
        depth[i] = 1 + max((depth[j] for j in prereq), default=-1)
    target = int(np.argmax(depth))
    # ground-truth direct item-preconditions (for scoring rule recovery)
    pre = []
    for i in range(n_items):
        p = set(inputs[i]) | set(tools[i])
        if tool[i] >= 0:
            p.add(tool[i])
        pre.append(p)
    craft_actions = [i for i in range(n_items) if kind[i] == "C"]  # -> actions 5..
    return dict(n_items=n_items, n_cells=next_cell, kind=kind, cell=cell,
                tool=tool, inputs=inputs, tools=tools, depth=depth,
                target=target, true_pre=pre, craft_actions=craft_actions)


class DeviceVecTechTree:
    is_discrete = True

    def __init__(self, num_envs, spec, grid=11, view=5, max_steps=300,
                 n_resource=5, goal=None, grant=None, seed=0,
                 max_cells=None, nav_goal=None):
        """max_cells: pad the cell-type one-hot to this fixed width (so obs
        dim is identical across trees — needed for a tree-agnostic skill).
        nav_goal: navigation-skill mode. None=normal; 'random'=resample a
        target resource cell-type per env each reset; int=fixed target. In nav
        mode obs = egocentric(max_cells) + target-cell-type one-hot, and reward
        is +1 (terminate) when the item of the target cell-type is collected."""
        self.num_envs = num_envs
        self.spec = spec
        self.G = grid
        self.P = view
        self.half = view // 2
        self.max_steps = max_steps
        self.n_resource = n_resource          # cells of EACH resource type
        self.n_items = spec["n_items"]
        self.n_cells = spec["n_cells"]        # 0=empty, 1..(n_cells-1)=resources
        self.n_cell_types = max_cells if max_cells else (self.n_cells + 1)
        self.goal_idx = goal
        self.goal_conditioned = goal is not None
        self._nav = nav_goal                  # None | 'random' | int
        self.nav_mode = nav_goal is not None
        self.action_dim = 5 + len(spec["craft_actions"])
        if self.nav_mode:
            self.obs_dim = self.P * self.P * self.n_cell_types + self.n_cell_types
        else:
            self.obs_dim = self.P * self.P * self.n_cell_types + self.n_items \
                + (self.n_items if self.goal_conditioned else 0)
        self._grant = (None if grant is None else
                       torch.as_tensor(grant, dtype=torch.long, device=DEVICE))

        # precompute lookup tensors -------------------------------------
        # cell type (1..n_cells-1) -> resource item id (-1 if none) and tool id
        c2i = torch.full((self.n_cells,), -1, dtype=torch.long, device=DEVICE)
        c2t = torch.full((self.n_cells,), -1, dtype=torch.long, device=DEVICE)
        for it in range(self.n_items):
            if spec["kind"][it] == "R":
                c2i[spec["cell"][it]] = it
                c2t[spec["cell"][it]] = spec["tool"][it]
        self.cell2item, self.cell2tool = c2i, c2t
        self.resource_cells = [spec["cell"][it] for it in range(self.n_items)
                               if spec["kind"][it] == "R"]
        # per craft action: input-consume vector, tool-required mask, out item
        self.craft_out = spec["craft_actions"]
        ncraft = len(self.craft_out)
        self.craft_in = torch.zeros(ncraft, self.n_items, dtype=torch.long, device=DEVICE)
        self.craft_tool = torch.zeros(ncraft, self.n_items, dtype=torch.bool, device=DEVICE)
        for k, it in enumerate(self.craft_out):
            for item, cnt in spec["inputs"][it].items():
                self.craft_in[k, item] = cnt
            for t in spec["tools"][it]:
                self.craft_tool[k, t] = True
        self.craft_out_t = torch.tensor(self.craft_out, dtype=torch.long, device=DEVICE)
        self._gen = torch.Generator(device=DEVICE); self._gen.manual_seed(seed)
        self.reset()

    # ---- generation -----------------------------------------------------
    def _rand_cells(self, n):
        G = self.G
        grid = torch.zeros(n, G, G, dtype=torch.long, device=DEVICE)
        scores = torch.rand(n, G * G, generator=self._gen, device=DEVICE)
        order = scores.argsort(dim=-1)
        slot = 0
        env_ar = torch.arange(n, device=DEVICE).unsqueeze(-1)
        for rc in self.resource_cells:
            idx = order[:, slot:slot + self.n_resource]
            grid[env_ar, idx // G, idx % G] = rc
            slot += self.n_resource
        a = order[:, slot]
        return grid, torch.stack([a // G, a % G], dim=-1)

    def _sample_nav(self, n):
        rc = torch.tensor(self.resource_cells, device=DEVICE)
        if self._nav == "random":
            return rc[torch.randint(len(rc), (n,), generator=self._gen, device=DEVICE)]
        return torch.full((n,), int(self._nav), dtype=torch.long, device=DEVICE)

    def _nav_dist(self):
        """Manhattan distance from each agent to the nearest cell of its
        target type (for dense nav-reward shaping)."""
        N, G = self.num_envs, self.G
        mask = self.grid == self.nav_vec.view(N, 1, 1)
        rows = torch.arange(G, device=DEVICE).view(1, G, 1)
        cols = torch.arange(G, device=DEVICE).view(1, 1, G)
        pr, pc = self.pos[:, 0].view(N, 1, 1), self.pos[:, 1].view(N, 1, 1)
        d = torch.where(mask, (rows - pr).abs() + (cols - pc).abs(),
                        torch.full((N, G, G), G * G, device=DEVICE))
        return d.view(N, -1).min(-1).values.float()

    def reset(self):
        self.grid, self.pos = self._rand_cells(self.num_envs)
        self.inv = torch.zeros(self.num_envs, self.n_items, dtype=torch.long,
                               device=DEVICE)
        if self._grant is not None:
            self.inv[:] = self._grant
        self.unlocked = torch.zeros(self.num_envs, self.n_items, dtype=torch.bool,
                                    device=DEVICE)
        self.steps = torch.zeros(self.num_envs, device=DEVICE)
        if self.nav_mode:
            self.nav_vec = self._sample_nav(self.num_envs)
            self.nav_prevdist = self._nav_dist()
        self._set_state()
        return self.state

    def _egocentric(self):
        N, G, P, h = self.num_envs, self.G, self.P, self.half
        padded = torch.full((N, G + 2 * h, G + 2 * h), self.n_cells,  # WALL = n_cells
                            dtype=torch.long, device=DEVICE)
        padded[:, h:h + G, h:h + G] = self.grid
        env_ar = torch.arange(N, device=DEVICE)
        out = torch.empty(N, P, P, dtype=torch.long, device=DEVICE)
        for dr in range(P):
            for dc in range(P):
                out[:, dr, dc] = padded[env_ar, self.pos[:, 0] + dr, self.pos[:, 1] + dc]
        return out

    def _set_state(self):
        ego = self._egocentric()
        oh = torch.nn.functional.one_hot(ego, self.n_cell_types).float()
        if self.nav_mode:
            g = torch.nn.functional.one_hot(self.nav_vec, self.n_cell_types).float()
            self.state = torch.cat([oh.reshape(self.num_envs, -1), g], dim=-1)
            return
        parts = [oh.reshape(self.num_envs, -1),
                 self.inv.float().clamp(max=5.0) / 5.0]
        if self.goal_conditioned:
            g = torch.zeros(self.num_envs, self.n_items, device=DEVICE)
            g[:, self.goal_idx] = 1.0
            parts.append(g)
        self.state = torch.cat(parts, dim=-1)

    def _fire(self, mask_item_pairs, reward):
        pass

    def step(self, action):
        a = action.reshape(self.num_envs).long()
        N = self.num_envs
        reward = torch.zeros(N, device=DEVICE)
        env_ar = torch.arange(N, device=DEVICE)
        goal_was = (self.unlocked[:, self.goal_idx].clone()
                    if self.goal_conditioned else None)
        if self.nav_mode:
            nav_item = self.cell2item[self.nav_vec]
            nav_before = self.inv[env_ar, nav_item].clone()

        # movement (0-3)
        deltas = torch.tensor([[-1, 0], [1, 0], [0, -1], [0, 1]], device=DEVICE)
        mv = a < 4
        if bool(mv.any()):
            np_ = (self.pos + deltas[a.clamp(max=3)]).clamp(0, self.G - 1)
            self.pos = torch.where(mv.unsqueeze(-1), np_, self.pos)

        # collect (4): resolve current cell -> resource item if tool met
        coll = a == 4
        cur_cell = self.grid[env_ar, self.pos[:, 0], self.pos[:, 1]]      # (N,)
        valid_cell = (cur_cell >= 1) & (cur_cell < self.n_cells)
        item_here = torch.where(valid_cell, self.cell2item[cur_cell.clamp(max=self.n_cells - 1)],
                                torch.full_like(cur_cell, -1))
        tool_here = torch.where(valid_cell, self.cell2tool[cur_cell.clamp(max=self.n_cells - 1)],
                                torch.full_like(cur_cell, -1))
        has_tool = (tool_here < 0) | (self.inv[env_ar, tool_here.clamp(min=0)] >= 1)
        do_coll = coll & (item_here >= 0) & has_tool
        if bool(do_coll.any()):
            idx = do_coll.nonzero(as_tuple=True)[0]
            items = item_here[idx]
            self.inv[idx, items] += 1
            newly = ~self.unlocked[idx, items]
            self.unlocked[idx, items] = True
            reward[idx] += newly.float()

        # craft (5..): one action per craft item
        for k, out_item in enumerate(self.craft_out):
            am = a == (5 + k)
            if not bool(am.any()):
                continue
            need = self.craft_in[k]                       # (n_items,)
            ok = am & (self.inv >= need).all(dim=-1)
            if bool(self.craft_tool[k].any()):
                ok = ok & (self.inv[:, self.craft_tool[k]] >= 1).all(dim=-1)
            if bool(ok.any()):
                idx = ok.nonzero(as_tuple=True)[0]
                self.inv[idx] -= need
                self.inv[idx, out_item] += 1
                newly = ~self.unlocked[idx, out_item]
                self.unlocked[idx, out_item] = True
                reward[idx] += newly.float()

        self.steps += 1
        if self.nav_mode:
            got = self.inv[env_ar, nav_item] > nav_before
            cur = self._nav_dist()
            reward = (self.nav_prevdist - cur) * 0.1 + got.float()  # dense shaping
            terminated = got
        elif self.goal_conditioned:
            terminated = self.unlocked[:, self.goal_idx].clone()
            reward = (terminated & (~goal_was)).float()
        else:
            terminated = torch.zeros(N, dtype=torch.bool, device=DEVICE)
        truncated = self.steps >= self.max_steps
        done = terminated | truncated
        if bool(done.any()):
            self._reset_done(done)
        if self.nav_mode:
            self.nav_prevdist = self._nav_dist()   # refresh for next step (post-reset)
        self._set_state()
        return self.state, reward, terminated, truncated, done

    def _reset_done(self, done):
        n = int(done.sum().item())
        if n == 0:
            return
        grid, pos = self._rand_cells(n)
        self.grid[done] = grid
        self.pos[done] = pos
        self.inv[done] = 0 if self._grant is None else self._grant
        self.unlocked[done] = False
        self.steps[done] = 0
        if self.nav_mode:
            self.nav_vec[done] = self._sample_nav(n)
