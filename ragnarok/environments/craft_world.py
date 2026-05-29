"""DeviceVecCraftWorld — a GPU-batched crafting gridworld with a genuine
skill DEPENDENCY DAG (the v6.0 substrate for real developmental learning).

A Crafter-style tech tree where deep achievements are UNREACHABLE without
their prerequisite skills, so reuse/composition is the only route to depth:

    collect_wood
      -> make_table            (wood)
           -> make_wood_pickaxe (wood + table)
                -> collect_stone (wood_pickaxe)        -> make_stone_pickaxe (wood+stone+table)
                -> collect_coal  (wood_pickaxe)        -> make_furnace       (stone+table)
                     make_stone_pickaxe -> collect_iron (stone_pickaxe)
                       collect_iron+collect_coal+furnace -> make_iron_pickaxe

Design choices for tractability (preserving the DEPENDENCY structure, which
is what matters): table/furnace/pickaxes are inventory tools the agent
carries (no placement/proximity); resources are non-depleting (collecting
does not remove the cell), so the world is stationary and skills repeatable.
The challenge is navigation (egocentric view) + the sparse, deep tech tree.

Batched, device-resident (N parallel envs). obs = flattened egocentric P x P
cell-type one-hot + inventory (+ optional goal one-hot). Actions (10):
0-3 move, 4 collect (resolves current cell + tool req), 5-9 craft. Reward:
sparse +1 per FIRST-TIME achievement this episode (Crafter convention), or
goal-conditioned (+1 & terminate on achieving a target node).
"""

import torch

from ragnarok.infrastructure.device import DEVICE

# Cell types (WALL only used for out-of-bounds padding in the egocentric view).
EMPTY, TREE, STONE, COAL, IRON, WALL = 0, 1, 2, 3, 4, 5
N_CELL_TYPES = 6
RESOURCE_TYPES = [TREE, STONE, COAL, IRON]

# Inventory item indices.
WOOD, STONE_I, COAL_I, IRON_I, WPICK, SPICK, IPICK, TABLE, FURNACE = range(9)
N_ITEMS = 9

# Achievement (DAG node) indices and their human names, ordered by depth.
ACH_NAMES = ["collect_wood", "make_table", "make_wood_pickaxe", "collect_stone",
             "collect_coal", "make_stone_pickaxe", "make_furnace",
             "collect_iron", "make_iron_pickaxe"]
N_ACH = len(ACH_NAMES)
(A_WOOD, A_TABLE, A_WPICK, A_STONE, A_COAL, A_SPICK, A_FURNACE, A_IRON,
 A_IPICK) = range(N_ACH)

# Approx dependency depth of each achievement (for the learning-to-learn curve).
ACH_DEPTH = {A_WOOD: 0, A_TABLE: 1, A_WPICK: 2, A_STONE: 3, A_COAL: 3,
             A_SPICK: 4, A_FURNACE: 4, A_IRON: 5, A_IPICK: 6}

# Prerequisite achievements that must precede each node (the DAG edges).
ACH_PREREQS = {
    A_WOOD: [], A_TABLE: [A_WOOD], A_WPICK: [A_TABLE],
    A_STONE: [A_WPICK], A_COAL: [A_WPICK],
    A_SPICK: [A_STONE], A_FURNACE: [A_STONE],
    A_IRON: [A_SPICK], A_IPICK: [A_IRON, A_COAL, A_FURNACE],
}


class DeviceVecCraftWorld:
    obs_dim = None        # set in __init__ (depends on view size + goal flag)
    action_dim = 10
    is_discrete = True

    def __init__(self, num_envs: int, grid: int = 9, view: int = 5,
                 max_steps: int = 200, n_resource: int = 6, goal=None,
                 seed: int = 0):
        self.num_envs = num_envs
        self.G = grid
        self.P = view                       # odd; egocentric window P x P
        self.half = view // 2
        self.max_steps = max_steps
        self.n_resource = n_resource        # cells of EACH resource type
        self.goal_conditioned = goal is not None
        self.goal_idx = goal
        self.obs_dim = self.P * self.P * N_CELL_TYPES + N_ITEMS \
            + (N_ACH if self.goal_conditioned else 0)
        self._gen = torch.Generator(device=DEVICE)
        self._gen.manual_seed(seed)
        self.reset()

    # ---- generation -----------------------------------------------------
    def _rand_cells(self, n_envs):
        """Place n_resource cells of each resource type at distinct random
        positions per env; return (grid, agent_pos)."""
        G = self.G
        grid = torch.zeros(n_envs, G, G, dtype=torch.long, device=DEVICE)
        # random permutation of cell indices per env, assign slots
        scores = torch.rand(n_envs, G * G, generator=self._gen, device=DEVICE)
        order = scores.argsort(dim=-1)                       # (n_envs, G*G)
        slot = 0
        for rt in RESOURCE_TYPES:
            idx = order[:, slot:slot + self.n_resource]      # (n_envs, n_res)
            rows, cols = idx // G, idx % G
            env_ar = torch.arange(n_envs, device=DEVICE).unsqueeze(-1)
            grid[env_ar, rows, cols] = rt
            slot += self.n_resource
        # agent on a cell guaranteed empty (next free slot)
        a_idx = order[:, slot]
        apos = torch.stack([a_idx // G, a_idx % G], dim=-1)  # (n_envs, 2)
        return grid, apos

    def reset(self):
        self.grid, self.pos = self._rand_cells(self.num_envs)
        self.inv = torch.zeros(self.num_envs, N_ITEMS, dtype=torch.long,
                               device=DEVICE)
        self.unlocked = torch.zeros(self.num_envs, N_ACH, dtype=torch.bool,
                                    device=DEVICE)
        self.steps = torch.zeros(self.num_envs, dtype=torch.long, device=DEVICE)
        self._set_state()
        return self.state

    # ---- observation ----------------------------------------------------
    def _egocentric(self):
        """(N, P, P) cell types around each agent, WALL-padded for OOB."""
        N, G, P, h = self.num_envs, self.G, self.P, self.half
        padded = torch.full((N, G + 2 * h, G + 2 * h), WALL, dtype=torch.long,
                            device=DEVICE)
        padded[:, h:h + G, h:h + G] = self.grid
        env_ar = torch.arange(N, device=DEVICE)
        out = torch.empty(N, P, P, dtype=torch.long, device=DEVICE)
        # agent at (pos+h) in padded coords; window top-left = pos
        for dr in range(P):
            for dc in range(P):
                rr = self.pos[:, 0] + dr
                cc = self.pos[:, 1] + dc
                out[:, dr, dc] = padded[env_ar, rr, cc]
        return out                       # center cell shows what the agent stands on

    def _set_state(self):
        ego = self._egocentric()                              # (N,P,P)
        onehot = torch.nn.functional.one_hot(ego, N_CELL_TYPES).float()
        flat = onehot.reshape(self.num_envs, -1)
        inv = self.inv.float().clamp(max=5.0) / 5.0           # normalised counts
        parts = [flat, inv]
        if self.goal_conditioned:
            g = torch.zeros(self.num_envs, N_ACH, device=DEVICE)
            g[:, self.goal_idx] = 1.0
            parts.append(g)
        self.state = torch.cat(parts, dim=-1)

    # ---- dynamics -------------------------------------------------------
    def _cur_cell(self):
        env_ar = torch.arange(self.num_envs, device=DEVICE)
        return self.grid[env_ar, self.pos[:, 0], self.pos[:, 1]]

    def _fire(self, ach_idx, mask, reward):
        """Mark achievement ach_idx for envs in mask; +1 reward on first time."""
        newly = mask & (~self.unlocked[:, ach_idx])
        self.unlocked[:, ach_idx] |= mask
        reward += newly.float()
        return newly

    def step(self, action):
        a = action.reshape(self.num_envs).long()
        N = self.num_envs
        reward = torch.zeros(N, device=DEVICE)

        # --- movement (0-3) ---
        deltas = torch.tensor([[-1, 0], [1, 0], [0, -1], [0, 1]], device=DEVICE)
        mv = a < 4
        if bool(mv.any()):
            d = deltas[a.clamp(max=3)]
            newpos = (self.pos + d).clamp(0, self.G - 1)
            self.pos = torch.where(mv.unsqueeze(-1), newpos, self.pos)

        # --- collect (4) ---
        coll = a == 4
        cell = self._cur_cell()
        # wood: tree, no tool
        m = coll & (cell == TREE)
        self.inv[m, WOOD] += 1; self._fire(A_WOOD, m, reward)
        # stone: needs wood pickaxe
        m = coll & (cell == STONE) & (self.inv[:, WPICK] >= 1)
        self.inv[m, STONE_I] += 1; self._fire(A_STONE, m, reward)
        # coal: needs wood pickaxe
        m = coll & (cell == COAL) & (self.inv[:, WPICK] >= 1)
        self.inv[m, COAL_I] += 1; self._fire(A_COAL, m, reward)
        # iron: needs stone pickaxe
        m = coll & (cell == IRON) & (self.inv[:, SPICK] >= 1)
        self.inv[m, IRON_I] += 1; self._fire(A_IRON, m, reward)

        # --- craft (5-9) ---  all recipes produce a tool flag (capped at 1)
        self._craft_tool(a == 5, [(WOOD, 1)], TABLE, A_TABLE, reward,
                         req_tools=[])
        self._craft_tool(a == 6, [(WOOD, 1)], WPICK, A_WPICK, reward,
                         req_tools=[TABLE])
        self._craft_tool(a == 7, [(WOOD, 1), (STONE_I, 1)], SPICK, A_SPICK,
                         reward, req_tools=[TABLE])
        self._craft_tool(a == 8, [(STONE_I, 1)], FURNACE, A_FURNACE, reward,
                         req_tools=[TABLE])
        self._craft_tool(a == 9, [(WOOD, 1), (COAL_I, 1), (IRON_I, 1)], IPICK,
                         A_IPICK, reward, req_tools=[TABLE, FURNACE])

        # --- bookkeeping / termination ---
        self.steps += 1
        if self.goal_conditioned:
            terminated = self.unlocked[:, self.goal_idx]
        else:
            terminated = torch.zeros(N, dtype=torch.bool, device=DEVICE)
        truncated = self.steps >= self.max_steps
        done = terminated | truncated
        if bool(done.any()):
            self._reset_done(done)
        self._set_state()
        return self.state, reward, terminated, truncated, done

    def _craft_tool(self, mask, consume, tool_idx, ach, reward, req_tools=()):
        """Craft a tool (boolean flag, capped at 1) if inputs + required tools
        present and not already owned."""
        ok = mask.clone()
        for it, q in consume:
            ok &= self.inv[:, it] >= q
        for t in req_tools:
            ok &= self.inv[:, t] >= 1
        ok &= self.inv[:, tool_idx] == 0
        for it, q in consume:
            self.inv[ok, it] -= q
        self.inv[ok, tool_idx] = 1
        self._fire(ach, ok, reward)
        return ok

    def _reset_done(self, done):
        n = int(done.sum().item())
        if n == 0:
            return
        grid, apos = self._rand_cells(n)
        self.grid[done] = grid
        self.pos[done] = apos
        self.inv[done] = 0
        self.unlocked[done] = False
        self.steps[done] = 0
