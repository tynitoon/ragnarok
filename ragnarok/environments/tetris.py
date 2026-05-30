"""DeviceVecTetris — GPU-batched Tetris with PLACEMENT-as-MACRO-ACTION + pixels.

The bridge to complex games (P4). Instead of frame-by-frame control (a brutal
long-horizon problem), each step the agent chooses WHERE to drop the current
piece: a macro-action = (column, rotation). The piece falls, lands on the
stack, full rows clear (+reward), and a new piece appears. Game over when a
piece cannot be placed. This collapses the horizon to ~one decision per piece
and turns Tetris into a placement-planning problem — what the project's
options/planning machinery is built for.

Device-resident: N boards on the accelerator, `.state` is a flattened RGB
image (the board + the current piece shown above it), `.step(action)` a
discrete macro-action per env. action = rot * W + col (action_dim = 4 * W).

Pieces: the 7 tetrominoes. Reward: +1 per cell placed*0 ... actually +line^2
bonus per clear (1->1, 2->4, 3->9, 4->16) and a small per-piece survival; -big
on game over. Optional shaping (lower stack / fewer holes) can be added.
"""

import torch

from ragnarok.infrastructure.device import DEVICE

# 7 tetrominoes, each as a list of 4 rotations; each rotation = list of (r,c)
# cells in a 4x4 box (r down from top, c right). Minimal rotation sets.
_PIECES = [
    [[(0, 0), (0, 1), (0, 2), (0, 3)], [(0, 0), (1, 0), (2, 0), (3, 0)],
     [(0, 0), (0, 1), (0, 2), (0, 3)], [(0, 0), (1, 0), (2, 0), (3, 0)]],          # I
    [[(0, 0), (0, 1), (1, 0), (1, 1)]] * 4,                                         # O
    [[(0, 1), (1, 0), (1, 1), (1, 2)], [(0, 1), (1, 1), (1, 2), (2, 1)],
     [(1, 0), (1, 1), (1, 2), (2, 1)], [(0, 1), (1, 0), (1, 1), (2, 1)]],          # T
    [[(0, 1), (0, 2), (1, 0), (1, 1)], [(0, 0), (1, 0), (1, 1), (2, 1)],
     [(0, 1), (0, 2), (1, 0), (1, 1)], [(0, 0), (1, 0), (1, 1), (2, 1)]],          # S
    [[(0, 0), (0, 1), (1, 1), (1, 2)], [(0, 1), (1, 0), (1, 1), (2, 0)],
     [(0, 0), (0, 1), (1, 1), (1, 2)], [(0, 1), (1, 0), (1, 1), (2, 0)]],          # Z
    [[(0, 2), (1, 0), (1, 1), (1, 2)], [(0, 1), (1, 1), (2, 1), (2, 2)],
     [(1, 0), (1, 1), (1, 2), (2, 0)], [(0, 0), (0, 1), (1, 1), (2, 1)]],          # L
    [[(0, 0), (1, 0), (1, 1), (1, 2)], [(0, 1), (0, 2), (1, 1), (2, 1)],
     [(1, 0), (1, 1), (1, 2), (2, 2)], [(0, 1), (1, 1), (2, 0), (2, 1)]],          # J
]


class DeviceVecTetris:
    def __init__(self, num_envs, width=8, height=14, tile=3, max_pieces=300,
                 hole_penalty=0.0, height_penalty=0.0, img=None, max_steps=None,
                 seed=0):
        # img is accepted for a uniform game interface (ignored; the board sets
        # the image size). max_steps, if given, caps pieces per game.
        if max_steps is not None:
            max_pieces = max_steps
        self.num_envs = num_envs
        self.W, self.Hb = width, height
        self.tile = tile
        # image shows a (height+4) x width board (4 top rows preview the piece)
        self.img_hw = max(self.Hb + 4, self.W) * tile     # square-ish, padded
        self.IH = (self.Hb + 4)
        self.obs_dim = 3 * self.img_hw * self.img_hw
        self.n_rot = 4
        self.action_dim = self.n_rot * self.W
        self.max_pieces = max_pieces
        self.HOLE_PEN = hole_penalty
        self.HGT_PEN = height_penalty
        self._gen = torch.Generator(device=DEVICE); self._gen.manual_seed(seed)
        # precompute piece cell tensors: (7,4,4,2) long (piece, rot, cell, (r,c))
        self._cells = torch.tensor(_PIECES, dtype=torch.long, device=DEVICE)
        self.reset()

    def _new_piece(self, mask):
        n = self.num_envs
        pid = torch.randint(0, 7, (n,), generator=self._gen, device=DEVICE)
        self.piece = torch.where(mask, pid, self.piece)

    def reset(self):
        n = self.num_envs
        self.board = torch.zeros(n, self.Hb, self.W, dtype=torch.bool, device=DEVICE)
        self.piece = torch.zeros(n, dtype=torch.long, device=DEVICE)
        self.lines = torch.zeros(n, device=DEVICE)        # cumulative lines cleared
        self.pieces_placed = torch.zeros(n, dtype=torch.long, device=DEVICE)
        self.over = torch.zeros(n, dtype=torch.bool, device=DEVICE)
        self.cum_lines = torch.zeros(n, device=DEVICE)
        self.cum_games = torch.zeros(n, device=DEVICE)
        self._new_piece(torch.ones(n, dtype=torch.bool, device=DEVICE))
        self._render()
        return self.state

    # ---- placement -------------------------------------------------------
    @torch.no_grad()
    def step(self, action):
        N = self.num_envs
        a = action.reshape(N).long()
        rot = (a // self.W).clamp(0, self.n_rot - 1)
        col = (a % self.W)
        cells = self._cells[self.piece, rot]              # (N,4,2) (r,c) in 4x4 box
        cr, cc = cells[:, :, 0], cells[:, :, 1]           # (N,4)
        # normalise the piece to start at column 0, then CLAMP the chosen column
        # so the piece always fits horizontally (no off-board placement -> no
        # unfair death; the agent just can't push past the wall, as in real Tetris)
        cc = cc - cc.min(dim=1, keepdim=True).values
        pw = cc.max(dim=1, keepdim=True).values + 1       # (N,1) piece width
        col = col.view(N, 1).clamp(min=0).minimum(self.W - pw)
        cc = cc + col
        valid_col = torch.ones(N, dtype=torch.bool, device=DEVICE)

        # find the lowest landing: for each env, the max drop offset d s.t. all
        # (cr+d, cc) are in-board and empty. Scan d from top.
        reward = torch.zeros(N, device=DEVICE)
        landing = torch.full((N,), -1, dtype=torch.long, device=DEVICE)
        still_clear = valid_col & ~self.over               # clear fall path so far
        ar = torch.arange(N, device=DEVICE)
        # the piece falls straight down: landing = deepest base-row d such that
        # it fits at EVERY row from 0..d (a clear path), not just at d.
        for d in range(self.Hb):
            rr = cr + d                                    # (N,4)
            in_board = (rr < self.Hb).all(dim=1)
            rrc = rr.clamp(0, self.Hb - 1)
            occ = self.board[ar.view(N, 1), rrc, cc].any(dim=1)
            fits = in_board & ~occ
            clear_here = fits & still_clear
            landing = torch.where(clear_here, torch.full_like(landing, d), landing)
            still_clear = still_clear & fits               # once blocked, stays blocked
        placeable = (landing >= 0) & valid_col & ~self.over
        # place piece at the landing
        rr = (cr + landing.view(N, 1)).clamp(0, self.Hb - 1)
        pl = placeable.view(N, 1).expand(N, 4)
        env_idx = ar.view(N, 1).expand(N, 4)
        self.board[env_idx[pl], rr[pl], cc[pl]] = True
        self.pieces_placed += placeable.long()

        # clear full lines
        full = self.board.all(dim=2)                       # (N,Hb) rows fully filled
        ncleared = full.sum(dim=1).float()
        if bool((ncleared > 0).any()):
            self._clear_lines(full)
        self.lines += ncleared
        self.cum_lines += ncleared
        reward += ncleared ** 2 * 2.0                      # 2,8,18,32 (multi-line worth more)
        reward += placeable.float() * 0.5                  # SURVIVAL/progress per piece
        #   (no absolute hole/height penalty: it makes living net-negative and
        #    the agent learns to suicide. Survival reward implicitly rewards
        #    clean stacking, since a clean board survives longer.)
        if self.HGT_PEN or self.HOLE_PEN:                  # opt-in potential shaping only
            heights, holes = self._heights_holes()
            reward -= self.HGT_PEN * heights.float() + self.HOLE_PEN * holes.float()

        # game over: piece couldn't be placed (or top row used)
        topfilled = self.board[:, 0, :].any(dim=1)
        dead = (~placeable & ~self.over) | (topfilled & ~self.over)
        reward -= dead.float() * 5.0
        self.over = self.over | dead
        self.cum_games += dead.float()

        self.pieces_placed += 0
        truncated = self.pieces_placed >= self.max_pieces
        done = self.over | truncated
        self._reset_done(done)
        self._new_piece(placeable & ~done)                 # next piece for alive
        self._render()
        terminated = dead
        return self.state, reward, terminated, truncated, done

    def _clear_lines(self, full):
        """Remove full rows per env, shift everything above down."""
        N = self.num_envs
        for i in range(N):
            f = full[i]
            if bool(f.any()):
                keep = self.board[i][~f]                    # (k,W)
                k = keep.shape[0]
                nb = torch.zeros(self.Hb, self.W, dtype=torch.bool, device=DEVICE)
                if k > 0:
                    nb[self.Hb - k:] = keep
                self.board[i] = nb

    def _heights_holes(self):
        # column heights = Hb - first filled row; holes = empty cells under the top
        N = self.num_envs
        filled = self.board                                # (N,Hb,W)
        rows = torch.arange(self.Hb, device=DEVICE).view(1, self.Hb, 1)
        first = torch.where(filled, rows, torch.full_like(rows, self.Hb))
        toprow = first.min(dim=1).values                   # (N,W) first filled row per col
        heights = (self.Hb - toprow).clamp(min=0)
        col_filled = filled.sum(dim=1)                     # (N,W) filled count per col
        holes = (heights - col_filled).clamp(min=0).sum(dim=1)
        return heights.sum(dim=1), holes

    def _reset_done(self, done):
        if bool(done.any()):
            self.board[done] = False
            self.lines = torch.where(done, torch.zeros_like(self.lines), self.lines)
            self.pieces_placed = torch.where(done, torch.zeros_like(self.pieces_placed),
                                             self.pieces_placed)
            self.over = self.over & ~done
            self._new_piece(done)

    # ---- rendering ------------------------------------------------------
    def _render(self):
        N = self.num_envs
        S = self.img_hw // self.tile                       # cells across the square
        g = torch.zeros(N, 3, S, S, device=DEVICE)
        # board occupies rows [4 .. 4+Hb), cols [0..W); top 4 rows show the piece
        off = 4
        ar = torch.arange(N, device=DEVICE)
        # stack (blue-ish)
        b = self.board.float()                             # (N,Hb,W)
        g[:, 2, off:off + self.Hb, :self.W] = torch.maximum(g[:, 2, off:off + self.Hb, :self.W], b)
        g[:, 1, off:off + self.Hb, :self.W] = torch.maximum(g[:, 1, off:off + self.Hb, :self.W], b * 0.4)
        # current piece preview (top-left of the 4x4 area), green
        cells = self._cells[self.piece, 0]                 # (N,4,2)
        pr, pc = cells[:, :, 0].clamp(0, 3), cells[:, :, 1].clamp(0, S - 1)
        g[ar.view(N, 1), 1, pr, pc] = 1.0
        img = g.repeat_interleave(self.tile, 2).repeat_interleave(self.tile, 3)
        self._img = img.reshape(N, -1)

    @property
    def state(self):
        return self._img

    def stats(self):
        return dict(mean_lines=float(self.cum_lines.mean()),
                    games=float(self.cum_games.sum()))

    @torch.no_grad()
    def evaluate_placements(self):
        """For the current (board, piece) of each env, return outcome metrics for
        ALL placements WITHOUT mutating state: (N, action_dim, 4) =
        [lines, holes, height, dead]. Post-place (pre-compaction) metrics — a
        fast, planning-grade approximation of the dynamics (where the piece lands
        = gravity+collision; rotation via the piece cells; line completion).
        This is the 'understanding of the dynamics' a planner/world-model uses."""
        N, W, Hb = self.num_envs, self.W, self.Hb
        ar = torch.arange(N, device=DEVICE)
        rows = torch.arange(Hb, device=DEVICE).view(1, Hb, 1)
        out = torch.zeros(N, self.action_dim, 4, device=DEVICE)
        for a in range(self.action_dim):
            rot, col = a // W, a % W
            cells = self._cells[self.piece, rot]
            cr, cc = cells[:, :, 0], cells[:, :, 1]
            cc = cc - cc.min(dim=1, keepdim=True).values
            pw = cc.max(dim=1, keepdim=True).values + 1
            colc = torch.full((N, 1), col, device=DEVICE).clamp(min=0).minimum(W - pw)
            cc = cc + colc
            landing = torch.full((N,), -1, dtype=torch.long, device=DEVICE)
            clear = torch.ones(N, dtype=torch.bool, device=DEVICE)
            for d in range(Hb):
                rr = cr + d
                inb = (rr < Hb).all(dim=1)
                rrc = rr.clamp(0, Hb - 1)
                occ = self.board[ar.view(N, 1), rrc, cc].any(dim=1)
                fits = inb & ~occ
                ch = fits & clear
                landing = torch.where(ch, torch.full_like(landing, d), landing)
                clear = clear & fits
            placeable = landing >= 0
            nb = self.board.clone()
            rr = (cr + landing.view(N, 1)).clamp(0, Hb - 1)
            pl = placeable.view(N, 1).expand(N, 4)
            ei = ar.view(N, 1).expand(N, 4)
            nb[ei[pl], rr[pl], cc[pl]] = True
            full = nb.all(dim=2)
            lines = full.sum(dim=1).float()
            first = torch.where(nb, rows, torch.full_like(rows, Hb)).min(dim=1).values
            heights = (Hb - first).clamp(min=0)
            holes = (heights - nb.sum(dim=1)).clamp(min=0).sum(dim=1).float()
            out[:, a, 0] = lines
            out[:, a, 1] = holes
            out[:, a, 2] = heights.sum(dim=1).float()
            out[:, a, 3] = ((~placeable) | nb[:, 0, :].any(dim=1)).float()
        return out

