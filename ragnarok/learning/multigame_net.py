"""MultiGameConvNet — a shared CNN encoder with per-game actor/critic heads.

The substrate for cross-game knowledge ACCUMULATION: one perception encoder is
trained on several games at once, building GENERAL game-perception; each game
keeps its own small policy/value head (games have different action spaces).
Later, a NEW game can reuse the shared encoder + a fresh head — the test of
"more games known => a new game learned faster".

Use `set_game(name)` to select the active head, then call like a normal
actor-critic net: `logits, value = net(obs)`. This lets it drop straight into
DiscretePPO (one optimiser over all params; training game A only produces
gradients for the shared encoder + A's head).
"""

import torch
import torch.nn as nn

from ragnarok.infrastructure.device import DEVICE


class MultiGameConvNet(nn.Module):
    def __init__(self, img_hw, action_dims: dict, hidden=256):
        """action_dims: {game_name: action_dim}."""
        super().__init__()
        self.img_hw = img_hw
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 4, stride=2), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2), nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1), nn.ReLU())
        with torch.no_grad():
            d = self.conv(torch.zeros(1, 3, img_hw, img_hw)).reshape(1, -1).shape[1]
        self.fc = nn.Sequential(nn.Linear(d, hidden), nn.ReLU())
        self.actors = nn.ModuleDict(
            {g: nn.Linear(hidden, a) for g, a in action_dims.items()})
        self.critics = nn.ModuleDict(
            {g: nn.Linear(hidden, 1) for g in action_dims})
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.orthogonal_(m.weight, gain=2 ** 0.5)
                nn.init.zeros_(m.bias)
        for g in action_dims:
            nn.init.orthogonal_(self.actors[g].weight, gain=0.01)
        self.active = next(iter(action_dims))
        self.to(DEVICE)

    def set_game(self, name):
        self.active = name
        return self

    def encode(self, obs):
        B = obs.shape[0]
        x = obs.view(B, 3, self.img_hw, self.img_hw)
        return self.fc(self.conv(x).reshape(B, -1))

    def forward(self, obs):
        h = self.encode(obs)
        return self.actors[self.active](h), self.critics[self.active](h).squeeze(-1)

    def encoder_state(self):
        """conv+fc weights — the transferable general perception."""
        return {k: v.clone() for k, v in self.state_dict().items()
                if k.startswith("conv.") or k.startswith("fc.")}
