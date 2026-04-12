"""Episodic memory: vector store of past experiences.

Stores latent states (h_t) with their associated actions and rewards.
Retrieves similar past experiences by L2 distance to provide context
to the policy — "I've seen something like this before."
"""

import numpy as np


class EpisodicMemory:
    """Bounded episodic memory with L2 nearest-neighbor retrieval.

    Stores (h_t, action, reward) tuples. When full, uses reservoir
    sampling for replacement.
    """

    def __init__(self, state_dim: int, action_dim: int, capacity: int = 50_000):
        self.capacity = capacity
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.count = 0

        # Pre-allocate storage
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)

    @property
    def size(self) -> int:
        return min(self.count, self.capacity)

    def add(self, state: np.ndarray, action: np.ndarray, reward: float):
        """Add an experience to memory.

        Uses reservoir sampling when buffer is full to maintain
        a uniform random sample of all experiences seen.
        """
        if self.count < self.capacity:
            idx = self.count
        else:
            # Reservoir sampling
            idx = np.random.randint(0, self.count + 1)
            if idx >= self.capacity:
                self.count += 1
                return  # Skip this sample

        self.states[idx] = state
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.count += 1

    def add_batch(self, states: np.ndarray, actions: np.ndarray, rewards: np.ndarray):
        """Add a batch of experiences."""
        for i in range(len(states)):
            self.add(states[i], actions[i], rewards[i])

    def query(self, state: np.ndarray, k: int = 5) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Find k nearest neighbors by L2 distance.

        Args:
            state: query state vector (state_dim,)
            k: number of neighbors

        Returns:
            (states, actions, rewards, distances) of k nearest neighbors
        """
        if self.size == 0:
            empty_s = np.zeros((0, self.state_dim), dtype=np.float32)
            empty_a = np.zeros((0, self.action_dim), dtype=np.float32)
            empty_r = np.zeros(0, dtype=np.float32)
            empty_d = np.zeros(0, dtype=np.float32)
            return empty_s, empty_a, empty_r, empty_d

        k = min(k, self.size)
        # L2 distance (brute force — fast for <500K entries at 128-dim)
        diffs = self.states[:self.size] - state[np.newaxis, :]
        distances = np.sqrt(np.sum(diffs ** 2, axis=1))

        # Top-k nearest
        indices = np.argpartition(distances, k)[:k]
        sorted_idx = indices[np.argsort(distances[indices])]

        return (
            self.states[sorted_idx],
            self.actions[sorted_idx],
            self.rewards[sorted_idx],
            distances[sorted_idx],
        )

    def get_context(self, state: np.ndarray, k: int = 5) -> np.ndarray:
        """Get context vector: mean of k nearest neighbor states.

        This context vector is concatenated with (h, z) as input to the policy.
        Returns zeros if memory is empty.
        """
        if self.size == 0:
            return np.zeros(self.state_dim, dtype=np.float32)

        states, _, _, _ = self.query(state, k)
        return states.mean(axis=0)

    def state_dict(self) -> dict:
        """Serialize memory state."""
        return {
            "states": self.states[:self.size].copy(),
            "actions": self.actions[:self.size].copy(),
            "rewards": self.rewards[:self.size].copy(),
            "count": self.count,
            "capacity": self.capacity,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
        }

    @classmethod
    def from_state_dict(cls, state: dict) -> "EpisodicMemory":
        """Deserialize memory from state dict."""
        mem = cls(state["state_dim"], state["action_dim"], state["capacity"])
        n = len(state["states"])
        mem.states[:n] = state["states"]
        mem.actions[:n] = state["actions"]
        mem.rewards[:n] = state["rewards"]
        mem.count = state["count"]
        return mem
