"""Sequence replay buffer for world model training.

Stores episodes and samples subsequences for training the RSSM.
"""

import numpy as np
from collections import deque


class Episode:
    """A single episode of (obs, action, reward, done) transitions."""

    __slots__ = ("observations", "actions", "rewards", "dones", "length")

    def __init__(self, observations: np.ndarray, actions: np.ndarray,
                 rewards: np.ndarray, dones: np.ndarray):
        self.observations = observations  # (T, obs_dim)
        self.actions = actions            # (T, action_dim) one-hot for discrete
        self.rewards = rewards            # (T,)
        self.dones = dones                # (T,)
        self.length = len(observations)


class ReplayBuffer:
    """Episode-based replay buffer with sequence sampling.

    Stores complete episodes and samples random subsequences
    for training the world model.
    """

    def __init__(self, capacity: int = 1_000_000):
        self.capacity = capacity
        self.episodes: deque[Episode] = deque()
        self.total_steps = 0

    def add_episode(self, observations: np.ndarray, actions: np.ndarray,
                    rewards: np.ndarray, dones: np.ndarray):
        """Add a complete episode to the buffer."""
        episode = Episode(
            observations=np.asarray(observations, dtype=np.float32),
            actions=np.asarray(actions, dtype=np.float32),
            rewards=np.asarray(rewards, dtype=np.float32),
            dones=np.asarray(dones, dtype=np.float32),
        )
        self.episodes.append(episode)
        self.total_steps += episode.length

        # Remove oldest episodes if over capacity
        while self.total_steps > self.capacity and len(self.episodes) > 1:
            removed = self.episodes.popleft()
            self.total_steps -= removed.length

    @property
    def max_episode_length(self) -> int:
        """Longest episode currently in the buffer."""
        if not self.episodes:
            return 0
        return max(ep.length for ep in self.episodes)

    def sample_sequences(self, batch_size: int, seq_length: int
                         ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Sample random subsequences from stored episodes.

        Automatically caps seq_length to the longest available episode
        to avoid excessive padding.

        Returns:
            obs:     (batch, actual_seq_length, obs_dim)
            actions: (batch, actual_seq_length, action_dim)
            rewards: (batch, actual_seq_length)
            dones:   (batch, actual_seq_length)
        """
        # Cap sequence length to what's actually available
        effective_seq_length = min(seq_length, self.max_episode_length)
        seq_length = max(effective_seq_length, 2)  # Need at least 2 steps

        # Filter episodes long enough
        valid_episodes = [ep for ep in self.episodes if ep.length >= seq_length]
        if not valid_episodes:
            # Fall back: use any episode, use their full length
            valid_episodes = list(self.episodes)

        obs_list, act_list, rew_list, done_list = [], [], [], []

        for _ in range(batch_size):
            ep = valid_episodes[np.random.randint(len(valid_episodes))]

            if ep.length >= seq_length:
                start = np.random.randint(0, ep.length - seq_length + 1)
                obs_list.append(ep.observations[start:start + seq_length])
                act_list.append(ep.actions[start:start + seq_length])
                rew_list.append(ep.rewards[start:start + seq_length])
                done_list.append(ep.dones[start:start + seq_length])
            else:
                # Pad shorter episodes
                pad_len = seq_length - ep.length
                obs_list.append(np.pad(ep.observations, ((0, pad_len), (0, 0))))
                act_list.append(np.pad(ep.actions, ((0, pad_len), (0, 0))))
                rew_list.append(np.pad(ep.rewards, (0, pad_len)))
                done_list.append(np.pad(ep.dones, (0, pad_len), constant_values=1.0))

        return (
            np.stack(obs_list),
            np.stack(act_list),
            np.stack(rew_list),
            np.stack(done_list),
        )

    @property
    def num_episodes(self) -> int:
        return len(self.episodes)

    def __len__(self) -> int:
        return self.total_steps
