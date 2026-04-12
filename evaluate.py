"""Evaluate a trained Ragnarok agent.

Usage:
    python evaluate.py --checkpoint checkpoints/cartpole_xxx_final.pt --env cartpole --episodes 10
"""

import argparse
import numpy as np
import torch

from ragnarok.infrastructure.config import RagnarokConfig
from ragnarok.infrastructure.device import DEVICE
from ragnarok.environments.wrapper import RagnarokEnv
from ragnarok.environments.registry import get_env_spec
from ragnarok.core.agent import RagnarokAgent


def evaluate(checkpoint_path: str, env_name: str, episodes: int = 10,
             render: bool = False, seed: int = 42):
    """Run evaluation episodes with a trained agent."""
    spec = get_env_spec(env_name)
    config = RagnarokConfig(seed=seed)
    config.world_model.obs_dim = spec.obs_dim
    config.world_model.action_dim = spec.action_dim

    env = RagnarokEnv(spec.gym_name, seed=seed)
    agent = RagnarokAgent(config, env)
    agent.load(checkpoint_path)

    print(f"[Ragnarok Eval] Loaded: {checkpoint_path}")
    print(f"[Ragnarok Eval] Environment: {spec.gym_name}")
    print(f"[Ragnarok Eval] Episodes: {episodes}")

    rewards = []
    lengths = []

    for ep in range(1, episodes + 1):
        obs = env.reset()
        h, z = agent.rssm.initial_state(1, DEVICE)
        prev_action = torch.zeros(1, env.action_dim, device=DEVICE)

        total_reward = 0.0
        steps = 0
        done = False

        while not done:
            obs_t = torch.tensor(obs, device=DEVICE).unsqueeze(0)

            with torch.no_grad():
                h, z = agent.rssm.encode_observation(obs_t, h, z, prev_action)
                action_t = agent.actor_critic.act(h, z, deterministic=True)
                action_np = action_t.squeeze(0).cpu().numpy()

            obs, reward, terminated, truncated, _ = env.step(action_np)
            done = terminated or truncated
            total_reward += reward
            steps += 1
            prev_action = action_t

        rewards.append(total_reward)
        lengths.append(steps)
        print(f"  Episode {ep}: reward = {total_reward:.1f}, steps = {steps}")

    env.close()

    print(f"\n[Ragnarok Eval] Results over {episodes} episodes:")
    print(f"  Mean reward: {np.mean(rewards):.1f} +/- {np.std(rewards):.1f}")
    print(f"  Min/Max: {np.min(rewards):.1f} / {np.max(rewards):.1f}")
    print(f"  Mean length: {np.mean(lengths):.1f}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Ragnarok agent")
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint path")
    parser.add_argument("--env", type=str, default="cartpole", help="Environment name")
    parser.add_argument("--episodes", type=int, default=10, help="Number of eval episodes")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    evaluate(args.checkpoint, args.env, args.episodes, seed=args.seed)


if __name__ == "__main__":
    main()
