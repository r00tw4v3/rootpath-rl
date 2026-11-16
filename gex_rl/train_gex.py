#!/usr/bin/env python3
"""
==============================================================================
GEx-RL: Fast Tactical Exploration & Execution RL Trainer
Trains GPU-accelerated PPO over 2,000,000 timesteps at 250,000+ SPS.
Exports SB3-compatible MaskablePPO checkpoint: gex_rl/gex_policy_best.zip
==============================================================================
"""

import os
import sys
import time
from typing import Tuple

import numpy as np
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical

# Add root directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gex_rl.gex_env import GExGPUEnv, GExGymEnv, OBS_DIM, N_ACTIONS


class ActorCriticMLP(nn.Module):
    """Actor-Critic MLP with Action Masking matching SB3 net_arch=[128, 128]."""

    def __init__(self, obs_dim: int = OBS_DIM, act_dim: int = N_ACTIONS, hidden_dim: int = 128):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, act_dim),
        )
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def get_action(self, obs: torch.Tensor, masks: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = self.actor(obs)
        logits[~masks] = -1e9
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        value = self.critic(obs).squeeze(-1)
        return action, log_prob, value

    def evaluate(self, obs: torch.Tensor, masks: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = self.actor(obs)
        logits[~masks] = -1e9
        dist = Categorical(logits=logits)
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()
        value = self.critic(obs).squeeze(-1)
        return log_prob, entropy, value


def train_gex(total_timesteps: int = 2_000_000) -> float:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_envs = 4096 if device == "cuda" else 2048
    rollout_len = 16
    batch_size = num_envs * rollout_len  # 65,536 per iteration
    n_updates = (total_timesteps // batch_size) + 1

    print("==================================================================")
    print(" [GEx-RL Trainer] Starting Tactical High-Throughput Engine")
    print(f" Device: {device} ({torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'})")
    print(f" Parallel Envs: {num_envs} | Batch size: {batch_size}")
    print(f" Target Timesteps: {total_timesteps:,}")
    print("==================================================================")

    env = GExGPUEnv(num_envs=num_envs, device=device)
    policy = ActorCriticMLP().to(device)
    optimizer = optim.Adam(policy.parameters(), lr=5e-4)

    start_time = time.time()
    steps_collected = 0

    for update in range(1, n_updates + 1):
        b_obs = torch.zeros((rollout_len, num_envs, OBS_DIM), device=device)
        b_masks = torch.zeros((rollout_len, num_envs, N_ACTIONS), device=device, dtype=torch.bool)
        b_actions = torch.zeros((rollout_len, num_envs), device=device, dtype=torch.long)
        b_logprobs = torch.zeros((rollout_len, num_envs), device=device)
        b_rewards = torch.zeros((rollout_len, num_envs), device=device)
        b_values = torch.zeros((rollout_len, num_envs), device=device)

        for step in range(rollout_len):
            masks = env.get_action_masks()
            with torch.no_grad():
                action, logprob, value = policy.get_action(env.obs, masks)

            b_obs[step] = env.obs
            b_masks[step] = masks
            b_actions[step] = action
            b_logprobs[step] = logprob
            b_values[step] = value

            next_obs, reward, dones = env.step(action)
            b_rewards[step] = reward
            steps_collected += num_envs

        # GAE Advantage Estimation
        with torch.no_grad():
            last_masks = env.get_action_masks()
            _, _, last_val = policy.get_action(env.obs, last_masks)

        advantages = torch.zeros_like(b_rewards)
        lastgaelam = 0
        gamma = 0.99
        gae_lambda = 0.95

        for t in reversed(range(rollout_len)):
            nextvalues = last_val if t == rollout_len - 1 else b_values[t + 1]
            delta = b_rewards[t] + gamma * nextvalues - b_values[t]
            advantages[t] = lastgaelam = delta + gamma * gae_lambda * lastgaelam

        returns = advantages + b_values

        flat_obs = b_obs.view(-1, OBS_DIM)
        flat_masks = b_masks.view(-1, N_ACTIONS)
        flat_actions = b_actions.view(-1)
        flat_logprobs = b_logprobs.view(-1)
        flat_advantages = advantages.view(-1)
        flat_returns = returns.view(-1)
        flat_advantages = (flat_advantages - flat_advantages.mean()) / (flat_advantages.std() + 1e-8)

        # PPO Mini-batch Optimization
        clip_coef = 0.2
        ent_coef = 0.001
        vf_coef = 0.5
        mini_batch_size = 4096
        indices = torch.randperm(flat_obs.shape[0])

        for start in range(0, flat_obs.shape[0], mini_batch_size):
            mb_idx = indices[start:start + mini_batch_size]
            new_logprob, entropy, new_value = policy.evaluate(flat_obs[mb_idx], flat_masks[mb_idx], flat_actions[mb_idx])
            logratio = new_logprob - flat_logprobs[mb_idx]
            ratio = logratio.exp()

            mb_advantages = flat_advantages[mb_idx]
            pg_loss1 = -mb_advantages * ratio
            pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()

            v_loss = 0.5 * ((new_value - flat_returns[mb_idx]) ** 2).mean()
            loss = pg_loss - ent_coef * entropy.mean() + vf_coef * v_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            optimizer.step()

        elapsed = time.time() - start_time
        sps = steps_collected / elapsed
        if update % 5 == 0 or update == n_updates:
            print(f"Iteration {update:02d}/{n_updates} | Timesteps: {steps_collected:,} | Elapsed: {elapsed:.2f}s | Throughput: {sps:,.0f} SPS")

    total_time = time.time() - start_time
    final_sps = steps_collected / total_time
    print(f"\n[+] GPU Training Complete: {steps_collected:,} steps in {total_time:.2f}s ({final_sps:,.0f} SPS)")

    # Export to SB3-compatible MaskablePPO Checkpoint
    output_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoint_path = os.path.join(output_dir, "gex_policy_best")

    raw_env = GExGymEnv()
    masked_env = ActionMasker(raw_env, lambda e: e.action_masks())
    sb3_model = MaskablePPO(
        "MlpPolicy",
        masked_env,
        policy_kwargs=dict(net_arch=[128, 128]),
        verbose=0,
        device="cpu",
    )

    # Transfer learned weights
    sd = sb3_model.policy.state_dict()
    cpu_policy = policy.cpu()

    sd["mlp_extractor.policy_net.0.weight"] = cpu_policy.actor[0].weight.data
    sd["mlp_extractor.policy_net.0.bias"] = cpu_policy.actor[0].bias.data
    sd["mlp_extractor.policy_net.2.weight"] = cpu_policy.actor[2].weight.data
    sd["mlp_extractor.policy_net.2.bias"] = cpu_policy.actor[2].bias.data
    sd["action_net.weight"] = cpu_policy.actor[4].weight.data
    sd["action_net.bias"] = cpu_policy.actor[4].bias.data

    sd["mlp_extractor.value_net.0.weight"] = cpu_policy.critic[0].weight.data
    sd["mlp_extractor.value_net.0.bias"] = cpu_policy.critic[0].bias.data
    sd["mlp_extractor.value_net.2.weight"] = cpu_policy.critic[2].weight.data
    sd["mlp_extractor.value_net.2.bias"] = cpu_policy.critic[2].bias.data
    sd["value_net.weight"] = cpu_policy.critic[4].weight.data
    sd["value_net.bias"] = cpu_policy.critic[4].bias.data

    sb3_model.policy.load_state_dict(sd)
    sb3_model.save(checkpoint_path)
    print(f"[+] Saved SB3-compatible MaskablePPO checkpoint to: {checkpoint_path}.zip")

    # Quick evaluation
    successes = 0
    ep_lengths = []
    for ep in range(100):
        obs, _ = raw_env.reset(seed=5000 + ep)
        done = False
        steps = 0
        while not done:
            masks = raw_env.action_masks()
            act, _ = sb3_model.predict(obs, action_masks=masks, deterministic=True)
            obs, r, term, trunc, _ = raw_env.step(int(act))
            steps += 1
            done = term or trunc
        if obs[-1] == 1.0:
            successes += 1
        ep_lengths.append(steps)

    print(f"[+] GEx-RL 100-Episode Validation: Success Rate = {successes}% | Mean Steps = {np.mean(ep_lengths):.2f}")
    return final_sps


if __name__ == "__main__":
    train_gex()
