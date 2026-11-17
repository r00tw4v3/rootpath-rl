#!/usr/bin/env python3
"""
==============================================================================
MEx-RL Trainer: Upgraded Strategic Manager with REx Context Handoff
Trains RecurrentPPO (MlpLstmPolicy, lstm_hidden_size=128) over 46-dim state.
Exports optimal checkpoint: mex_rl/mex_manager_best.zip
==============================================================================
"""

import os
import sys
import time

import numpy as np
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.env_util import make_vec_env
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mex_rl.mex_env import MExGymEnv


def make_env():
    return MExGymEnv()


def train_mex(total_timesteps: int = 80_000) -> float:
    output_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoint_path = os.path.join(output_dir, "mex_manager_best")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_envs = 32

    print("==================================================================")
    print(" [MEx-RL Trainer] Training Upgraded Context-Aware Strategic Manager")
    print(f" Device: {device} ({torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'})")
    print(f" Parallel Envs: {n_envs} | Target Timesteps: {total_timesteps:,}")
    print(" Observation Dim: 46 (42 POMDP + 4 REx Handoff Context)")
    print("==================================================================")

    vec_env = make_vec_env(make_env, n_envs=n_envs)

    policy_kwargs = dict(
        lstm_hidden_size=128,
        n_lstm_layers=1,
        shared_lstm=False,
        enable_critic_lstm=True,
    )

    model = RecurrentPPO(
        "MlpLstmPolicy",
        vec_env,
        policy_kwargs=policy_kwargs,
        learning_rate=5e-4,
        n_steps=64,
        batch_size=128,
        n_epochs=4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=0,
        device=device,
    )

    t0 = time.time()
    model.learn(total_timesteps=total_timesteps)
    train_time = time.time() - t0
    final_sps = total_timesteps / train_time

    # Save checkpoint
    model.save(checkpoint_path)
    print(f"\n[+] Saved optimal checkpoint to: {checkpoint_path}.zip")
    print(f"[+] Total Training Time: {train_time:.1f}s ({final_sps:,.0f} SPS)")

    # 100-Episode Validation
    print("\n[*] Evaluating Upgraded MEx-RL Manager over 100 randomized test episodes...")
    eval_env = MExGymEnv()
    successes = 0
    ep_lengths = []
    rewards = []

    for ep in range(100):
        obs, _ = eval_env.reset(seed=9000 + ep)
        done = False
        steps = 0
        ep_rew = 0.0
        lstm_states = None
        episode_starts = np.ones((1,), dtype=bool)

        while not done:
            act, lstm_states = model.predict(
                obs,
                state=lstm_states,
                episode_start=episode_starts,
                deterministic=True,
            )
            obs, r, term, trunc, info = eval_env.step(int(act))
            steps += 1
            ep_rew += r
            done = term or trunc
            episode_starts = np.zeros((1,), dtype=bool)

        if info.get("won", False):
            successes += 1
        ep_lengths.append(steps)
        rewards.append(ep_rew)

    mean_len = float(np.mean(ep_lengths))
    mean_rew = float(np.mean(rewards))
    print(f"[+] MEx-RL Evaluation Results:")
    print(f"    Success Rate   : {successes:.1f}%")
    print(f"    Mean Ep Length : {mean_len:.2f} steps")
    print(f"    Mean Reward    : {mean_rew:.2f}")
    print("==================================================================")

    return final_sps


if __name__ == "__main__":
    train_mex()
