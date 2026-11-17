#!/usr/bin/env python3
"""
==============================================================================
Unified Pipeline Orchestrator: Full-Chain Sequential Handoff
REx-RL (External Surface) -> MEx-RL (Strategic Manager) -> GEx-RL (Tactical Resolution)
Evaluates end-to-end multi-agent execution across 100 synthetic episodes.
==============================================================================
"""

import os
import sys
import time
from typing import Any, Dict, List

import numpy as np
from sb3_contrib import MaskablePPO, RecurrentPPO

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gex_rl.gex_env import GExGymEnv
from mex_rl.mex_env import MExGymEnv
from rex_rl.rex_env import RExGymEnv


class UnifiedPipelineOrchestrator:
    """Orchestrates sequential handoff between REx-RL, MEx-RL, and GEx-RL."""

    def __init__(self, root_dir: str = None):
        if root_dir is None:
            root_dir = os.path.dirname(os.path.abspath(__file__))
        self.root_dir = root_dir

        rex_model_path = os.path.join(root_dir, "rex_rl", "rex_surface_best.zip")
        mex_model_path = os.path.join(root_dir, "mex_rl", "mex_manager_best.zip")
        gex_model_path = os.path.join(root_dir, "gex_rl", "gex_policy_best.zip")

        print("[*] Loading trained agent policies for unified pipeline...")
        self.rex_model = MaskablePPO.load(rex_model_path, device="cpu")
        self.mex_model = RecurrentPPO.load(mex_model_path, device="cpu")
        self.gex_model = MaskablePPO.load(gex_model_path, device="cpu")
        print("[+] All 3 tier models loaded successfully.")

        self.rex_env = RExGymEnv()
        self.mex_env = MExGymEnv()
        self.gex_env = GExGymEnv()

    def run_full_chain_episode(self, seed: int = 42) -> Dict[str, Any]:
        """Runs a complete sequential 3-tier episode."""
        trace = []

        # =====================================================================
        # STAGE 1: REx-RL (External Surface & Anomaly Analysis)
        # =====================================================================
        obs_rex, _ = self.rex_env.reset(seed=seed)
        done_rex = False
        steps_rex = 0
        reward_rex = 0.0
        context_vector = None

        while not done_rex and steps_rex < 25:
            masks = self.rex_env.action_masks()
            act_rex, _ = self.rex_model.predict(obs_rex, action_masks=masks, deterministic=True)
            obs_rex, r, term, trunc, info = self.rex_env.step(int(act_rex))
            steps_rex += 1
            reward_rex += r
            done_rex = term or trunc
            trace.append({
                "stage": "REx-RL",
                "step": steps_rex,
                "action": int(act_rex),
                "reward": round(r, 2),
                "handoff_ready": info.get("handoff_ready", False),
            })
            if info.get("won", False):
                context_vector = info.get("context_vector")

        rex_success = context_vector is not None
        if not rex_success:
            context_vector = np.array([0.33, 0.80, 0.05, 0.33], dtype=np.float32)

        # =====================================================================
        # STAGE 2: MEx-RL (Strategic Manager Domain Routing)
        # =====================================================================
        obs_mex, _ = self.mex_env.reset(seed=seed, context_vector=context_vector)
        done_mex = False
        steps_mex = 0
        reward_mex = 0.0
        lstm_states = None
        ep_starts = np.ones((1,), dtype=bool)

        while not done_mex and steps_mex < 30:
            act_mex, lstm_states = self.mex_model.predict(
                obs_mex,
                state=lstm_states,
                episode_start=ep_starts,
                deterministic=True,
            )
            obs_mex, r, term, trunc, info = self.mex_env.step(int(act_mex))
            steps_mex += 1
            reward_mex += r
            done_mex = term or trunc
            ep_starts = np.zeros((1,), dtype=bool)
            trace.append({
                "stage": "MEx-RL",
                "step": steps_mex,
                "action": int(act_mex),
                "reward": round(r, 2),
                "compromise": bool(obs_mex[41] == 1.0),
            })

        mex_success = bool(obs_mex[41] == 1.0)

        # =====================================================================
        # STAGE 3: GEx-RL (Tactical Local Exploitation / Resolution)
        # =====================================================================
        obs_gex, _ = self.gex_env.reset(seed=seed)
        done_gex = False
        steps_gex = 0
        reward_gex = 0.0

        while not done_gex and steps_gex < 30:
            masks = self.gex_env.action_masks()
            act_gex, _ = self.gex_model.predict(obs_gex, action_masks=masks, deterministic=True)
            obs_gex, r, term, trunc, info = self.gex_env.step(int(act_gex))
            steps_gex += 1
            reward_gex += r
            done_gex = term or trunc
            trace.append({
                "stage": "GEx-RL",
                "step": steps_gex,
                "action": int(act_gex),
                "reward": round(r, 2),
                "root_achieved": bool(obs_gex[-1] == 1.0),
            })

        gex_success = bool(obs_gex[-1] == 1.0)
        full_chain_success = rex_success and mex_success and gex_success

        return {
            "rex_success": rex_success,
            "mex_success": mex_success,
            "gex_success": gex_success,
            "full_chain_success": full_chain_success,
            "steps_rex": steps_rex,
            "steps_mex": steps_mex,
            "steps_gex": steps_gex,
            "total_steps": steps_rex + steps_mex + steps_gex,
            "total_reward": reward_rex + reward_mex + reward_gex,
            "context_vector": context_vector,
            "trace": trace,
        }

    def evaluate_benchmark(self, n_episodes: int = 100) -> Dict[str, Any]:
        """Runs 100 episodes and computes end-to-end benchmark statistics."""
        print(f"\n[*] Evaluating Unified Pipeline over {n_episodes} full-chain episodes...")
        rex_wins = 0
        mex_wins = 0
        gex_wins = 0
        full_wins = 0

        rex_steps = []
        mex_steps = []
        gex_steps = []
        total_steps_list = []
        total_rewards = []

        t0 = time.time()
        for ep in range(n_episodes):
            res = self.run_full_chain_episode(seed=40000 + ep)
            if res["rex_success"]:
                rex_wins += 1
            if res["mex_success"]:
                mex_wins += 1
            if res["gex_success"]:
                gex_wins += 1
            if res["full_chain_success"]:
                full_wins += 1

            rex_steps.append(res["steps_rex"])
            mex_steps.append(res["steps_mex"])
            gex_steps.append(res["steps_gex"])
            total_steps_list.append(res["total_steps"])
            total_rewards.append(res["total_reward"])

        duration = time.time() - t0

        metrics = {
            "episodes": n_episodes,
            "rex_win_rate": (rex_wins / n_episodes) * 100.0,
            "mex_win_rate": (mex_wins / n_episodes) * 100.0,
            "gex_win_rate": (gex_wins / n_episodes) * 100.0,
            "full_chain_win_rate": (full_wins / n_episodes) * 100.0,
            "mean_rex_steps": float(np.mean(rex_steps)),
            "mean_mex_steps": float(np.mean(mex_steps)),
            "mean_gex_steps": float(np.mean(gex_steps)),
            "mean_total_steps": float(np.mean(total_steps_list)),
            "mean_total_reward": float(np.mean(total_rewards)),
            "duration": duration,
        }

        print("==========================================================================================")
        print(" [Unified Pipeline] End-to-End Orchestration Results (100 Episodes)")
        print("==========================================================================================")
        print(f" Stage 1: REx-RL (External Surface Mapping)  -> Accuracy: {metrics['rex_win_rate']:.1f}% | Steps: {metrics['mean_rex_steps']:.2f}")
        print(f" Stage 2: MEx-RL (Strategic Manager Routing) -> Accuracy: {metrics['mex_win_rate']:.1f}% | Steps: {metrics['mean_mex_steps']:.2f}")
        print(f" Stage 3: GEx-RL (Tactical Fast Resolution)  -> Accuracy: {metrics['gex_win_rate']:.1f}% | Steps: {metrics['mean_gex_steps']:.2f}")
        print("------------------------------------------------------------------------------------------")
        print(f" FULL-CHAIN SUCCESS RATE                     : {metrics['full_chain_win_rate']:.1f}%")
        print(f" Mean Total Pipeline Steps                   : {metrics['mean_total_steps']:.2f} steps")
        print(f" Mean Cumulative Reward                      : {metrics['mean_total_reward']:.2f}")
        print(f" Execution Wall Time                         : {duration:.2f}s ({n_episodes / duration:.1f} eps/s)")
        print("==========================================================================================")

        return metrics


def main():
    orchestrator = UnifiedPipelineOrchestrator()
    # Sample trace demonstration
    sample_res = orchestrator.run_full_chain_episode(seed=42)
    print("\n[Sample Full-Chain Episode Trace (Seed=42)]")
    print(f"  REx Handoff Context: {np.array2string(sample_res['context_vector'], precision=2)}")
    print(f"  Stage Steps: REx={sample_res['steps_rex']}, MEx={sample_res['steps_mex']}, GEx={sample_res['steps_gex']} (Total: {sample_res['total_steps']})")
    print(f"  Full-Chain Compromise: {'SUCCESS (100%)' if sample_res['full_chain_success'] else 'FAILED'}")

    # Full 100-episode benchmark
    orchestrator.evaluate_benchmark(n_episodes=100)


if __name__ == "__main__":
    main()
