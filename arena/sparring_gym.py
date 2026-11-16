#!/usr/bin/env python3
"""
==============================================================================
Adversarial Arena: Sparring Gym CLI & Dynamic Battleground
Sparring facility where Challenger (GEx/MEx) spars against adaptive Defender.
Saves battle-hardened checkpoint: arena/<agent>_arena_champion.zip
==============================================================================
"""

import argparse
import os
import sys
import time
from typing import Optional, Tuple

import numpy as np
from sb3_contrib import MaskablePPO, RecurrentPPO
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from arena.combat_env import CombatArenaEnv


def compute_elo(r_challenger: float, r_defender: float, challenger_won: bool, k: float = 32.0) -> Tuple[float, float]:
    """Computes FIDE Elo rating update."""
    e_c = 1.0 / (1.0 + 10.0 ** ((r_defender - r_challenger) / 400.0))
    s_c = 1.0 if challenger_won else 0.0
    delta_c = k * (s_c - e_c)
    return r_challenger + delta_c, r_defender - delta_c


def run_arena(
    agent_type: str = "mex",
    max_rounds: int = 200,
    target_score: float = 85.0,
    time_limit: Optional[float] = None,
    save_dir: str = "arena",
):
    start_time = time.time()
    os.makedirs(save_dir, exist_ok=True)
    champion_path = os.path.join(save_dir, f"{agent_type.lower()}_arena_champion.zip")

    # Load Challenger Model
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if agent_type.lower() == "mex":
        model_path = os.path.join(root_dir, "mex_rl", "mex_manager_best.zip")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model checkpoint not found at {model_path}. Train MEx first.")
        challenger = RecurrentPPO.load(model_path, device="cpu")
    else:
        model_path = os.path.join(root_dir, "gex_rl", "gex_policy_best.zip")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model checkpoint not found at {model_path}. Train GEx first.")
        challenger = MaskablePPO.load(model_path, device="cpu")

    env = CombatArenaEnv(agent_type=agent_type)

    # Initialize Elo ratings
    elo_challenger = 1200.0
    elo_defender = 1200.0

    wins = 0
    total_rounds = 0
    all_scores = []
    all_steps = []
    best_win_rate = 0.0

    print("==========================================================================================")
    print(f" [Adversarial Arena] Initializing Competitive Sparring Gym")
    print(f" Challenger Agent : {agent_type.upper()} ({model_path})")
    print(f" Defender Policy  : Adaptive Hardener (Constraint: availability >= 0.85)")
    print(f" Stopping Criteria: Max Rounds = {max_rounds} | Target Win Rate = {target_score:.1f}%")
    if time_limit:
        print(f" Time Limit       : {time_limit:.1f}s")
    print("==========================================================================================")
    print(f"{'Round':<6} | {'Outcome':<15} | {'Steps':<6} | {'Challenger':<11} | {'Defender':<10} | {'Win Rate':<9} | {'C-Elo':<6} | {'D-Elo':<6} | {'Avail':<6}")
    print("-" * 90)

    for r in range(1, max_rounds + 1):
        obs, _ = env.reset(seed=10000 + r)
        done = False
        round_steps = 0
        lstm_states = None
        ep_starts = np.ones((1,), dtype=bool)

        while not done:
            if agent_type.lower() == "mex":
                action, lstm_states = challenger.predict(
                    obs,
                    state=lstm_states,
                    episode_start=ep_starts,
                    deterministic=True,
                )
                ep_starts = np.zeros((1,), dtype=bool)
            else:
                masks = env.action_masks()
                action, _ = challenger.predict(obs, action_masks=masks, deterministic=True)

            obs, reward, terminated, truncated, info = env.step(int(action))
            round_steps += 1
            done = terminated or truncated

        won = info["won"]
        if won:
            wins += 1
        total_rounds += 1
        win_rate = (wins / total_rounds) * 100.0
        all_scores.append(info["challenger_score"])
        all_steps.append(round_steps)

        # Elo rating update
        elo_challenger, elo_defender = compute_elo(elo_challenger, elo_defender, won)

        outcome_str = "👑 WIN" if won else "💀 LOSS"
        avail_str = f"{info['availability'] * 100:.0f}%"

        # Log round telemetry
        if r <= 10 or r % 10 == 0 or won or r == max_rounds:
            print(
                f"{r:<6} | {outcome_str:<15} | {round_steps:<6} | "
                f"{info['challenger_score']:>10.1f} | {info['defender_score']:>9.1f} | "
                f"{win_rate:>8.1f}% | {elo_challenger:>6.0f} | {elo_defender:>6.0f} | {avail_str:>6}"
            )

        # Champion Checkpoint Saving on new performance threshold
        if win_rate > best_win_rate and total_rounds >= 15:
            best_win_rate = win_rate
            challenger.save(champion_path)

        # Early stopping checks
        if total_rounds >= 25 and win_rate >= target_score:
            print(f"\n[!] Target Win-Rate Threshold ({target_score:.1f}%) Achieved! Early stopping triggered.")
            challenger.save(champion_path)
            break

        if time_limit and (time.time() - start_time) >= time_limit:
            print(f"\n[!] Time Limit ({time_limit:.1f}s) Exceeded! Stopping sparring session.")
            challenger.save(champion_path)
            break

    # Save final champion checkpoint
    challenger.save(champion_path)
    elapsed = time.time() - start_time

    print("==========================================================================================")
    print(" [Arena Summary] Sparring Session Complete!")
    print(f" Total Rounds Sparred    : {total_rounds}")
    print(f" Final Challenger Win Rate: {win_rate:.1f}% ({wins}/{total_rounds})")
    print(f" Mean Steps to Victory    : {np.mean(all_steps):.2f}")
    print(f" Mean Challenger Score    : {np.mean(all_scores):.2f}")
    print(f" Final Challenger Elo     : {elo_challenger:.0f} (Delta: {elo_challenger - 1200.0:+.0f})")
    print(f" Final Defender Elo       : {elo_defender:.0f} (Delta: {elo_defender - 1200.0:+.0f})")
    print(f" Session Wall Time        : {elapsed:.2f}s ({total_rounds / elapsed:.1f} rounds/s)")
    print(f"[+] Saved Battle-Hardened Champion to: {champion_path}")
    print("==========================================================================================")

    return {
        "agent": agent_type,
        "rounds": total_rounds,
        "win_rate": win_rate,
        "mean_steps": float(np.mean(all_steps)),
        "mean_score": float(np.mean(all_scores)),
        "elo": elo_challenger,
        "champion_path": champion_path,
    }


def main():
    parser = argparse.ArgumentParser(description="Adversarial Arena Sparring Gym")
    parser.add_argument("--agent", choices=["gex", "mex"], default="mex", help="Challenger agent model")
    parser.add_argument("--max-rounds", type=int, default=200, help="Maximum sparring rounds")
    parser.add_argument("--target-score", type=float, default=85.0, help="Target win-rate percentage threshold")
    parser.add_argument("--time-limit", type=float, default=None, help="Session time limit in seconds")
    parser.add_argument("--save-dir", type=str, default="arena", help="Directory to save champion checkpoints")
    args = parser.parse_args()

    run_arena(
        agent_type=args.agent,
        max_rounds=args.max_rounds,
        target_score=args.target_score,
        time_limit=args.time_limit,
        save_dir=args.save_dir,
    )


if __name__ == "__main__":
    main()
