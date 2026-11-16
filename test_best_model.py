#!/usr/bin/env python3
"""
Test script to run deterministic evaluation using privesc_maskable_ppo_best.zip.
"""

from sb3_contrib import MaskablePPO
from env_v3 import LinuxPrivEscEnvV3, ALL_NODES, VULN_NODES

def main():
    print("Loading optimal model: privesc_maskable_ppo_best.zip ...")
    model = MaskablePPO.load("privesc_maskable_ppo_best")
    env = LinuxPrivEscEnvV3(step_penalty=-3.0, discovery_reward=2.0, chain_bonus=5.0, completion_reward=50.0)

    obs, _ = env.reset(seed=777)
    print(f"\n[Target Initialized]")
    print(f"  Real Target Vulnerabilities : {list(env.real_vulns)}")
    print(f"  Discoverable Findings       : {list(env.real_findings)}")
    print("-" * 65)

    total_reward = 0.0
    for step in range(1, 20):
        masks = env.action_masks()
        action, _ = model.predict(obs, action_masks=masks, deterministic=True)
        action = int(action)

        if action < 10:
            act_type = "CHECK"
            target_name = ALL_NODES[action]
        else:
            act_type = "EXPLOIT"
            target_name = VULN_NODES[action - 10]

        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        root_status = "👑 ROOT COMPROMISED!" if obs[-1] == 1.0 else "In Progress"
        print(f"Step {step:02d} | Action: {act_type:<7} -> {target_name:<14} | Reward: {reward:>6.2f} | Status: {root_status}")

        if terminated or truncated:
            break

    print("-" * 65)
    print(f"Episode Completed in {step} steps. Cumulative Reward: {total_reward:.2f}")

if __name__ == "__main__":
    main()
