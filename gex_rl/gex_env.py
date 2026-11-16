"""
==============================================================================
GEx-RL Environment: Fast Tactical Exploration & Execution RL
Batched PyTorch CUDA Vectorized State-Machine Environment + Gymnasium Wrapper.
==============================================================================
"""

import random
from typing import Dict, List, Optional, Set, Tuple

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch

RECON_NODES: List[str] = ["enum_users", "config_leak"]
VULN_NODES: List[str] = [
    "suid", "sudo", "cron", "path_hijack",
    "capabilities", "passwd_write", "service_write", "docker_group"
]
ALL_NODES: List[str] = RECON_NODES + VULN_NODES
N_NODES: int = len(ALL_NODES)  # 10
N_VULNS: int = len(VULN_NODES)  # 8
N_ACTIONS: int = N_NODES + N_VULNS  # 18
OBS_DIM: int = 2 * N_NODES + 1  # 21

PREREQUISITES: Dict[str, List[str]] = {
    "enum_users": [],
    "config_leak": [],
    "suid": [],
    "sudo": [],
    "cron": [],
    "path_hijack": ["enum_users"],
    "capabilities": [],
    "passwd_write": ["config_leak"],
    "service_write": ["enum_users", "config_leak"],
    "docker_group": [],
}


class GExGPUEnv:
    """High-Throughput PyTorch CUDA Vectorized State-Machine Environment."""

    def __init__(
        self,
        num_envs: int = 4096,
        device: str = "cuda",
        step_penalty: float = -3.0,
        discovery_reward: float = 2.0,
        chain_bonus: float = 5.0,
        completion_reward: float = 50.0,
        max_steps: int = 40,
    ):
        self.num_envs = num_envs
        self.device = torch.device(device if (device == "cuda" and torch.cuda.is_available()) else "cpu")
        self.step_penalty = step_penalty
        self.discovery_reward = discovery_reward
        self.chain_bonus = chain_bonus
        self.completion_reward = completion_reward
        self.max_steps = max_steps

        # GPU State Tensors
        self.obs = torch.zeros((self.num_envs, OBS_DIM), device=self.device, dtype=torch.float32)
        self.real_vulns = torch.zeros((self.num_envs, N_VULNS), device=self.device, dtype=torch.bool)
        self.real_findings = torch.zeros((self.num_envs, len(RECON_NODES)), device=self.device, dtype=torch.bool)
        self.steps = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)

        self.reset_all()

    def reset_all(self):
        all_ids = torch.arange(self.num_envs, device=self.device)
        self.reset(all_ids)

    def reset(self, env_ids: torch.Tensor):
        n = len(env_ids)
        if n == 0:
            return

        self.obs[env_ids] = 0.0
        self.steps[env_ids] = 0

        # Sample 1 or 2 real vulnerabilities per host
        k = torch.randint(1, 3, (n,), device=self.device)
        self.real_vulns[env_ids] = False

        rand_scores = torch.rand((n, N_VULNS), device=self.device)
        ranks = torch.argsort(rand_scores, dim=1)
        for i in range(n):
            self.real_vulns[env_ids[i], ranks[i, :k[i]]] = True

        # Recon findings
        self.real_findings[env_ids, 0] = True  # enum_users always present
        dep_vulns = self.real_vulns[env_ids, 5] | self.real_vulns[env_ids, 6]
        rand_leak = torch.rand(n, device=self.device) < 0.75
        self.real_findings[env_ids, 1] = dep_vulns | rand_leak

    def get_action_masks(self) -> torch.Tensor:
        masks = torch.zeros((self.num_envs, N_ACTIONS), device=self.device, dtype=torch.bool)
        not_terminated = self.obs[:, -1] == 0.0

        # Check actions (0..9)
        masks[:, 0] = (self.obs[:, 0] == 0.0) & not_terminated
        masks[:, 1] = (self.obs[:, 2] == 0.0) & not_terminated
        masks[:, 2] = (self.obs[:, 4] == 0.0) & not_terminated
        masks[:, 3] = (self.obs[:, 6] == 0.0) & not_terminated
        masks[:, 4] = (self.obs[:, 8] == 0.0) & not_terminated
        masks[:, 6] = (self.obs[:, 12] == 0.0) & not_terminated
        masks[:, 9] = (self.obs[:, 18] == 0.0) & not_terminated

        # Prerequisite checks
        enum_ok = (self.obs[:, 0] == 1.0) & (self.obs[:, 1] == 1.0)
        config_ok = (self.obs[:, 2] == 1.0) & (self.obs[:, 3] == 1.0)

        masks[:, 5] = (self.obs[:, 10] == 0.0) & enum_ok & not_terminated
        masks[:, 7] = (self.obs[:, 14] == 0.0) & config_ok & not_terminated
        masks[:, 8] = (self.obs[:, 16] == 0.0) & enum_ok & config_ok & not_terminated

        # Exploit actions (10..17)
        for v in range(N_VULNS):
            node_idx = len(RECON_NODES) + v
            act_idx = 10 + v
            is_real = self.obs[:, 2 * node_idx + 1] == 1.0
            masks[:, act_idx] = is_real & self.real_vulns[:, v] & not_terminated

        # Fallback to avoid empty mask
        deadlocks = (~masks.any(dim=1)) & not_terminated
        if deadlocks.any():
            masks[deadlocks, 0] = True

        return masks

    def step(self, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self.steps += 1
        rewards = torch.full((self.num_envs,), self.step_penalty, device=self.device, dtype=torch.float32)

        # 1. Checks (< 10)
        check_mask = actions < 10
        if check_mask.any():
            check_envs = torch.where(check_mask)[0]
            check_acts = actions[check_envs]

            recon_acts = check_acts < 2
            if recon_acts.any():
                r_envs = check_envs[recon_acts]
                r_acts = check_acts[recon_acts]
                self.obs[r_envs, 2 * r_acts] = 1.0
                is_real = self.real_findings[r_envs, r_acts].float()
                self.obs[r_envs, 2 * r_acts + 1] = is_real
                rewards[r_envs] += is_real * self.discovery_reward

            vuln_acts = check_acts >= 2
            if vuln_acts.any():
                v_envs = check_envs[vuln_acts]
                v_acts = check_acts[vuln_acts]
                v_idx = v_acts - 2
                self.obs[v_envs, 2 * v_acts] = 1.0
                is_real = self.real_vulns[v_envs, v_idx].float()
                self.obs[v_envs, 2 * v_acts + 1] = is_real
                rewards[v_envs] += is_real * self.discovery_reward

                dep_mask = (v_acts == 5) | (v_acts == 7) | (v_acts == 8)
                rewards[v_envs[dep_mask]] += is_real[dep_mask] * self.chain_bonus

        # 2. Exploits (>= 10)
        exploit_mask = actions >= 10
        if exploit_mask.any():
            exp_envs = torch.where(exploit_mask)[0]
            exp_acts = actions[exp_envs]
            v_idx = exp_acts - 10
            node_idx = 2 + v_idx

            valid_exp = (self.obs[exp_envs, 2 * node_idx + 1] == 1.0) & self.real_vulns[exp_envs, v_idx]
            valid_envs = exp_envs[valid_exp]
            valid_v = v_idx[valid_exp]
            valid_node = node_idx[valid_exp]

            if len(valid_envs) > 0:
                rewards[valid_envs] += 20.0
                self.real_vulns[valid_envs, valid_v] = False
                self.obs[valid_envs, 2 * valid_node + 1] = 0.0

                cleared = self.real_vulns[valid_envs].sum(dim=1) == 0
                cleared_envs = valid_envs[cleared]
                rewards[cleared_envs] += self.completion_reward
                self.obs[cleared_envs, -1] = 1.0

            invalid_envs = exp_envs[~valid_exp]
            rewards[invalid_envs] += -10.0

        terminated = self.obs[:, -1] == 1.0
        truncated = self.steps >= self.max_steps
        dones = terminated | truncated

        done_ids = torch.where(dones)[0]
        if len(done_ids) > 0:
            self.reset(done_ids)

        return self.obs.clone(), rewards, dones


class GExGymEnv(gym.Env):
    """Gymnasium Wrapper for GEx-RL Environment with Action Masking."""

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        step_penalty: float = -3.0,
        discovery_reward: float = 2.0,
        chain_bonus: float = 5.0,
        completion_reward: float = 50.0,
        max_steps: int = 40,
    ):
        super().__init__()
        self.step_penalty = float(step_penalty)
        self.discovery_reward = float(discovery_reward)
        self.chain_bonus = float(chain_bonus)
        self.completion_reward = float(completion_reward)
        self.max_steps = max_steps

        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(OBS_DIM,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(N_ACTIONS)

        self.state = np.zeros(OBS_DIM, dtype=np.float32)
        self.real_vulns: Set[str] = set()
        self.real_findings: Set[str] = set()
        self.steps = 0

    def reset(
        self, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        k = random.randint(1, 2)
        self.real_vulns = set(random.sample(VULN_NODES, k))

        self.real_findings = {"enum_users"}
        if "service_write" in self.real_vulns or "passwd_write" in self.real_vulns or random.random() < 0.75:
            self.real_findings.add("config_leak")

        self.state = np.zeros(OBS_DIM, dtype=np.float32)
        self.steps = 0
        return self.state.copy(), {}

    def action_masks(self) -> np.ndarray:
        mask = np.zeros(self.action_space.n, dtype=bool)
        if self.state[-1] == 1.0:
            return mask

        # Check actions (0..9)
        for idx, node_name in enumerate(ALL_NODES):
            checked_offset = 2 * idx
            if self.state[checked_offset] == 1.0:
                continue

            prereqs_met = True
            for req in PREREQUISITES[node_name]:
                req_idx = ALL_NODES.index(req)
                if self.state[2 * req_idx] == 0.0 or self.state[2 * req_idx + 1] == 0.0:
                    prereqs_met = False
                    break

            if prereqs_met:
                mask[idx] = True

        # Exploit actions (10..17)
        for v_idx, v_name in enumerate(VULN_NODES):
            node_idx = len(RECON_NODES) + v_idx
            is_real_offset = 2 * node_idx + 1
            act_idx = N_NODES + v_idx

            if self.state[is_real_offset] == 1.0 and v_name in self.real_vulns:
                mask[act_idx] = True

        if not np.any(mask):
            mask[0] = True

        return mask

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        self.steps += 1
        reward = self.step_penalty
        terminated = False
        truncated = False
        action = int(action)

        if action < N_NODES:
            node_name = ALL_NODES[action]
            checked_offset = 2 * action
            is_real_offset = 2 * action + 1

            if self.state[checked_offset] == 1.0:
                reward += -2.0
            else:
                prereqs_met = all(
                    self.state[2 * ALL_NODES.index(req)] == 1.0 and self.state[2 * ALL_NODES.index(req) + 1] == 1.0
                    for req in PREREQUISITES[node_name]
                )
                if not prereqs_met:
                    reward += -5.0
                else:
                    self.state[checked_offset] = 1.0
                    if node_name in RECON_NODES:
                        is_real = 1.0 if node_name in self.real_findings else 0.0
                        self.state[is_real_offset] = is_real
                        if is_real == 1.0:
                            reward += self.discovery_reward
                    else:
                        is_real = 1.0 if node_name in self.real_vulns else 0.0
                        self.state[is_real_offset] = is_real
                        if is_real == 1.0:
                            reward += self.discovery_reward
                            if len(PREREQUISITES[node_name]) > 0:
                                reward += self.chain_bonus

        else:
            v_idx = action - N_NODES
            v_name = VULN_NODES[v_idx]
            node_idx = len(RECON_NODES) + v_idx
            checked_offset = 2 * node_idx
            is_real_offset = 2 * node_idx + 1

            if self.state[checked_offset] == 1.0 and self.state[is_real_offset] == 1.0:
                reward += 20.0
                self.real_vulns.discard(v_name)
                self.state[is_real_offset] = 0.0
                if len(self.real_vulns) == 0:
                    reward += self.completion_reward
                    self.state[-1] = 1.0
                    terminated = True
            else:
                reward += -10.0

        if self.steps >= self.max_steps:
            truncated = True

        return self.state.copy(), reward, terminated, truncated, {}
