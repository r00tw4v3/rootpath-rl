"""
==============================================================================
Linux Privilege Escalation Gym Environment (V3)
DAG-Based Multi-Hop Prerequisite Simulator with SB3-Contrib Action Masking.

Architecture:
  - 10 Graph Nodes:
      0: enum_users     (Reconnaissance stage 1)
      1: config_leak    (Reconnaissance stage 2)
      2: suid           (Standalone Vuln)
      3: sudo           (Standalone Vuln)
      4: cron           (Standalone Vuln)
      5: path_hijack    (Dependent Vuln: requires enum_users)
      6: capabilities   (Standalone Vuln)
      7: passwd_write   (Dependent Vuln: requires config_leak)
      8: service_write  (Dependent Vuln: requires enum_users & config_leak)
      9: docker_group   (Standalone Vuln)

  - Action Space: Discrete(18)
      0..9:   CHECK nodes 0..9
      10..17: EXPLOIT vulnerability nodes (indices 2..9)

  - Observation Space: Box(low=0, high=1, shape=(21,), dtype=float32)
      For each of the 10 nodes: [checked (0/1), is_real (0/1)] = 20 dims
      + root_achieved flag (0/1) = 21 dims

  - SB3-Contrib Compatibility:
      Implements action_masks() returning np.ndarray(shape=(18,), dtype=bool).
==============================================================================
"""

import random
from typing import Dict, List, Optional, Set, Tuple

import gymnasium as gym
from gymnasium import spaces
import numpy as np

RECON_NODES: List[str] = ["enum_users", "config_leak"]
VULN_NODES: List[str] = [
    "suid",
    "sudo",
    "cron",
    "path_hijack",
    "capabilities",
    "passwd_write",
    "service_write",
    "docker_group",
]
ALL_NODES: List[str] = RECON_NODES + VULN_NODES  # 10 nodes total
N_NODES: int = len(ALL_NODES)
N_VULNS: int = len(VULN_NODES)
N_ACTIONS: int = N_NODES + N_VULNS  # 10 checks + 8 exploits = 18

# Multi-Step Dependency Graph (DAG) Prerequisites
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


class LinuxPrivEscEnvV3(gym.Env):
    """Next-Gen Linux Privilege Escalation DAG Environment with Action Masking."""

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        step_penalty: float = -0.5,
        discovery_reward: float = 2.0,
        chain_bonus: float = 5.0,
        completion_reward: float = 50.0,
        n_real_vulns: Tuple[int, int] = (1, 2),
        max_steps: int = 40,
    ):
        super().__init__()
        self.step_penalty = float(step_penalty)
        self.discovery_reward = float(discovery_reward)
        self.chain_bonus = float(chain_bonus)
        self.completion_reward = float(completion_reward)
        self.n_real_vulns = n_real_vulns
        self.max_steps = max_steps

        # 2 dims per node (checked, is_real) + 1 dim for root_achieved
        self.obs_dim = 2 * N_NODES + 1
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(N_ACTIONS)

        self.state = np.zeros(self.obs_dim, dtype=np.float32)
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

        # Sample active target vulnerabilities for this host
        k = random.randint(*self.n_real_vulns)
        self.real_vulns = set(random.sample(VULN_NODES, k))

        # Recon findings setup:
        # enum_users is universally discoverable on Linux targets
        self.real_findings = {"enum_users"}

        # config_leak is discoverable if dependent vulns exist, or with 75% random chance
        if "service_write" in self.real_vulns or "passwd_write" in self.real_vulns:
            self.real_findings.add("config_leak")
        else:
            if random.random() < 0.75:
                self.real_findings.add("config_leak")

        self.state = np.zeros(self.obs_dim, dtype=np.float32)
        self.steps = 0

        return self.state.copy(), {}

    def action_masks(self) -> np.ndarray:
        """Returns boolean mask of valid actions for the current state.

        Compatible with sb3_contrib.common.wrappers.ActionMasker.
        """
        mask = np.zeros(self.action_space.n, dtype=bool)

        # If root already achieved, terminate all actions
        if self.state[-1] == 1.0:
            return mask

        # 1. Evaluate CHECK actions (0..9)
        for idx, node_name in enumerate(ALL_NODES):
            checked_offset = 2 * idx
            # Rule: Cannot repeat check
            if self.state[checked_offset] == 1.0:
                continue

            # Rule: Multi-step DAG check - all prerequisites must be checked AND confirmed real
            prereqs_met = True
            for req in PREREQUISITES[node_name]:
                req_idx = ALL_NODES.index(req)
                req_checked = self.state[2 * req_idx]
                req_real = self.state[2 * req_idx + 1]
                if req_checked == 0.0 or req_real == 0.0:
                    prereqs_met = False
                    break

            if prereqs_met:
                mask[idx] = True

        # 2. Evaluate EXPLOIT actions (10..17)
        for v_idx, v_name in enumerate(VULN_NODES):
            node_idx = len(RECON_NODES) + v_idx
            is_real_offset = 2 * node_idx + 1
            act_idx = N_NODES + v_idx

            # Rule: Can only exploit if checked, confirmed real, and not yet exploited
            if self.state[is_real_offset] == 1.0 and v_name in self.real_vulns:
                mask[act_idx] = True

        # Fallback to prevent deadlocks: if no actions are available, allow un-checked root nodes
        if not np.any(mask):
            for idx, node_name in enumerate(ALL_NODES):
                if self.state[2 * idx] == 0.0:
                    mask[idx] = True
                    break

        return mask

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        self.steps += 1
        reward = self.step_penalty
        terminated = False
        truncated = False

        action = int(action)

        if action < N_NODES:
            # CHECK ACTION (0..9)
            node_idx = action
            node_name = ALL_NODES[node_idx]
            checked_offset = 2 * node_idx
            is_real_offset = 2 * node_idx + 1

            if self.state[checked_offset] == 1.0:
                # Invalid action: duplicate check penalty
                reward += -2.0
            else:
                # Verify DAG prerequisites
                prereqs_met = all(
                    self.state[2 * ALL_NODES.index(req)] == 1.0
                    and self.state[2 * ALL_NODES.index(req) + 1] == 1.0
                    for req in PREREQUISITES[node_name]
                )

                if not prereqs_met:
                    reward += -5.0  # Attempted check without prerequisite
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
                            # Grant chain bonus if this is a multi-hop dependent vulnerability
                            if len(PREREQUISITES[node_name]) > 0:
                                reward += self.chain_bonus

        else:
            # EXPLOIT ACTION (10..17)
            v_idx = action - N_NODES
            v_name = VULN_NODES[v_idx]
            node_idx = len(RECON_NODES) + v_idx
            checked_offset = 2 * node_idx
            is_real_offset = 2 * node_idx + 1

            if self.state[checked_offset] == 1.0 and self.state[is_real_offset] == 1.0:
                # Valid Exploit Execution
                reward += 20.0
                self.real_vulns.discard(v_name)
                self.state[is_real_offset] = 0.0  # Consumed exploit

                if len(self.real_vulns) == 0:
                    reward += self.completion_reward
                    self.state[-1] = 1.0  # Root flag set
                    terminated = True
            else:
                # Blind or invalid exploit attempt
                reward += -10.0

        if self.steps >= self.max_steps:
            truncated = True

        return self.state.copy(), reward, terminated, truncated, {}

    def render(self):
        status = {}
        for idx, name in enumerate(ALL_NODES):
            c = bool(self.state[2 * idx])
            r = bool(self.state[2 * idx + 1])
            status[name] = "REAL" if r else ("CHECKED_NONE" if c else "UNCHECKED")
        root = bool(self.state[-1])
        print(f"Step {self.steps} | Root: {root} | Real Vulns: {list(self.real_vulns)} | Nodes: {status}")
