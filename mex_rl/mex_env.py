"""
==============================================================================
MEx-RL Environment: Upgraded Strategic POMDP with REx-RL Context Handoff
46-dim Observation Vector (42 graph dims + 4 REx context dims), Recurrent LSTM.
==============================================================================
"""

import random
from typing import Dict, List, Optional, Set, Tuple

import gymnasium as gym
from gymnasium import spaces
import numpy as np

DOMAIN_NAMES = [
    "FilePermissions",
    "ServiceDaemons",
    "ScheduledTasks",
    "IdentityGroups",
]
N_DOMAINS = len(DOMAIN_NAMES)  # 4

DOMAINS: Dict[str, List[str]] = {
    "FilePermissions": ["suid_bin", "passwd_write", "cap_exec", "shadow_read"],
    "ServiceDaemons": ["service_write", "db_leak", "rpc_priv", "daemon_hijack"],
    "ScheduledTasks": ["cron_job", "timer_unit", "logrotate_pivot", "batch_script"],
    "IdentityGroups": ["docker_group", "sudo_nopasswd", "lxd_group", "admin_token"],
}

NODE_NAMES: List[str] = (
    DOMAINS["FilePermissions"]
    + DOMAINS["ServiceDaemons"]
    + DOMAINS["ScheduledTasks"]
    + DOMAINS["IdentityGroups"]
)
N_NODES = len(NODE_NAMES)  # 16
N_ACTIONS = N_DOMAINS      # 4 macro-actions
CONTEXT_DIM = 4            # [entry_service_type, anomaly_confidence, noise_level, privilege_tier]
OBS_DIM = 2 * N_NODES + N_DOMAINS + N_DOMAINS + 1 + 1 + CONTEXT_DIM  # 46 dims


class MExGymEnv(gym.Env):
    """Hierarchical Strategic POMDP Environment accepting REx-RL Context Handoff."""

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        step_penalty: float = -0.5,
        probe_reward: float = 6.0,
        unlock_reward: float = 15.0,
        resolve_reward: float = 20.0,
        completion_reward: float = 60.0,
        max_steps: int = 30,
        telemetry_noise_std: float = 0.05,
    ):
        super().__init__()
        self.step_penalty = float(step_penalty)
        self.probe_reward = float(probe_reward)
        self.unlock_reward = float(unlock_reward)
        self.resolve_reward = float(resolve_reward)
        self.completion_reward = float(completion_reward)
        self.max_steps = max_steps
        self.noise_std = telemetry_noise_std

        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(OBS_DIM,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(N_ACTIONS)

        self.state = np.zeros(OBS_DIM, dtype=np.float32)
        self.real_vulns: Set[int] = set()
        self.unlocked_domains: Set[int] = set()
        self.resolved_domains: Set[int] = set()
        self.target_d = 1
        self.steps = 0
        self.context_vector = np.zeros(CONTEXT_DIM, dtype=np.float32)

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
        context_vector: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # Context vector from REx-RL handoff or options
        if context_vector is not None:
            self.context_vector = np.clip(np.array(context_vector, dtype=np.float32), 0.0, 1.0)
        elif options and "context_vector" in options:
            self.context_vector = np.clip(np.array(options["context_vector"], dtype=np.float32), 0.0, 1.0)
        else:
            suggested = random.choice([1, 2])
            entry_type = suggested / 3.0
            self.context_vector = np.array([entry_type, 0.85, 0.05, 0.33], dtype=np.float32)

        # Identify target domain suggested by REx entry context (1 or 2)
        target_cand = int(round(float(self.context_vector[0]) * 3.0))
        self.target_d = target_cand if target_cand in [1, 2] else random.choice([1, 2])

        # Populate a solvable sequential chain from domain 0 up to target_d
        self.real_vulns = set()
        for d in range(self.target_d + 1):
            self.real_vulns.add(random.randint(d * 4, d * 4 + 3))

        self.unlocked_domains = {0}
        self.resolved_domains = set()
        self.steps = 0

        self.state = np.zeros(OBS_DIM, dtype=np.float32)
        # Domain 0 unlocked flag
        self.state[32] = 1.0

        # Noisy domain telemetry: dims [36..39]
        for d in range(N_DOMAINS):
            domain_nodes = set(range(d * 4, (d + 1) * 4))
            count = len(domain_nodes.intersection(self.real_vulns))
            noisy_val = np.clip((count / 4.0) + np.random.normal(0, self.noise_std), 0.0, 1.0)
            self.state[36 + d] = float(noisy_val)

        # REx-RL Context vector: dims [42..45]
        self.state[42:46] = self.context_vector

        return self.state.copy(), {"context_vector": self.context_vector}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        self.steps += 1
        reward = self.step_penalty
        terminated = False
        truncated = False
        d = int(action)

        if d not in self.unlocked_domains:
            reward += -3.0
        elif d in self.resolved_domains:
            reward += -2.0
        else:
            domain_nodes = list(range(d * 4, (d + 1) * 4))
            ready = [n for n in domain_nodes if self.state[2 * n] == 1.0 and self.state[2 * n + 1] == 1.0]

            if ready:
                target = ready[0]
                self.real_vulns.discard(target)
                self.state[2 * target + 1] = 0.0
                self.resolved_domains.add(d)
                reward += self.resolve_reward

                # Sequential unlock
                next_d = d + 1
                if next_d < N_DOMAINS and next_d not in self.unlocked_domains:
                    self.unlocked_domains.add(next_d)
                    self.state[32 + next_d] = 1.0
                    reward += self.unlock_reward

                # Multi-domain chain compromise achieved once target_d is resolved
                if self.target_d in self.resolved_domains:
                    reward += self.completion_reward
                    self.state[41] = 1.0
                    terminated = True
            else:
                unprobed = [n for n in domain_nodes if self.state[2 * n] == 0.0]
                if unprobed:
                    target = unprobed[0]
                    self.state[2 * target] = 1.0
                    if target in self.real_vulns:
                        self.state[2 * target + 1] = 1.0
                        reward += self.probe_reward
                    else:
                        self.state[2 * target + 1] = 0.0
                        reward += 1.0

        self.state[40] = float(self.steps / self.max_steps)
        if self.steps >= self.max_steps:
            truncated = True

        return self.state.copy(), reward, terminated, truncated, {"won": bool(self.state[41] == 1.0)}
