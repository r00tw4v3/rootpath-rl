"""
==============================================================================
Adversarial Arena: Unified Combat Environment & Adaptive Defender Policy
Dynamic competitive battleground supporting REx, MEx, and GEx with Defender constraint.
==============================================================================
"""

import random
from typing import Dict, List, Optional, Set, Tuple

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from gex_rl.gex_env import GExGymEnv
from mex_rl.mex_env import MExGymEnv
from rex_rl.rex_env import RExGymEnv

DOMAIN_NAMES = [
    "FilePermissions",
    "ServiceDaemons",
    "ScheduledTasks",
    "IdentityGroups",
]
N_DOMAINS = 4
N_NODES = 16


class DefenderPolicy:
    """Adaptive Defender allocating obfuscation & decoy budget while maintaining availability >= 0.85."""

    def __init__(self, n_nodes: int = N_NODES, min_availability: float = 0.85):
        self.n_nodes = n_nodes
        self.min_availability = min_availability
        self.max_hardened_nodes = int(np.floor(n_nodes * (1.0 - min_availability)))  # max 2 nodes

        self.hardened_nodes: Set[int] = set()
        self.decoy_nodes: Set[int] = set()
        self.waf_rate_limit_active: bool = False
        self.challenger_history: List[int] = []

    def reset(self):
        self.hardened_nodes = set()
        self.decoy_nodes = set()
        self.waf_rate_limit_active = random.random() < 0.20

        n_actions = random.randint(1, self.max_hardened_nodes)
        chosen = random.sample(range(self.n_nodes), n_actions)
        if chosen:
            self.hardened_nodes.add(chosen[0])
            if len(chosen) > 1:
                self.decoy_nodes.add(chosen[1])

    def adapt(self, challenger_action: int):
        self.challenger_history.append(challenger_action)
        if random.random() < 0.30:
            d = challenger_action % 4
            domain_nodes = list(range(d * 4, (d + 1) * 4))
            self.hardened_nodes = set(random.sample(domain_nodes, 1))

    def get_availability(self) -> float:
        total_affected = len(self.hardened_nodes.union(self.decoy_nodes))
        return 1.0 - (total_affected / float(self.n_nodes))


class CombatArenaEnv(gym.Env):
    """Unified Competitive Multi-Agent Arena: REx, MEx, or GEx vs. Adaptive Defender."""

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        agent_type: str = "mex",
        max_steps: int = 30,
        defender_difficulty: float = 1.0,
    ):
        super().__init__()
        self.agent_type = agent_type.lower()
        self.max_steps = max_steps
        self.difficulty = defender_difficulty

        self.defender = DefenderPolicy(n_nodes=N_NODES, min_availability=0.85)

        if self.agent_type == "rex":
            self.inner_env = RExGymEnv(max_steps=max_steps)
        elif self.agent_type == "gex":
            self.inner_env = GExGymEnv(max_steps=max_steps)
        else:
            self.inner_env = MExGymEnv(max_steps=max_steps)

        self.observation_space = self.inner_env.observation_space
        self.action_space = self.inner_env.action_space

        self.challenger_score = 0.0
        self.defender_score = 0.0
        self.steps = 0

    def reset(
        self, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self.defender.reset()
        self.challenger_score = 0.0
        self.defender_score = 0.0
        self.steps = 0

        obs, info = self.inner_env.reset(seed=seed, options=options)

        # Apply defender initial environmental friction
        if self.agent_type == "rex" and self.defender.waf_rate_limit_active:
            obs[21] = min(1.0, float(obs[21] + 0.25))  # WAF rate limit risk

        return obs, info

    def action_masks(self) -> np.ndarray:
        if hasattr(self.inner_env, "action_masks"):
            return self.inner_env.action_masks()
        return np.ones(self.action_space.n, dtype=bool)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        self.steps += 1
        action = int(action)

        self.defender.adapt(action)
        self.defender_score += 1.0  # Time delay reward for defender

        obs, reward, terminated, truncated, info = self.inner_env.step(action)

        # Adversarial Defender Perturbations
        if self.agent_type == "rex":
            if self.defender.waf_rate_limit_active:
                obs[21] = min(1.0, float(obs[21] + 0.10))
                if obs[21] > 0.85:
                    reward -= 5.0
                    self.defender_score += 8.0

        elif self.agent_type == "gex":
            # Decoy penalty if hitting hardened node
            if action < 10 and action in self.defender.decoy_nodes:
                reward -= 4.0
                self.defender_score += 12.0

        else:
            # MEx: if commanding domain with hardened node
            d = action
            domain_nodes = set(range(d * 4, (d + 1) * 4))
            if any(n in self.defender.hardened_nodes for n in domain_nodes):
                reward -= 1.0
                self.defender_score += 3.0

        self.challenger_score += reward

        if truncated and not terminated:
            self.defender_score += 35.0

        info["challenger_score"] = self.challenger_score
        info["defender_score"] = self.defender_score
        info["availability"] = self.defender.get_availability()
        info["won"] = bool(terminated)

        return obs, reward, terminated, truncated, info
