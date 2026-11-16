"""
==============================================================================
Adversarial Arena: Combat Environment & Adaptive Defender Policy
Dynamic competitive battleground enforcing availability >= 0.85 constraint.
==============================================================================
"""

import random
from typing import Dict, List, Optional, Set, Tuple

import gymnasium as gym
from gymnasium import spaces
import numpy as np

# 16-node multi-domain POMDP definitions
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
        self.challenger_history: List[int] = []

    def reset(self):
        self.hardened_nodes = set()
        self.decoy_nodes = set()
        # Random initial defense allocation
        n_actions = random.randint(1, self.max_hardened_nodes)
        chosen = random.sample(range(self.n_nodes), n_actions)
        if chosen:
            self.hardened_nodes.add(chosen[0])
            if len(chosen) > 1:
                self.decoy_nodes.add(chosen[1])

    def adapt(self, challenger_action: int):
        self.challenger_history.append(challenger_action)
        # Adapt: with 30% probability, relocate defense to protect the active domain
        if random.random() < 0.35:
            d = challenger_action % 4
            domain_nodes = list(range(d * 4, (d + 1) * 4))
            self.hardened_nodes = set(random.sample(domain_nodes, 1))

    def get_availability(self) -> float:
        total_affected = len(self.hardened_nodes.union(self.decoy_nodes))
        return 1.0 - (total_affected / float(self.n_nodes))


class CombatArenaEnv(gym.Env):
    """Competitive POMDP Arena: Challenger Policy vs. Adaptive Defender Policy."""

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

        if self.agent_type == "gex":
            # GEx tactical interface: 21 dims, 18 actions
            self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(21,), dtype=np.float32)
            self.action_space = spaces.Discrete(18)
        else:
            # MEx strategic interface: 42 dims, 4 macro-actions
            self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(42,), dtype=np.float32)
            self.action_space = spaces.Discrete(N_DOMAINS)

        self.state = None
        self.real_vulns: Set[int] = set()
        self.unlocked_domains: Set[int] = set()
        self.resolved_domains: Set[int] = set()
        self.steps = 0

        self.challenger_score = 0.0
        self.defender_score = 0.0
        self.decoy_triggered = False

    def reset(
        self, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.defender.reset()
        self.challenger_score = 0.0
        self.defender_score = 0.0
        self.decoy_triggered = False
        self.steps = 0

        if self.agent_type == "gex":
            # 21-dim tactical state
            self.state = np.zeros(21, dtype=np.float32)
            k = random.randint(1, 2)
            self.real_vulns = set(random.sample(range(8), k))
        else:
            # 42-dim strategic POMDP state
            d0_v = random.randint(0, 3)
            d1_v = random.randint(4, 7) if random.random() < 0.8 else random.randint(8, 11)
            self.real_vulns = {d0_v, d1_v}
            self.unlocked_domains = {0}
            self.resolved_domains = set()

            self.state = np.zeros(42, dtype=np.float32)
            self.state[32] = 1.0  # domain 0 unlocked

            # Noisy telemetry influenced by defender obfuscation
            for d in range(N_DOMAINS):
                domain_nodes = set(range(d * 4, (d + 1) * 4))
                count = len(domain_nodes.intersection(self.real_vulns))
                # If defender hardened this domain, add obfuscation noise
                noise = 0.15 if any(n in self.defender.hardened_nodes for n in domain_nodes) else 0.05
                noisy_val = np.clip((count / 4.0) + np.random.normal(0, noise), 0.0, 1.0)
                self.state[36 + d] = float(noisy_val)

        return self.state.copy(), {}

    def action_masks(self) -> np.ndarray:
        """Compatible with GEx ActionMasker if using tactical agent."""
        if self.agent_type == "gex":
            mask = np.zeros(18, dtype=bool)
            if self.state[-1] == 1.0:
                return mask
            # Standalone and available checks
            for i in range(10):
                if self.state[2 * i] == 0.0:
                    mask[i] = True
            for v in range(8):
                if self.state[2 * (2 + v) + 1] == 1.0:
                    mask[10 + v] = True
            if not np.any(mask):
                mask[0] = True
            return mask
        else:
            return np.ones(N_DOMAINS, dtype=bool)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        self.steps += 1
        action = int(action)

        # Defender adapts to Challenger move
        self.defender.adapt(action)
        self.defender_score += 1.0  # Time delay reward for defender

        terminated = False
        truncated = False
        reward = -0.5

        if self.agent_type == "gex":
            # Tactical GEx logic against defender
            if action < 10:
                node_idx = action
                if node_idx in self.defender.decoy_nodes:
                    # Triggered decoy!
                    reward += -5.0
                    self.defender_score += 15.0
                    self.decoy_triggered = True
                elif node_idx in self.defender.hardened_nodes:
                    # Hardened node: requires re-inspection
                    reward += -1.5
                    self.defender_score += 5.0
                else:
                    self.state[2 * node_idx] = 1.0
                    if (node_idx - 2) in self.real_vulns:
                        self.state[2 * node_idx + 1] = 1.0
                        reward += 5.0
            else:
                v_idx = action - 10
                node_idx = 2 + v_idx
                if self.state[2 * node_idx + 1] == 1.0 and v_idx in self.real_vulns:
                    reward += 20.0
                    self.real_vulns.discard(v_idx)
                    self.state[2 * node_idx + 1] = 0.0
                    if len(self.real_vulns) == 0:
                        reward += 60.0
                        self.state[-1] = 1.0
                        terminated = True

        else:
            # Strategic MEx logic against defender
            d = action
            if d not in self.unlocked_domains:
                reward += -3.0
                self.defender_score += 3.0
            elif d in self.resolved_domains:
                reward += -2.0
            else:
                domain_nodes = list(range(d * 4, (d + 1) * 4))
                # Check for decoy
                if any(n in self.defender.decoy_nodes for n in domain_nodes) and not self.decoy_triggered:
                    reward += -4.0
                    self.defender_score += 12.0
                    self.decoy_triggered = True

                ready = [n for n in domain_nodes if self.state[2 * n] == 1.0 and self.state[2 * n + 1] == 1.0]
                if ready:
                    target = ready[0]
                    self.real_vulns.discard(target)
                    self.state[2 * target + 1] = 0.0
                    self.resolved_domains.add(d)
                    reward += 20.0

                    next_d = d + 1
                    if next_d < N_DOMAINS and next_d not in self.unlocked_domains:
                        self.unlocked_domains.add(next_d)
                        self.state[32 + next_d] = 1.0
                        reward += 15.0

                    if len(self.resolved_domains) >= 2 and len(self.real_vulns) == 0:
                        reward += 60.0
                        self.state[41] = 1.0
                        terminated = True
                else:
                    unprobed = [n for n in domain_nodes if self.state[2 * n] == 0.0]
                    if unprobed:
                        target = unprobed[0]
                        # If node is hardened by defender, higher probe friction
                        if target in self.defender.hardened_nodes:
                            reward += -1.0
                            self.defender_score += 4.0

                        self.state[2 * target] = 1.0
                        if target in self.real_vulns:
                            self.state[2 * target + 1] = 1.0
                            reward += 5.0
                        else:
                            self.state[2 * target + 1] = 0.0
                            reward += 1.0

            self.state[40] = float(self.steps / self.max_steps)

        self.challenger_score += reward

        if self.steps >= self.max_steps:
            truncated = True
            if not terminated:
                # Defender gets bonus for running out Challenger clock
                self.defender_score += 40.0

        info = {
            "challenger_score": self.challenger_score,
            "defender_score": self.defender_score,
            "availability": self.defender.get_availability(),
            "won": bool(terminated),
        }

        return self.state.copy(), reward, terminated, truncated, info
