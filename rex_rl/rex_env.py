"""
==============================================================================
REx-RL Environment: Pre-Access External Surface & Anomaly RL
Vectorized PyTorch CUDA Engine (RExGPUEnv) + Gymnasium Wrapper (RExGymEnv)
24-dim Observation Vector, 12 Discrete Actions, Dynamic Action Masking.
==============================================================================
"""

import random
from typing import Dict, List, Optional, Tuple

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch

OBS_DIM = 24
N_ACTIONS = 12
HTTP_PORTS = [0, 1, 4, 7]  # Indices for 80, 443, 8080, 8443


class RExGymEnv(gym.Env):
    """Gymnasium Environment for Pre-Access Surface Discovery & Anomaly Detection."""

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        step_penalty: float = -0.5,
        discovery_reward: float = 2.5,
        anomaly_reward: float = 12.0,
        handoff_reward: float = 50.0,
        rate_limit_penalty: float = -8.0,
        max_steps: int = 25,
    ):
        super().__init__()
        self.step_penalty = float(step_penalty)
        self.discovery_reward = float(discovery_reward)
        self.anomaly_reward = float(anomaly_reward)
        self.handoff_reward = float(handoff_reward)
        self.rate_limit_penalty = float(rate_limit_penalty)
        self.max_steps = max_steps

        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(OBS_DIM,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(N_ACTIONS)

        self.state = np.zeros(OBS_DIM, dtype=np.float32)
        self.steps = 0
        self.ground_truth = {}
        self.action_history = set()

    def reset(
        self, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.steps = 0
        self.action_history = set()
        self.state = np.zeros(OBS_DIM, dtype=np.float32)

        # Ground truth generation:
        # Guarantee at least 1 HTTP service open, plus 1 or 2 other services
        open_http = random.choice(HTTP_PORTS)
        other_candidates = [p for p in range(8) if p != open_http]
        extra_ports = random.sample(other_candidates, random.randint(1, 3))
        self.ground_truth["open_ports"] = set([open_http] + extra_ports)

        # Simulated vulnerabilities / anomalies on this host:
        self.ground_truth["has_auth_bypass"] = random.random() < 0.65
        self.ground_truth["has_upload_vuln"] = random.random() < 0.50
        self.ground_truth["has_api_leak"] = random.random() < 0.60
        self.ground_truth["has_boundary_5xx"] = random.random() < 0.70

        # Primary entry service type (0: web, 1: api, 2: auth, 3: daemon)
        self.ground_truth["entry_service"] = random.choice([0, 1, 2, 3])
        self.ground_truth["privilege_tier"] = random.choice([1.0, 2.0, 3.0])  # user/service/admin

        return self.state.copy(), {}

    def action_masks(self) -> np.ndarray:
        """Returns boolean mask of valid actions for the current state."""
        mask = np.zeros(N_ACTIONS, dtype=bool)

        # Finished episode
        if self.state[23] == 1.0 and 11 in self.action_history:
            return mask

        # Actions 0, 1: Port scans (valid once)
        if 0 not in self.action_history:
            mask[0] = True
        if 1 not in self.action_history:
            mask[1] = True

        # Action 2: Fingerprinting (valid if any port is discovered open)
        has_open_port = any(self.state[p] >= 0.5 for p in range(8))
        if has_open_port and 2 not in self.action_history:
            mask[2] = True

        # Actions 3..6: Web surface mapping (valid only if an HTTP port is open >= 0.5)
        http_open = any(self.state[p] >= 0.5 for p in HTTP_PORTS)
        if http_open:
            for a in range(3, 7):
                if a not in self.action_history:
                    mask[a] = True

        # Actions 7..10: Header audit, boundary diagnostic, access control, metadata
        has_surface = any(self.state[13:18] > 0.0) or any(self.state[0:8] == 1.0)
        if has_surface:
            for a in range(7, 11):
                if a not in self.action_history:
                    mask[a] = True

        # Action 11: HANDOFF_TO_MEX (valid once HandoffReady == 1.0)
        if self.state[23] == 1.0:
            mask[11] = True

        # Fallback to avoid empty mask before termination
        if not np.any(mask):
            mask[11] = True

        return mask

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        self.steps += 1
        reward = self.step_penalty
        terminated = False
        truncated = False
        action = int(action)

        # Rate limit risk penalty if threshold exceeded
        if self.state[21] > 0.85:
            reward += self.rate_limit_penalty

        # Cooldown / passive decay for rate limit risk
        self.state[21] = max(0.0, float(self.state[21] - 0.05))

        self.action_history.add(action)

        if action == 0:
            # Scan ports 0..3
            self.state[21] = min(1.0, float(self.state[21] + 0.15))
            discovered = 0
            for p in range(4):
                if p in self.ground_truth["open_ports"]:
                    self.state[p] = 0.5  # open, unprobed
                    discovered += 1
            reward += discovered * self.discovery_reward

        elif action == 1:
            # Scan ports 4..7
            self.state[21] = min(1.0, float(self.state[21] + 0.15))
            discovered = 0
            for p in range(4, 8):
                if p in self.ground_truth["open_ports"]:
                    self.state[p] = 0.5
                    discovered += 1
            reward += discovered * self.discovery_reward

        elif action == 2:
            # Fingerprint open ports
            self.state[21] = min(1.0, float(self.state[21] + 0.10))
            fp_count = 0
            for p in range(8):
                if self.state[p] == 0.5:
                    self.state[p] = 1.0  # fingerprinted
                    fp_count += 1
            reward += fp_count * self.discovery_reward

        elif action == 3:
            # Web Endpoint Mapping
            self.state[21] = min(1.0, float(self.state[21] + 0.15))
            self.state[8] = 0.80  # 2xx
            self.state[9] = 0.15  # 3xx
            self.state[15] = 1.0  # REST_API
            reward += self.discovery_reward + 2.0

        elif action == 4:
            # API Schema Discovery
            self.state[21] = min(1.0, float(self.state[21] + 0.12))
            self.state[16] = 1.0  # QueryParams
            if self.ground_truth["has_api_leak"]:
                self.state[19] = 1.0  # ResponseSizeDelta
                reward += self.discovery_reward + 3.0
            else:
                reward += self.discovery_reward

        elif action == 5:
            # Auth Flow Inspection
            self.state[21] = min(1.0, float(self.state[21] + 0.10))
            self.state[10] = 0.40  # 401
            self.state[11] = 0.20  # 403
            self.state[13] = 1.0  # AuthForm
            if self.ground_truth["has_auth_bypass"]:
                self.state[20] = 1.0  # ErrorPatternDetected
                self.state[23] = 1.0  # Entry confirmed -> HandoffReady
                reward += self.anomaly_reward
            else:
                reward += self.discovery_reward

        elif action == 6:
            # Input Surface Parameter Mapping
            self.state[21] = min(1.0, float(self.state[21] + 0.15))
            self.state[14] = 1.0 if self.ground_truth["has_upload_vuln"] else 0.5
            reward += self.discovery_reward

        elif action == 7:
            # Header Config Audit
            self.state[17] = 1.0  # HeaderSensitivity
            reward += self.discovery_reward

        elif action == 8:
            # Boundary Parameter Anomaly Diagnostics
            self.state[21] = min(1.0, float(self.state[21] + 0.20))
            if self.ground_truth["has_boundary_5xx"]:
                self.state[12] = 0.75  # 5xx_anomaly
                self.state[18] = 1.0  # LatencySpike
                self.state[20] = 1.0  # ErrorPatternDetected
                self.state[23] = 1.0  # HandoffReady
                reward += self.anomaly_reward
            else:
                reward += 1.0

        elif action == 9:
            # Access-Control Boundary Check
            self.state[11] = 0.80  # 403
            if self.ground_truth["has_auth_bypass"]:
                self.state[23] = 1.0  # HandoffReady
                reward += self.anomaly_reward
            else:
                reward += self.discovery_reward

        elif action == 10:
            # Version Metadata Correlation
            self.state[23] = 1.0  # Correlated entry vector verified -> HandoffReady
            reward += self.discovery_reward + 3.0

        elif action == 11:
            # HANDOFF_TO_MEX
            if self.state[23] == 1.0:
                reward += self.handoff_reward
                terminated = True
            else:
                reward += -10.0

        # Update CoverageRatio: fraction of observed features active
        coverage = (np.count_nonzero(self.state[0:8] == 1.0) + np.count_nonzero(self.state[13:18] == 1.0)) / 13.0
        self.state[22] = float(np.clip(coverage, 0.0, 1.0))

        if self.steps >= self.max_steps:
            truncated = True

        # Construct 4-dim context vector for MEx-RL handoff:
        # [entry_service_type (0..3 norm), anomaly_confidence (0..1), noise_level (0..1), privilege_tier (0..1)]
        context_vector = np.array([
            float(self.ground_truth["entry_service"] / 3.0),
            float(self.state[20]),  # ErrorPatternDetected / Anomaly confidence
            float(self.state[21]),  # RateLimitRisk / noise level
            float(self.ground_truth["privilege_tier"] / 3.0),
        ], dtype=np.float32)

        info = {
            "handoff_ready": bool(self.state[23] == 1.0),
            "context_vector": context_vector,
            "won": bool(terminated),
        }

        return self.state.copy(), reward, terminated, truncated, info


class RExGPUEnv:
    """High-Throughput PyTorch CUDA Vectorized Environment for REx-RL."""

    def __init__(
        self,
        num_envs: int = 4096,
        device: str = "cuda",
        step_penalty: float = -0.5,
        discovery_reward: float = 2.5,
        anomaly_reward: float = 12.0,
        handoff_reward: float = 50.0,
        max_steps: int = 25,
    ):
        self.num_envs = num_envs
        self.device = torch.device(device if (device == "cuda" and torch.cuda.is_available()) else "cpu")
        self.step_penalty = step_penalty
        self.discovery_reward = discovery_reward
        self.anomaly_reward = anomaly_reward
        self.handoff_reward = handoff_reward
        self.max_steps = max_steps

        self.obs = torch.zeros((self.num_envs, OBS_DIM), device=self.device, dtype=torch.float32)
        self.action_history = torch.zeros((self.num_envs, N_ACTIONS), device=self.device, dtype=torch.bool)
        self.has_anomaly = torch.zeros((self.num_envs,), device=self.device, dtype=torch.bool)
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
        self.action_history[env_ids] = False
        self.steps[env_ids] = 0
        self.has_anomaly[env_ids] = torch.rand(n, device=self.device) < 0.70

    def get_action_masks(self) -> torch.Tensor:
        masks = torch.zeros((self.num_envs, N_ACTIONS), device=self.device, dtype=torch.bool)
        not_done = self.obs[:, 23] == 0.0  # Handoff not executed

        # Scans (0, 1)
        masks[:, 0] = (~self.action_history[:, 0]) & not_done
        masks[:, 1] = (~self.action_history[:, 1]) & not_done

        # Fingerprint (2)
        has_ports = (self.obs[:, 0:8] >= 0.5).any(dim=1)
        masks[:, 2] = has_ports & (~self.action_history[:, 2]) & not_done

        # Web actions (3..6)
        http_open = (self.obs[:, HTTP_PORTS] >= 0.5).any(dim=1)
        for a in range(3, 7):
            masks[:, a] = http_open & (~self.action_history[:, a]) & not_done

        # Detailed analysis (7..10)
        has_surface = (self.obs[:, 13:18] > 0.0).any(dim=1) | (self.obs[:, 0:8] == 1.0).any(dim=1)
        for a in range(7, 11):
            masks[:, a] = has_surface & (~self.action_history[:, a]) & not_done

        # Handoff (11)
        masks[:, 11] = (self.obs[:, 23] == 1.0) & not_done

        deadlocks = (~masks.any(dim=1)) & not_done
        if deadlocks.any():
            masks[deadlocks, 0] = True

        return masks

    def step(self, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self.steps += 1
        rewards = torch.full((self.num_envs,), self.step_penalty, device=self.device, dtype=torch.float32)

        # Mark action history
        env_idx = torch.arange(self.num_envs, device=self.device)
        self.action_history[env_idx, actions] = True

        # Port scans (0, 1)
        s0 = actions == 0
        if s0.any():
            self.obs[s0, 0] = 0.5  # HTTP open
            self.obs[s0, 1] = 0.5  # HTTPS open
            rewards[s0] += 2.0 * self.discovery_reward

        s1 = actions == 1
        if s1.any():
            self.obs[s1, 4] = 0.5  # HTTP-Alt
            self.obs[s1, 7] = 0.5  # API
            rewards[s1] += 2.0 * self.discovery_reward

        # Fingerprint (2)
        s2 = actions == 2
        if s2.any():
            open_mask = self.obs[s2, 0:8] == 0.5
            self.obs[s2, 0:8] = torch.where(open_mask, torch.tensor(1.0, device=self.device), self.obs[s2, 0:8])
            rewards[s2] += 2.0 * self.discovery_reward

        # Web (3..6)
        s3 = actions == 3
        if s3.any():
            self.obs[s3, 8] = 0.8
            self.obs[s3, 15] = 1.0
            rewards[s3] += self.discovery_reward + 2.0

        s4 = actions == 4
        if s4.any():
            self.obs[s4, 16] = 1.0
            rewards[s4] += self.discovery_reward

        s5 = actions == 5
        if s5.any():
            self.obs[s5, 10] = 0.4
            self.obs[s5, 13] = 1.0
            s5_anom = s5 & self.has_anomaly
            if s5_anom.any():
                self.obs[s5_anom, 20] = 1.0
                self.obs[s5_anom, 23] = 1.0  # Handoff ready
                rewards[s5_anom] += self.anomaly_reward
            rewards[s5] += self.discovery_reward

        s6 = actions == 6
        if s6.any():
            self.obs[s6, 14] = 1.0
            rewards[s6] += self.discovery_reward

        # Anomaly diagnostics (8)
        s8 = actions == 8
        if s8.any():
            s8_anom = s8 & self.has_anomaly
            if s8_anom.any():
                self.obs[s8_anom, 12] = 0.75
                self.obs[s8_anom, 18] = 1.0
                self.obs[s8_anom, 23] = 1.0  # Handoff ready
                rewards[s8_anom] += self.anomaly_reward

        # Metadata correlation (10)
        s10 = actions == 10
        if s10.any():
            self.obs[s10, 23] = 1.0  # Handoff ready
            rewards[s10] += self.discovery_reward + 3.0

        # Handoff to MEx (11)
        s11 = actions == 11
        terminated = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        if s11.any():
            val_envs = s11 & (self.obs[:, 23] == 1.0)
            rewards[val_envs] += self.handoff_reward
            terminated[val_envs] = True

        truncated = self.steps >= self.max_steps
        dones = terminated | truncated

        done_ids = torch.where(dones)[0]
        if len(done_ids) > 0:
            self.reset(done_ids)

        return self.obs.clone(), rewards, dones
