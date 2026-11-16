"""
Kengaytirilgan Linux PrivEsc simulyator (v2) - "machine" darajasiga tayyorgarlik.

8 ta zaiflik turi, har birida CHECK va EXPLOIT action'lari bor (jami 16 action).
Har "mashina"da ba'zan 1-2 ta zaiflik haqiqiy bo'ladi, qolgani yo'q.

Observation (17 o'lcham):
    Har bir zaiflik turi uchun 2 ta float: [tekshirildimi, zaif_bormi]
    8 tur * 2 = 16, + 1 ta "root_olindi" bayrog'i = 17

Vuln turlari (index tartibida):
    0: SUID binary
    1: Sudo misconfig
    2: Cron job (writable script)
    3: PATH hijacking
    4: Capabilities (masalan cap_setuid)
    5: Writable /etc/passwd
    6: Writable systemd service
    7: Docker group member

Action encoding:
    0..7   -> shu indexdagi vulnni TEKSHIRISH (check)
    8..15  -> shu indexdagi vulnni EXPLOIT qilish (exploit = index-8)
"""

import random
import numpy as np
import gymnasium as gym
from gymnasium import spaces

VULN_NAMES = [
    "suid", "sudo", "cron", "path_hijack",
    "capabilities", "passwd_write", "service_write", "docker_group",
]
N_VULNS = len(VULN_NAMES)


class LinuxPrivEscEnvV2(gym.Env):
    def __init__(self, n_real_vulns=(1, 2)):
        super().__init__()
        self.observation_space = spaces.Box(low=0, high=1, shape=(2 * N_VULNS + 1,), dtype=np.float32)
        self.action_space = spaces.Discrete(2 * N_VULNS)

        self.n_real_vulns = n_real_vulns
        self.state = None
        self.real_vulns = None
        self.steps = 0
        self.max_steps = 40

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        k = random.randint(*self.n_real_vulns)
        self.real_vulns = set(random.sample(range(N_VULNS), k))
        self.state = np.zeros(2 * N_VULNS + 1, dtype=np.float32)
        self.steps = 0
        return self.state.copy(), {}

    def step(self, action):
        self.steps += 1
        reward = -0.5  # har qadam uchun kichik "vaqt narxi"
        terminated = False
        truncated = False

        if action < N_VULNS:
            vuln_idx = action
            checked_offset = 2 * vuln_idx
            is_real_offset = 2 * vuln_idx + 1

            if self.state[checked_offset] == 0:
                self.state[checked_offset] = 1.0
                is_real = 1.0 if vuln_idx in self.real_vulns else 0.0
                self.state[is_real_offset] = is_real
                reward += 2.0
            else:
                reward += -2.0

        else:
            vuln_idx = action - N_VULNS
            is_real_offset = 2 * vuln_idx + 1

            if self.state[is_real_offset] == 1.0:
                reward += 30.0
                self.real_vulns.discard(vuln_idx)
                self.state[is_real_offset] = 0.0

                if len(self.real_vulns) == 0:
                    reward += 30.0
                    self.state[-1] = 1.0
                    terminated = True
            else:
                reward += -5.0

        if self.steps >= self.max_steps:
            truncated = True

        return self.state.copy(), reward, terminated, truncated, {}

    def render(self):
        checked = {VULN_NAMES[i]: self.state[2 * i] for i in range(N_VULNS)}
        print(f"Checked: {checked}, real_vulns={[VULN_NAMES[i] for i in self.real_vulns]}")

