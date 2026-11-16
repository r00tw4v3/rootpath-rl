"""
Oddiy Linux PrivEsc CTF simulyator muhiti (Gymnasium formatida).

Holat (observation) - 7 ta 0/1 qiymatdan iborat vektor:
    [0] SUID tekshirildimi
    [1] SUID zaiflik topildimi (haqiqatda bormi)
    [2] Sudo tekshirildimi
    [3] Sudo zaiflik topildimi
    [4] Cron tekshirildimi
    [5] Cron zaiflik topildimi
    [6] Root olindimi (episode tugash belgisi)

Harakatlar (action) - 6 ta:
    0: SUID qidirish   (find / -perm -4000)
    1: Sudo tekshirish (sudo -l)
    2: Cron tekshirish (cat /etc/crontab)
    3: SUID exploit ishga tushirish
    4: Sudo exploit ishga tushirish
    5: Cron exploit ishga tushirish
"""

import random
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class LinuxPrivEscEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Box(low=0, high=1, shape=(7,), dtype=np.float32)
        self.action_space = spaces.Discrete(6)
        self.state = None
        self.real_vuln = None
        self.steps = 0
        self.max_steps = 30

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.real_vuln = random.choice(["suid", "sudo", "cron"])
        self.state = np.zeros(7, dtype=np.float32)
        self.steps = 0
        return self.state.copy(), {}

    def step(self, action):
        self.steps += 1
        reward = 0.0
        terminated = False
        truncated = False

        if action == 0:  # SUID tekshir
            if self.state[0] == 0:
                self.state[0] = 1
                self.state[1] = 1.0 if self.real_vuln == "suid" else 0.0
                reward = 2.0
            else:
                reward = -2.0

        elif action == 1:  # Sudo tekshir
            if self.state[2] == 0:
                self.state[2] = 1
                self.state[3] = 1.0 if self.real_vuln == "sudo" else 0.0
                reward = 2.0
            else:
                reward = -2.0

        elif action == 2:  # Cron tekshir
            if self.state[4] == 0:
                self.state[4] = 1
                self.state[5] = 1.0 if self.real_vuln == "cron" else 0.0
                reward = 2.0
            else:
                reward = -2.0

        elif action == 3:  # SUID exploit
            if self.state[1] == 1.0:
                reward = 50.0
                self.state[6] = 1.0
                terminated = True
            else:
                reward = -5.0

        elif action == 4:  # Sudo exploit
            if self.state[3] == 1.0:
                reward = 50.0
                self.state[6] = 1.0
                terminated = True
            else:
                reward = -5.0

        elif action == 5:  # Cron exploit
            if self.state[5] == 1.0:
                reward = 50.0
                self.state[6] = 1.0
                terminated = True
            else:
                reward = -5.0

        if self.steps >= self.max_steps:
            truncated = True

        return self.state.copy(), reward, terminated, truncated, {}

    def render(self):
        print(f"State: {self.state}, real_vuln={self.real_vuln}")
