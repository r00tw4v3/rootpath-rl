#!/usr/bin/env python3
"""
==============================================================================
Phase V4: Live System Configuration Inspector & Observation Mapper
Connects live target inspection stdout to the 21-dim RL observation vector.
Compatible with privesc_maskable_ppo_best.zip and Dockerfile.lab environments.
==============================================================================
"""

import os
import re
import subprocess
import sys
from typing import Dict, List, Tuple

import numpy as np
from sb3_contrib import MaskablePPO

# Node definitions matching V3 & V4 architecture
RECON_NODES = ["enum_users", "config_leak"]
VULN_NODES = [
    "suid", "sudo", "cron", "path_hijack",
    "capabilities", "passwd_write", "service_write", "docker_group"
]
ALL_NODES = RECON_NODES + VULN_NODES


class SystemAuditInspector:
    """Read-only inspection executor for Linux hosts or Docker containers."""

    def __init__(self, container_name: str = None):
        self.container_name = container_name

    def run_cmd(self, cmd: str, timeout: int = 3) -> str:
        try:
            if self.container_name:
                exec_cmd = ["docker", "exec", self.container_name, "bash", "-c", cmd]
            else:
                exec_cmd = ["bash", "-c", cmd]

            res = subprocess.run(
                exec_cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            return res.stdout.strip()
        except Exception as e:
            return f"ERR: {str(e)}"

    def inspect_system(self) -> Dict[str, bool]:
        """Runs standard non-destructive auditing commands and regex parsers."""
        findings = {}

        # 1. Check user accounts & enumeration
        users_out = self.run_cmd("cat /etc/passwd")
        findings["enum_users"] = bool(re.search(r"(root|audituser|r00tw4v3):", users_out))

        # 2. Check configuration leaks
        leaks_out = self.run_cmd("grep -ri 'TOKEN\\|PASSWORD' /var/log/audit_leaks/ /tmp/ 2>/dev/null")
        findings["config_leak"] = bool(re.search(r"(TOKEN|PASSWORD|SECRET)", leaks_out, re.IGNORECASE))

        # 3. Check SUID binaries
        suid_out = self.run_cmd("find / -perm -4000 -type f 2>/dev/null")
        findings["suid"] = bool(re.search(r"(audit_suid|test_suid|ping|mount)", suid_out))

        # 4. Check Sudo permissions
        sudo_out = self.run_cmd("sudo -n -l 2>/dev/null || cat /etc/sudoers.d/* 2>/dev/null")
        findings["sudo"] = bool(re.search(r"NOPASSWD", sudo_out))

        # 5. Check Cron configurations
        cron_out = self.run_cmd("cat /etc/crontab /etc/cron.d/* 2>/dev/null")
        findings["cron"] = bool(re.search(r"(/tmp/|audit_cron)", cron_out))

        # 6. Check PATH hijacking opportunities
        path_out = self.run_cmd("echo $PATH; ls -ld /usr/local/bin /tmp")
        findings["path_hijack"] = bool(re.search(r"drwxrwxrwx.*(/tmp|/usr/local/bin)", path_out))

        # 7. Check file capabilities
        caps_out = self.run_cmd("getcap -r /opt /usr/bin 2>/dev/null")
        findings["capabilities"] = bool(re.search(r"cap_setuid", caps_out))

        # 8. Check Passwd write permissions
        passwd_perm = self.run_cmd("ls -l /etc/passwd")
        findings["passwd_write"] = bool(re.search(r"^-..w..w", passwd_perm))

        # 9. Check writable service files
        service_perm = self.run_cmd("find /etc/systemd/system/ -type f -perm -o+w 2>/dev/null")
        findings["service_write"] = bool(len(service_perm) > 0)

        # 10. Check Docker group membership
        groups_out = self.run_cmd("id -Gn")
        findings["docker_group"] = bool(re.search(r"\bdocker\b", groups_out))

        return findings

    def construct_observation_vector(self, findings: Dict[str, bool]) -> np.ndarray:
        """Constructs 21-dimensional float32 observation vector."""
        obs = np.zeros(21, dtype=np.float32)
        for idx, name in enumerate(ALL_NODES):
            obs[2 * idx] = 1.0  # marked checked
            obs[2 * idx + 1] = 1.0 if findings.get(name, False) else 0.0
        obs[-1] = 0.0  # initial root state
        return obs


def evaluate_live_system():
    inspector = SystemAuditInspector()
    print("==================================================================")
    print(" [V4 Live Inspector] Auditing System Configuration State")
    print("==================================================================")

    findings = inspector.inspect_system()
    for name in ALL_NODES:
        status = "CONFIRMED (+)" if findings.get(name, False) else "ABSENT (-)"
        print(f"  [{status:<13}] {name:<16}")

    obs = inspector.construct_observation_vector(findings)
    print("\n[+] Observation Vector (21-dim):")
    print("   ", np.array2string(obs, precision=1, separator=", "))

    # Load pre-trained model and evaluate prioritized next action
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "privesc_maskable_ppo_best")
    if os.path.exists(f"{model_path}.zip"):
        print("\n[*] Evaluating Prioritized Action Policy (privesc_maskable_ppo_best)...")
        model = MaskablePPO.load(model_path)

        # Construct masks for the real state
        masks = np.zeros(18, dtype=bool)
        # Valid exploits are actions 10..17 where vuln is confirmed
        for v_idx, v_name in enumerate(VULN_NODES):
            if findings.get(v_name, False):
                masks[10 + v_idx] = True

        # Fallback to recon checks if no immediate exploit
        if not np.any(masks):
            masks[0:10] = True

        action, _ = model.predict(obs, action_masks=masks, deterministic=True)
        action = int(action)

        if action < 10:
            decision_type = "CHECK"
            target_name = ALL_NODES[action]
        else:
            decision_type = "REMEDIATE / AUDIT"
            target_name = VULN_NODES[action - 10]

        print("==================================================================")
        print(f" [Model Recommendation] Action {action}: {decision_type} -> {target_name}")
        print("==================================================================")


if __name__ == "__main__":
    evaluate_live_system()
