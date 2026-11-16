# RootPath RL: Autonomous Linux Privilege Escalation & Multi-WAN Systems

A high-performance Reinforcement Learning research & engineering pipeline for autonomous Linux Privilege Escalation planning and resilient Multi-WAN network aggregation.

---

## Architecture Overview

```
                                      +------------------------------------+
                                      |   LinuxPrivEscEnvV3 (Gymnasium)    |
                                      +------------------------------------+
                                                        |
                    +-----------------------------------+-----------------------------------+
                    |                                                                       |
       +-------------------------+                                             +-------------------------+
       |   DAG Dependency Graph  |                                             |   sb3-contrib Masking   |
       +-------------------------+                                             +-------------------------+
       |  enum_users  config_leak|                                             | - No duplicate checks   |
       |       \          /      |                                             | - Prereqs enforced      |
       |      service_write      |                                             | - Valid exploits only   |
       |  path_hijack  passwd... |                                             +-------------------------+
       +-------------------------+
                    |
                    v
       +-------------------------+
       | MaskablePPO (Winner Mlp)|
       +-------------------------+
       | 100% Success Rate       |
       | 7.22 Mean Steps to Root |
       +-------------------------+
```

---

## 1. Network Resilience & Multi-WAN Bonding (`bond.sh`)

Aggregates multiple WAN connections (`wlan1`, `usb0`, `wlan0`, `eth0`) via Policy-Based Routing (PBR) and multipath `nexthop` routing:
- **Dynamic Interface Detection:** Automatically discovers IP addresses and default gateways using `ip route` and `nmcli`.
- **Health Verification:** Verifies live outbound connectivity per interface (`curl -s --interface <dev> --max-time 4 ifconfig.me`) before routing traffic.
- **Fail-Safe Nexthop:** Dead interfaces are omitted dynamically, preventing packet drops from interrupting training or Git operations.
- **Isolated Routing Tables:** Maps `usb0` $\to$ `101`, `wlan1` $\to$ `102`, `wlan0` $\to$ `103`, `eth0` $\to$ `104`.

**Usage:**
```bash
sudo bash bond.sh
```

---

## 2. Environment Evolution

| Feature | V1 (`env.py`) | V2 (`env_v2.py`) | V3 (`env_v3.py`) |
|:---|:---:|:---:|:---:|
| **Categories / Nodes** | 3 (suid, sudo, cron) | 8 categories | **10 DAG Nodes** (2 Recon + 8 Vulns) |
| **Observation Space** | `Box(7,)` | `Box(17,)` | **`Box(21,)`** |
| **Action Space** | `Discrete(6)` | `Discrete(16)` | **`Discrete(18)`** |
| **Prerequisites (DAG)**| None (flat) | None (flat) | **Multi-step DAG Dependencies** |
| **Action Masking** | None | None | **Full `sb3-contrib` Action Masking** |
| **Reward Shaping** | Fixed | Fixed | **Configurable Parameters** in `__init__` |

### DAG Dependencies in V3:
- `path_hijack` $\leftarrow$ `enum_users`
- `passwd_write` $\leftarrow$ `config_leak`
- `service_write` $\leftarrow$ `enum_users` + `config_leak`
- Standalone vectors (`suid`, `sudo`, `cron`, `capabilities`, `docker_group`) can be investigated directly.

---

## 3. Autotuning & Benchmarks (`autotune_v3.py`)

Trained with `MaskablePPO("MlpPolicy", device="cpu")` over 100,000 steps across varying step penalties and hyperparameter schedules. Evaluated over 100 randomized test episodes.

| Candidate | Step Penalty | LR | Ent Coef | Success Rate | Mean Ep Length | Mean Reward |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| `CANDIDATE_1` | `-0.5` | `3e-4` | `0.01` | 100.0% | 8.57 | 85.22 |
| `CANDIDATE_2` | `-1.5` | `3e-4` | `0.005` | 100.0% | 8.45 | 76.31 |
| **`CANDIDATE_3` (BEST)** | **`-3.0`** | **`5e-4`** | **`0.001`** | **100.0%** | **7.22** | **65.95** |
| `CANDIDATE_4` | `-1.5` | `7e-4` | `0.01` | 100.0% | 8.77 | 91.50 |

The winning model (`CANDIDATE_3`) achieves **100% exploit success** with an optimal mean episode length of **7.22 steps**, saved as `privesc_maskable_ppo_best.zip`.

---

## 4. Quickstart & Verification

```bash
# 1. Activate environment
source venv/bin/activate

# 2. Run deterministic evaluation with the optimal model
python3 test_best_model.py

# 3. Run automated autotuning benchmark sweep
python3 autotune_v3.py

# 4. Sync repository to GitHub
bash sync_github.sh
```
