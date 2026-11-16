# RootPath RL: Three-Module Autonomous Ecosystem

A high-performance Reinforcement Learning architecture for autonomous Linux state-machine exploration, hierarchical strategic planning, and adversarial sparring.

---

## Ecosystem Architecture

```
                                      +------------------------------------+
                                      |       RootPath RL Ecosystem        |
                                      +------------------------------------+
                                                        |
         +----------------------------------------------+----------------------------------------------+
         |                                              |                                              |
+----------------------------------+          +----------------------------------+          +----------------------------------+
|      GEx-RL (Tactical Fast)      |          |    MEx-RL (Strategic Manager)    |          |    Adversarial Arena (Sparring)  |
+----------------------------------+          +----------------------------------+          +----------------------------------+
| - Pure PyTorch CUDA Tensors      |          | - POMDP Multi-Domain Graph       |          | - Plug-and-Play Battleground     |
| - 4096 Parallel Environments     |          | - 4 Strategic Macro Domains      |          | - Adaptive Hardening Defender    |
| - 260,450 Steps/Sec (Throughput) |          | - Recurrent LSTM Memory          |          | - FIDE Elo & Telemetry Dashboard |
| - MaskablePPO Checkpoint         |          | - RecurrentPPO Checkpoint        |          | - Champion Checkpoint Saving     |
+----------------------------------+          +----------------------------------+          +----------------------------------+
```

---

## 1. Modules Overview

### 1. `GEx-RL` (`gex_rl/`) — Tactical High-Throughput Exploration
- **Environment:** `GExGPUEnv` (pure PyTorch CUDA batched tensor state-machine with 4,096 parallel envs) and `GExGymEnv` (Gymnasium wrapper).
- **Policy:** `MaskablePPO` (`MlpPolicy`, `net_arch=[128, 128]`).
- **Throughput:** **260,450 SPS** on NVIDIA GeForce RTX 5050 Laptop GPU.
- **Artifact:** `gex_rl/gex_policy_best.zip`.

### 2. `MEx-RL` (`mex_rl/`) — Strategic Hierarchical Manager
- **Environment:** `MExGymEnv` — 16-node POMDP multi-domain graph spanning `FilePermissions`, `ServiceDaemons`, `ScheduledTasks`, and `IdentityGroups` with sequential unlock dependencies and noisy telemetry.
- **Policy:** `RecurrentPPO` (`MlpLstmPolicy`, `lstm_hidden_size=128`) maintaining temporal memory across sequential steps to command domain macro-actions.
- **Throughput:** **109,011 SPS** on GPU.
- **Artifact:** `mex_rl/mex_manager_best.zip`.

### 3. `Adversarial Arena` (`arena/`) — Plug-and-Play Competitive Gym
- **Environment:** `CombatArenaEnv` — Vectorized competitive battleground featuring an evolving **Defender Policy** allocating obfuscation and decoys under a strict availability constraint (`availability >= 0.85`).
- **Sparring Launcher:** `arena/sparring_gym.py` supporting `--agent [gex|mex]`, `--max-rounds`, `--target-score`, and `--time-limit`.
- **Live Telemetry:** Tracks round-by-round win rates, step counts, scores, and FIDE Elo ratings, automatically exporting battle-hardened champion checkpoints (`arena/<agent>_arena_champion.zip`).

---

## 2. Competitive Sparring Benchmark (200 Episodes)

| Agent Ecosystem | Model Architecture | Training Throughput | Sparring Win Rate | Mean Steps | Mean Cumulative Score |
|:---|:---:|:---:|:---:|:---:|:---:|
| **GEx-RL (Baseline)** | `Tactical MLP (CUDA Vec)` | `260,450 SPS` | **85.5%** | **10.52** | 59.82 |
| **GEx-RL (Arena Champion)** | `Tactical MLP (CUDA Vec)` | `260,450 SPS` | **85.5%** | **10.52** | 59.82 |
| **MEx-RL (Manager Best)** | `Strategic Recurrent LSTM` | `109,011 SPS` | **79.5%** | **11.80** | 115.61 |
| **MEx-RL (Arena Champion)** | `Strategic Recurrent LSTM` | `109,011 SPS` | **79.5%** | **11.80** | 115.61 |

---

## 3. Quickstart & Execution

```bash
# 1. Activate environment
source venv/bin/activate

# 2. Train Tactical GEx-RL (2,000,000 steps on GPU)
python3 gex_rl/train_gex.py

# 3. Train Strategic MEx-RL (Recurrent LSTM Manager)
python3 mex_rl/train_mex.py

# 4. Launch Adversarial Arena Sparring
python3 arena/sparring_gym.py --agent mex --max-rounds 200 --target-score 85.0

# 5. Run Full Benchmark Across 200 Scenarios
python3 benchmark_all.py

# 6. Synchronize to GitHub
bash sync_github.sh
```
