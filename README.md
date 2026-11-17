# RootPath RL: Autonomous Multi-Agent RL Ecosystem

A high-performance Reinforcement Learning architecture for autonomous external surface exploration, hierarchical strategic management, tactical Linux state-machine resolution, and online adversarial sparring.

---

## Ecosystem Architecture

```
                                      +------------------------------------+
                                      |       RootPath RL Ecosystem        |
                                      +------------------------------------+
                                                        |
         +-----------------------------+----------------+-----------------------------+
         |                             |                |                             |
+------------------+          +------------------+     |     +------------------+          +--------------------+
|      REx-RL      |          |      MEx-RL      |     |     |      GEx-RL      |          | Adversarial Arena  |
|  (Pre-Access)    |          |   (Strategic)    |     |     |    (Tactical)    |          |    (Sparring)      |
+------------------+          +------------------+     |     +------------------+          +--------------------+
| - 24-dim Obs     |          | - 46-dim Obs     |     |     | - 21-dim Obs     |          | - Online Training  |
| - 12 Actions     |          | - 4 Strategic    |     |     | - Pure CUDA      |          | - Adaptive WAF/    |
| - 623,920 SPS    |          |   Domains + REx  |     |     | - 260,450 SPS    |          |   Hardening        |
| - CUDA Vec Tensor|          |   Context Handoff|     |     | - MaskablePPO    |          | - FIDE Elo &       |
| - MaskablePPO    |          | - Recurrent LSTM |     |     | - Local DAG Res  |          |   Telemetry        |
+------------------+          +------------------+     |     +------------------+          +--------------------+
         |                             |               |
         | Context Vector (4-dim)      | Domain Target |
         +---------------------------->+-------------->+
                                       
                                       =============================================
                                       Unified Pipeline Orchestrator (REx->MEx->GEx)
                                       100.0% Full-Chain Success | 20.64 Total Steps
                                       =============================================
```

---

## 1. Modules Overview

### 1. `REx-RL` (`rex_rl/`) — Pre-Access External Surface & Anomaly RL
- **Environment:** `RExGPUEnv` (pure PyTorch CUDA vectorized tensor engine with 4,096 parallel envs) and `RExGymEnv` (Gymnasium wrapper).
- **Observation Space:** 24-dimensional continuous vector representing port states (`0..7`), HTTP status distributions (`8..12`), surface flags (`13..17`), anomaly indicators (`18..20`), operational constraints (`21..22`), and handoff status (`23`).
- **Action Space:** 12 discrete actions with dynamic action masking (network inventory, endpoint enumeration, boundary diagnostics, and `HANDOFF_TO_MEX`).
- **Policy:** `MaskablePPO` (`MlpPolicy`, `net_arch=[128, 128]`).
- **Throughput:** **623,920 SPS** on NVIDIA GeForce RTX 5050 Laptop GPU.
- **Checkpoint:** `rex_rl/rex_surface_best.zip` (**100.0% Handoff Accuracy**, **4.17 Mean Steps**).

### 2. `MEx-RL` (`mex_rl/`) — Strategic Hierarchical Manager with Context Handoff
- **Environment:** `MExGymEnv` — 46-dimensional state POMDP multi-domain graph spanning `FilePermissions`, `ServiceDaemons`, `ScheduledTasks`, and `IdentityGroups`. Incorporates the 4-dim REx context vector `[entry_service_type, anomaly_confidence, noise_level, privilege_tier]` to route directly into valid sequential unlock paths.
- **Policy:** `RecurrentPPO` (`MlpLstmPolicy`, `lstm_hidden_size=128`) maintaining temporal memory across sequential steps to command domain macro-actions under noisy telemetry.
- **Throughput:** **109,011 SPS** on GPU.
- **Checkpoint:** `mex_rl/mex_manager_best.zip` (**100.0% Success Rate**, **9.07 Mean Steps**).

### 3. `GEx-RL` (`gex_rl/`) — Tactical High-Throughput Exploration
- **Environment:** `GExGPUEnv` (pure PyTorch CUDA batched tensor state-machine with 4,096 parallel envs) and `GExGymEnv` (Gymnasium wrapper) with dynamic Action Masking.
- **Policy:** `MaskablePPO` (`MlpPolicy`, `net_arch=[128, 128]`).
- **Throughput:** **260,450 SPS** on RTX 5050 GPU.
- **Checkpoint:** `gex_rl/gex_policy_best.zip` (**100.0% Success Rate**, **7.13 Mean Steps**).

### 4. `Adversarial Arena` (`arena/`) — Plug-and-Play Battleground & Sparring Gym
- **Environment:** `CombatArenaEnv` — Vectorized competitive battleground featuring an evolving **Defender Policy** allocating obfuscation, decoys, and rate limits under strict operational constraints (`availability >= 0.85`).
- **Online Co-Evolutionary Sparring:** `arena/sparring_gym.py` supporting `--agent [rex|mex|gex]`, `--online-train` (live co-evolutionary weight updates), `--max-rounds`, and `--target-score`.
- **Live Telemetry & Elo:** Dynamically tracks round-by-round win rates, step counts, scores, and FIDE Elo ratings, automatically exporting battle-hardened champion checkpoints (`arena/<agent>_arena_champion.zip`).

### 5. `Unified Pipeline Orchestrator` (`pipeline_orchestrator.py`)
- Seamlessly chains **REx $\rightarrow$ MEx $\rightarrow$ GEx**:
  1. **Stage 1 (REx):** Probes external attack surface, identifies anomaly, constructs 4-dim context vector.
  2. **Stage 2 (MEx):** Ingests REx context, navigates multi-domain POMDP graph, unmasks target vulnerability.
  3. **Stage 3 (GEx):** Receives target domain, resolves internal DAG prerequisites, claims root state.
- **Performance:** **100.0% Full-Chain Success**, **20.64 Mean Total Steps**, **220 episodes/sec**.

---

## 2. Benchmark Summary (200 Episodes Against Adaptive Defender)

| Agent Ecosystem | Operational Scope | Architecture | Training SPS | Win Rate | Mean Steps | Mean Score | Elo Rating |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **REx-RL (Baseline Best)** | External / Pre-Access | `Pre-Access Surface (CUDA Vec)` | `623,920 SPS` | **100.0%** | **4.17** | 66.76 | 1365 |
| **REx-RL (Arena Champion)** ⭐ | External / Pre-Access | `Pre-Access Surface (CUDA Vec)` | `623,920 SPS` | **100.0%** | **4.17** | 66.76 | 1365 |
| **MEx-RL (Manager Best)** | Strategic Multi-Domain | `Strategic Recurrent LSTM` | `109,011 SPS` | **100.0%** | **9.07** | 155.06 | 1365 |
| **MEx-RL (Arena Champion)** ⭐ | Strategic Multi-Domain | `Strategic Recurrent LSTM` | `109,011 SPS` | **100.0%** | **8.94** | 155.49 | 1365 |
| **GEx-RL (Baseline Best)** | Tactical Resolution | `Tactical MLP (CUDA Vec)` | `260,450 SPS` | **100.0%** | **7.13** | 63.51 | 1366 |
| **GEx-RL (Arena Champion)** ⭐ | Tactical Resolution | `Tactical MLP (CUDA Vec)` | `260,450 SPS` | **100.0%** | **7.13** | 63.51 | 1366 |

### End-to-End Pipeline Evaluation (100 Consecutive Episodes)
- **Stage 1 (REx-RL):** `100.0%` Accuracy | `4.26` mean steps
- **Stage 2 (MEx-RL):** `100.0%` Accuracy | `9.23` mean steps
- **Stage 3 (GEx-RL):** `100.0%` Accuracy | `7.15` mean steps
- **Full-Chain Success Rate:** **`100.0%`**
- **Total Pipeline Mean Steps:** **`20.64` steps**
- **Pipeline Execution Speed:** **`220.0` episodes/sec**

---

## 3. Quickstart & CLI Usage

```bash
# 1. Activate Python virtual environment
source venv/bin/activate

# 2. Train Pre-Access REx-RL (2,000,000 steps on GPU CUDA Vec)
python3 rex_rl/train_rex.py

# 3. Train Strategic MEx-RL with Context Handoff (Recurrent LSTM Manager)
python3 mex_rl/train_mex.py

# 4. Train Tactical GEx-RL (2,000,000 steps on GPU CUDA Vec)
python3 gex_rl/train_gex.py

# 5. Launch Online Co-Evolutionary Sparring in Adversarial Arena
python3 arena/sparring_gym.py --agent rex --max-rounds 50 --online-train
python3 arena/sparring_gym.py --agent mex --max-rounds 50 --online-train
python3 arena/sparring_gym.py --agent gex --max-rounds 50 --online-train

# 6. Run End-to-End 3-Tier Pipeline Orchestrator (REx -> MEx -> GEx)
python3 pipeline_orchestrator.py --episodes 100

# 7. Run Full Ecosystem Benchmark Suite
python3 benchmark_all.py

# 8. Synchronize all deliverables, models, and docs to GitHub
bash sync_github.sh
```
