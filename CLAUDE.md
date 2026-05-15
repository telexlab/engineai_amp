# EngineAI AMP — Claude Code Guide

## What This Repo Does

Training pipeline for PM01 locomotion policies using Isaac Lab + Isaac Sim. Produces `.mnn` policy files that are deployed in either:
- `engineai_ros2_workspace` — via ROS2 interface
- `engineai_robotics_native_sdk` — via native control framework

This repo never touches the real robot. It only produces model files.

---

## Repo Structure

```
source/engineai_lab/
├── algorithms/       # RL algorithm implementations (PPO, AMP)
├── assets/pm01/      # PM01 meshes and URDF for simulation
├── robots/           # Robot definitions
├── tasks/velocity/   # Task environments
│   ├── config/pm01/  # PM01-specific task configs
│   └── mdp/          # Reward functions, observations, actions
└── utils/            # Helper utilities
scripts/
├── train.py          # Launch training
└── play.py           # Visualise/evaluate a trained policy
dataset/              # Motion capture data for AMP training
config/               # Global configs
```

---

## Supported Training Environments

| Task | Description |
|------|-------------|
| `Flat-PM01-v0` | Standard PPO — robot learns to walk from scratch, rewarded for forward progress |
| `Flat-AMP-PM01-v0` | AMP — robot imitates motion capture reference data, produces more natural motion |

---

## Setup (GPU machine required)

### Prerequisites

- NVIDIA GPU with recent drivers
- Conda

### 1. Create environment

```bash
conda create -n engineai_lab python=3.11
conda activate engineai_lab
```

### 2. Install Isaac Sim

Follow: https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/pip_installation.html

Install dependencies only — stop before "Installing Isaac Lab".

### 3. Clone Isaac Lab (specific commit)

```bash
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab
git checkout 4df6560e
./isaaclab -i rsl_rl
```

### 4. Install this package

```bash
cd ~/engineai_amp/engineai_amp
pip install -e .
```

---

## Training

```bash
conda activate engineai_lab

# PPO flat locomotion
python scripts/train.py --task=Flat-PM01-v0 --num_envs 4096 --headless --run_name <name>

# AMP motion imitation
python scripts/train.py --task=Flat-AMP-PM01-v0 --num_envs 4096 --headless --run_name <name>
```

> **Note:** 8GB VRAM is tight for 4096 envs. Reduce to `--num_envs 256` or `--num_envs 512` if OOM.

Trained runs saved to `logs/rsl_rl/<name>/`.

---

## Evaluation

```bash
python scripts/play.py --task=Flat-PM01-v0 --num_envs 128 --load_run <name>
```

---

## Deployment Pipeline

```
Train → PyTorch model (.pt)
    ↓
Export to ONNX
    ↓
Convert to MNN:
    git clone https://github.com/alibaba/mnn
    cd mnn && mkdir build && cd build
    cmake .. -DMNN_BUILD_CONVERTER=ON
    make -j8
    ./MNNConvert -f ONNX --modelFile policy.onnx --MNNModel policy.mnn --bizCode MNN
    ↓
Drop .mnn into deployment repo and update YAML config:
  ROS2 ws:     src/interface_example/config/pm01/rl_basic/basic/policies/
  Native SDK:  assets/config/pm01_edu/rl_walking_example/
```

---

## Using Your Own Model (not trained with this repo)

You do not need this repo to deploy a custom policy. Any model that matches the PM01 interface can be converted and deployed directly:

- **Input**: 42-element observation vector
- **Output**: 12 joint action values (leg joints only)

Train in any framework → export ONNX → convert to MNN → deploy.

If your model has a different observation/action space, modify the runner in the native SDK (`src/runner/rl_walking_example/`) to build the correct observation vector.

---

## Hardware Note

This laptop (RTX 3000 Ada, 8GB VRAM) can run training but with reduced environments. A dedicated training workstation with 24GB+ VRAM is recommended for full `--num_envs 4096` runs.
