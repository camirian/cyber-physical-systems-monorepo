# Autonomous Cyber-Physical Systems (CPS) Monorepo

> **DEPRECATION NOTICE:** The legacy Coursera Machine Learning note-taking repository has been archived. 

This repository has been fundamentally architected into a **Level 5 Cyber-Physical Systems Engineering Monorepo**. We do not perform rote data science or static CSV classification here. This is a rigorous testbed for **Physical AI**—where differentiable models govern physical reality, bounded strictly by thermodynamics, Hamiltonian mechanics, and formal control theory.

## 🌌 The 5-Phase Master Architecture

This repository is structured around the physical deployment lifecycle of an autonomous agent:

### PHASE 1: The Epistemological Baseline (`src/phase_1_epistemology/`)
*Beyond interpretability to structural equivariance.*
We utilize **Information Geometry** and **Lie Group Equivariance (SO(3), SE(3))**. Neural networks here do not just map inputs to outputs; they are mathematically constrained to preserve the symmetries of the physical laws governing the system (e.g., rotation/translation invariance).

### PHASE 2: The Differentiable Simulator (`src/phase_2_simulator/`)
*Digital twins that bridge the Sim-to-Real gap.*
Static datasets are banned. We integrate with high-fidelity, stochastic physics engines. By using differentiable physics, we can backpropagate gradients *through* the simulation itself, allowing for the joint optimization of hardware morphology and control policies.

### PHASE 3: The Physics-AI Convergence (`src/phase_3_convergence/`)
*Strictly Conservative AI.*
We employ **Hamiltonian Neural Networks (HNNs)**. Standard ML models diverge over time when predicting physical trajectories. HNNs learn the scalar Hamiltonian (Total Energy) and analytically derive equations of motion via PyTorch Autograd ($\dot{q} = \frac{\partial \mathcal{H}}{\partial p}, \dot{p} = -\frac{\partial \mathcal{H}}{\partial q}$). This guarantees symplectic phase space volume preservation and strict energy conservation.

### PHASE 4: MBSE & Formal Safety Verification (`src/phase_4_mbse/`)
*Zero Blast Radius.*
The "Blast Radius" of any AI agent is bounded by **Control Barrier Functions (CBFs)** and **Control Lyapunov Functions (CLFs)**. The AI proposes a nominal control vector $u_{nom}$, and a high-speed convex optimizer (Quadratic Program) projects it into the mathematically proven Safe Set $\mathcal{C}$. *The system cannot crash, mathematically.*

### PHASE 5: Edge Execution (`src/phase_5_edge/`)
*Bare-metal, real-time control constraints.*
Deployment pipelines specifically targeting edge compute (e.g., Jetson Orin Nano). This includes TensorRT bindings, kernel-level CUDA optimizations, and Wait-and-Verify automated hardware rollback loops to satisfy strict sub-10ms latency requirements.

---

## 🛠️ Repository Execution

```bash
# Phase 3: Run the Hamiltonian Neural Network (HNN) Smoke Test
python src/phase_3_convergence/pinns/hamiltonian_nn.py

# Phase 4: Run the Control Barrier Function (CBF) QP Filter
python src/phase_4_mbse/cbf_safety_filter.py
```

## 📐 Interface Control Documents (ICD)

All boundaries between the software inference graphs and the physical hardware actuators are strictly defined in `docs/icd/ICD_TEMPLATE.md` using the DSRP framework.
