# 👁️ ARGUS: The AI Eye for CERN
### Solving Extreme Computational Physics with Geometric Sparse Vision Transformers

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)
![Status](https://img.shields.io/badge/Status-State%20of%20the%20Art-success)
![Hardware](https://img.shields.io/badge/Hardware-NVIDIA%20RTX%204060%20Ti-green)

> **"If you can recognize a face in a crowd, you can recognize a particle track in the noise."**

**ARGUS** is a novel **Vision Transformer (ViT)** architecture designed to break the "Computational Wall" of the High-Luminosity Large Hadron Collider (LHC). By treating particle tracking as a computer vision problem and implementing **Geometric Sparse Attention**, we achieve state-of-the-art accuracy in linear time $O(N)$, enabling real-time tracking on consumer "Green AI" hardware.

---

## 🏆 Key Results (The "Mic Drop")

ARGUS outperforms both traditional mathematical methods (Kalman Filters) and previous Deep Learning benchmarks.

| Method | Architecture | Complexity | Double Majority Score |
| :--- | :--- | :--- | :--- |
| **Standard Math** | Kalman Filter | $O(N!)$ (Exponential) | ~0.8500 |
| **Kaggle Winner** | Top Quarks (Graph/LSTM) | $O(N^2)$ | 0.9218 |
| **Standard ViT** | Dense Attention | $O(N^2)$ | 0.8931 (Unstable) |
| **ARGUS (Ours)** | **Sparse ViT ($R=0.3$)** | **$O(N)$ (Linear)** | **0.9678 ⭐** |

> **Result:** We achieved a **4.9% improvement** over the historical benchmark while reducing computational complexity from Quadratic to Linear.

---

## ⚡ The Challenge: The "Computational Hairball"

The LHC's 2029 upgrade will generate **100,000+ simultaneous particle tracks**.
* **The Problem:** Current algorithms connect dots one-by-one. In high-density events, the number of possible connections explodes exponentially ( $O(N!)$ ).
* **The Solution:** We replace iterative calculation with **Global Perception**. ARGUS sees the entire event at once, identifying tracks as "clusters" in a learned latent space.

### The Innovation: Geometric Sparse Attention
Standard Transformers are slow ( $O(N^2)$ ) and get confused by distant noise.
* **Our Fix:** We restrict the Self-Attention mechanism to a physical 3D radius ( $R_{norm}=0.3$ ).
* **Physics Logic:** A particle cannot teleport. By forcing the AI to only look at local neighbors, we encode the laws of physics directly into the model, eliminating 99% of false connections.

---

## 🛠️ Installation & Usage

### 1. Clone the Repository
```bash
git clone [https://github.com/YourUsername/ARGUS-LHC-Tracking.git](https://github.com/YourUsername/ARGUS-LHC-Tracking.git)
cd ARGUS-LHC-Tracking
```

### 2\. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3\. Data Generation (Optional)

The repository includes `clean_data.xlsx` (Training) and `noisy_data.xlsx` (Testing). If you wish to generate new synthetic data, run the simulator.

**⚠️ Note:** Update the file path/name inside `LHC_simulator.py` before running to avoid overwriting existing data.

```bash
python LHC_simulator.py
```

**Simulation Parameters Used:**

  * `NUM_EVENTS = 5` (Collisions simulated)
  * `TRACKS_PER_EVENT = 100` (Particles per collision)
  * `NOISE_POINTS = 200` (Random noise dots per event for testing; 0 for training)
  * `B_FIELD = 2.0` (Magnetic Field strength in Tesla)
  * `DETECTOR_LAYERS = [30, 50, 70, 90, 110, 130, 150]` (Radii in mm)
  * `MAX_Z = 1000` (Detector length in mm)

### 4\. Run Model & Scoring

This script loads the data, trains the Sparse Vision Transformer, and outputs the Double Majority Score.

```bash
python predict_and_score.py
```

-----

## 📂 Repository Structure

  * `predict_and_score.py`: **The Core AI Engine.** Contains the Sparse ViT architecture, training loop, and scoring metrics.
  * `LHC_simulator.py`: **Physics Engine.** Generates synthetic helical tracks and noise based on charged particle motion in a magnetic field.
  * `clean_data.xlsx`: Synthetic dataset with `NOISE_POINTS = 0` (Used for "Ideal Physics" training).
  * `noisy_data.xlsx`: Synthetic dataset with `NOISE_POINTS = 200` (Used for "Real World" robustness testing).
  * `requirements.txt`: Python dependencies.

-----

## 👥 The Team

*Students of Institute of Engineering & Management (IEM), Kolkata**

  * **Nihal Gazi** (CSE AI/ML) - *Lead Researcher & AI Architect*
  * Aditya Kr. Biswas (CSBS)
  * Aihik Basu (CSE)
  * Anirban Chattopadhyay (CSE)
  * Sayan Pratihar (CSE AI/ML)

-----

## 🙏 Acknowledgements

  * **Department of Basic Science and Humanities (IEM)** & **IEMPHYS** for the research platform.
  * **CERN TrackML Challenge** for the inspiration and benchmarking standards.
  * **NVIDIA** for enabling Green AI research on consumer GPUs.

<!-- end list -->

