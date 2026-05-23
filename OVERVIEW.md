# Mouse Behavioral Segmentation — Project Overview
**Kaplan Lab | Open Field Test | DeepLabCut → VAME 0.12.0**

---

## End Goal

Quantify how cannabidiol (CBD) treatment affects mouse locomotor and anxiety-related behavior during the Open Field Test (OFT) across three developmental windows (P7–21, P14–28, P28–42). The pipeline produces unsupervised behavioral motifs for each animal — discrete, recurring movement patterns identified without manual annotation — which are then compared statistically across treatment groups (CBD vs. vehicle, male vs. female).

---

## Dataset

- **38 sessions** across multiple cohorts and treatment groups
- **Groups:** P7–21 CBD/Vehicle, P14–28 CBD/Vehicle, P28–42 CBD/Vehicle
- **Sex:** mixed male and female
- **Pose estimation:** DeepLabCut, 27 keypoints per frame (nose, ears, spine, tail, shoulders, etc.)
- **Video:** OFT arena, top-down view

---

## Full Pipeline

### Step 1 — Pose Estimation (DeepLabCut)
Raw `.mp4` videos are passed through a pretrained DLC model. Output: `.h5` files containing x/y coordinates and likelihood scores for 27 keypoints per frame, per session.

**Scripts:** `dlc.py`, `dlcMany.py`, `dlcCoordinates.py`, `h5_to_csv.py`

### Step 2 — Preprocessing (VAME)
```python
vame.preprocessing(config, run_lowconf_cleaning=False)
```
- Egocentrically aligns each frame: body centered on `mouse_center`, rotated so `tail_base` defines the forward axis
- Produces `_processed.nc` files in `data/processed/`
- **Critical:** `run_lowconf_cleaning=False` — many keypoints (shoulders, distal tail) have >80% frames below the default confidence threshold; enabling cleaning wipes entire time series to NaN and crashes `savgol_filter`

### Step 3 — Create Training Set
```python
vame.create_trainset(config)
```
- Selects 4 high-confidence keypoints: `mouse_center`, `tail_base`, `head_midpoint`, `neck` (all >0.80 mean likelihood across sessions)
- Excludes egocentric reference points (they become trivial constants after alignment)
- Outputs 5 features per frame: `neck_x`, `neck_y`, `head_midpoint_x`, `head_midpoint_y`, `tail_base_y`
- Saves `train_seq.npy` and `test_seq.npy` to `data/train/`

**Why 5 features and not all 27 keypoints:**
- 7 keypoints have mean likelihood < 0.5 — training on noisy/missing data corrupts the latent space
- Egocentric alignment makes `mouse_center` (0,0) and `tail_base_x` constant — excluded automatically
- Remaining 5 features capture body curvature and head/neck posture, which encode behaviorally meaningful information

### Step 4 — Train Model
```python
vame.train_model(config)
```
An **RNN-VAE** (Recurrent Variational Autoencoder) with GRU transition units. The model encodes sequences of pose features into a low-dimensional latent space (`zdims`) and reconstructs them. A separate prediction head decodes the next `prediction_steps` frames from the same latent encoding (used only as a training regularizer).

The loss is:
```
L = MSE_reconstruction + beta * KL_divergence + MSE_future_prediction
```

KL weight is annealed from 0 → 1 over `annealtime` epochs starting at `kl_start`, so the encoder first learns useful structure before being forced into a Gaussian prior.

**Current config (VAME_5pt):**
| Parameter | Value | Reason |
|-----------|-------|--------|
| `zdims` | 4 | 5 input features; latent dim > input dim is wasteful |
| `beta` | 0.1 | With 5 features + mean reduction; higher beta causes posterior collapse |
| `annealtime` | 100 | Slow KL ramp; latent space forms before Gaussian pressure |
| `kl_start` | 6 | Short warm-up before annealing begins |
| `model_convergence` | 150 | Must be > `kl_start + annealtime = 106` |
| `hidden_size` | 256 | Sufficient for 5-feature input |
| `mse_reconstruction_reduction` | `"mean"` | **Critical** — default `"sum"` inflates MSE ~1400× |
| `mse_prediction_reduction` | `"mean"` | Same fix |

### Step 5 — Evaluate Model
```python
vame.evaluate_model(config)
```
Reconstructs held-out test sequences and plots actual vs. decoded pose trajectories. Confirms the encoder has learned meaningful kinematics (not just memorizing training data or collapsing to the prior).

### Step 6 — Segment Sessions
```python
vame.segment_session(config)
```
The trained encoder embeds every frame from every session into the latent space. An HMM (Hidden Markov Model) then segments the continuous latent trajectory into discrete behavioral motifs. The number of motifs is set by `n_clusters`.

### Step 7 — Community Detection
```python
vame.community(config)
```
Hierarchically clusters motifs across sessions so that similar motifs discovered in different animals are grouped together. Must run before `motif_videos()`.

### Step 8 — Visualization
```python
vame.motif_videos(config)       # representative video clips per motif
visualize_umap(config)          # 2D UMAP projection of latent space (30,000 points)
```

---

## What Has Been Tried

### Run 1 — Original Model (Feb 2026) ❌ Posterior collapse
**Config:** Default VAME params (`beta=1.0`, `annealtime=4`, `kl_start=2`, `mse_reduction="sum"`, 27 keypoints)

**Problem:** Two compounding issues:
1. `mse_reduction="sum"` made MSE ~1400× larger than KL → the KL term was negligible → the VAE learned no latent structure, acting as a plain autoencoder
2. `kl_start=2, annealtime=4` → KL fully active by epoch 6 → latent space forced into Gaussian before any useful representation formed

**Symptoms:** KL flatlined at ~0.05 (KL/dim ≈ 0.004), both MSE and KL stopped changing. Motifs all looked similar — arena-position artifacts rather than behavioral postures.

**Training stopped** at epoch 126/500 (early stopping with `model_convergence=50`).

---

### Hyperparameter Optimization (Feb–Mar 2026)
Ran an Optuna TPE study (10 trials × 30 epochs each) over: `zdims`, `lr`, `beta`, `annealtime`, `kl_start`, `dropout`, `prediction_steps`, `hidden_size`. Persisted to `vame_hpo.db`.

**Best trial:** Trial 6 — test loss 702 (45% improvement over baseline 1274)
- Key insight: top trials all used long `annealtime` (25–48 epochs) and non-zero dropout. Short `annealtime` (same as baseline) → same failure mode.
- Full results in `vame_hpo.ipynb` and `VAME_OFT_Report.md`

---

### Run 2 — Trial 6 Parameters + Egocentric Preprocessing (Apr 1 2026) ❌ Premature stopping
**Config:** `beta=0.1`, `zdims=4`, `annealtime=100`, `kl_start=6`, `model_convergence=50`, 5 features

**What changed from Run 1:**
- Switched to 5 high-confidence keypoints only
- Fixed `mse_reduction` to `"mean"`
- Added egocentric preprocessing
- Reduced beta to 0.1 (preventing posterior collapse with fewer features)

**Problem:** `model_convergence=50` triggered early stopping at **epoch 50**, but KL annealing doesn't finish until epoch `kl_start + annealtime = 106`. The best model was saved before the VAE ever experienced meaningful KL pressure.

**KL/dim at best epoch:** 0.376 (vs 0.004 in Run 1) — the fix worked, but training was cut too short.

---

### Run 3 — Corrected Convergence (Apr 2 2026) ✓ In progress
**Config:** `beta=0.1`, `zdims=4`, `annealtime=100`, `kl_start=6`, **`model_convergence=150`**, 5 features

**Fix:** `model_convergence=150 > kl_start + annealtime = 106`, giving 50 post-ramp epochs for the model to stabilize before early stopping can fire.

**Training pace:** ~155 sec/epoch on RTX 5060 Ti.

---

## Known Bugs in VAME 0.12.0

| Bug | Symptom | Fix |
|-----|---------|-----|
| `beta_norm: True` ignored | Setting has no effect — `rnn_vae.py` never reads it | Set `beta` directly to the correct value |
| `mse_reduction="sum"` (default) | MSE ~1400× KL; VAE trains as plain autoencoder | Set both MSE reductions to `"mean"` |
| `run_lowconf_cleaning=True` (default) | Crashes `savgol_filter` with ValueError on keypoints with >80% low-confidence frames | Always pass `run_lowconf_cleaning=False` |
| `model_convergence` too short | Early stopping fires before KL annealing completes | Enforce `model_convergence > kl_start + annealtime` |
| Config not reloaded after `create_trainset` | `num_features` not updated in memory | Always reload config dict after `create_trainset()` |

---

## Keypoint Selection Rationale

DLC tracks 27 keypoints. Many are unreliable:

| Likelihood tier | Keypoints | Decision |
|----------------|-----------|----------|
| Critical (<0.5) | tail5, tail4, mid_backend3, mid_backend2, tail1, right_shoulder, left_shoulder | Excluded — >80% frames below threshold |
| Moderate (0.5–0.69) | mid_backend, mid_back, ear tips, tail2 | Excluded — noisy signal |
| Good (≥0.70) | nose, head_midpoint, neck, mouse_center, tail_base, front paws, hind paws | Candidates |

**Final selection for training:** `mouse_center` (egocentric center), `tail_base` (orientation reference), `head_midpoint`, `neck` — all with mean likelihood >0.80 across all 38 sessions.

After egocentric alignment, `mouse_center` becomes (0,0) and `tail_base_x` becomes a constant — so the model trains on **5 features**: `neck_x`, `neck_y`, `head_midpoint_x`, `head_midpoint_y`, `tail_base_y`.

---

## File Structure (tracked in this repo)

```
VAME/
├── OVERVIEW.md                  ← this file
├── CLAUDE.md                    ← Claude Code project instructions
├── VAME_OFT_Report.md           ← detailed HPO + training report
├── dlc.py                       ← single-video DLC inference
├── dlcMany.py                   ← batch DLC inference
├── dlcCoordinates.py            ← coordinate extraction from DLC output
├── h5_to_csv.py                 ← convert DLC .h5 to readable .csv
├── VamePipeline.ipynb           ← main VAME pipeline notebook
├── vame_train_5pt.ipynb         ← 5-keypoint training run
├── vame_hpo.ipynb               ← Optuna HPO study
├── vame_hpo.db                  ← HPO study (resumable SQLite)
├── vame_hpo_ego.db              ← HPO study on egocentric data
├── vame_fix.ipynb               ← debugging and fix validation
├── eightmotifs.ipynb            ← motif analysis at n=8
├── femaleGOVehicleOFT.ipynb     ← female GO vehicle group analysis
├── Open-Field-Test/
│   ├── config.yaml              ← VAME project config (current best)
│   ├── config_backup_*.yaml     ← config history per run
│   ├── cluster.py               ← cluster count sweep script
│   ├── OpenFieldTest.ipynb      ← session-level analysis
│   └── model/
│       └── evaluate/
│           └── future_reconstruction.png   ← model eval plot
└── [analysis PNGs]              ← motif usage, HPO history, loss curves
```

**Not tracked (too large / reproducible from raw data):**
- `Open-Field-Test/data/` — raw `.h5` DLC files, processed `.nc` files, training `.npy` arrays
- `Open-Field-Test/model/best_model/` — `.pkl` model weights and snapshots
- `Open-Field-Test/results/` — per-session segmentation outputs (~20 GB)
- `Open-Field-Test/logs/` — training logs
- Raw `.mp4` videos
