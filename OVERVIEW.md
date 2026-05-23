# Mouse Behavioral Segmentation — Project Overview
**Kaplan Lab | Open Field Test | [DeepLabCut](https://github.com/DeepLabCut/DeepLabCut) → [VAME 0.12.0](https://github.com/LINCellularNeuroscience/VAME)**

---

## End Goal

Quantify how cannabidiol (CBD) treatment affects mouse locomotor and anxiety-related behavior during the Open Field Test (OFT) across three developmental windows (P7–21, P14–28, P28–42). The pipeline produces unsupervised behavioral **motifs** — discrete, recurring movement patterns identified without manual annotation — which are compared statistically across treatment groups (CBD vs. vehicle, male vs. female).

---

## Two-Sentence Summary

This project uses DeepLabCut to track mouse pose during Open Field Tests, then feeds those pose features into VAME (a recurrent variational autoencoder) to automatically segment behavior into discrete, recurring movement motifs — no manual annotation required. The goal is to quantify how CBD treatment affects locomotion and anxiety-related behavior across three developmental windows (P7–21, P14–28, P28–42) by comparing motif usage between treatment and vehicle groups.

---

## Dataset

- **38 sessions** across multiple cohorts and treatment groups
- **Groups:** P7–21 CBD/Vehicle, P14–28 CBD/Vehicle, P28–42 CBD/Vehicle
- **Sex:** mixed male and female
- **Pose estimation:** DeepLabCut, 27 keypoints per frame
- **Arena:** Open Field Test, top-down camera

---

## Pipeline (Correct Order)

```
DLC inference → h5_to_csv (inspection) →
vame.preprocessing() → vame.create_trainset() → vame.train_model() →
vame.evaluate_model() → vame.segment_session() → vame.community() →
vame.motif_videos() → visualize_umap()
```

See `scripts/` for DLC tools. See `Open-Field-Test/config.yaml` for current parameters.

---

## Repo Structure (Timeline)

```
OVERVIEW.md                          ← this file
CLAUDE.md                            ← Claude Code project instructions
scripts/                             ← DLC pipeline tools
01_run1-initial-feb2026/             ← first VAME attempt (default params)
02_hpo-study-mar2026/                ← Optuna HPO + model diagnosis
03_run2-5pt-apr2026/                 ← 5-keypoint + egocentric (bug fixes)
04_run3-fixed-convergence-apr2026/   ← corrected convergence config
Open-Field-Test/                     ← VAME project: configs, analysis outputs
```

---

## scripts/ — DLC Tools

| File | Purpose |
|------|---------|
| `dlc.py` | Single-video DLC inference |
| `dlcMany.py` | Batch DLC inference across all sessions |
| `dlcCoordinates.py` | Extract and inspect keypoint coordinates |
| `h5_to_csv.py` | Convert DLC `.h5` output to readable `.csv` |

---

## 01_run1-initial-feb2026/ — First VAME Attempt

**Dates:** Feb 17–22, 2026

**What was tried:** Default VAME 0.12.0 parameters. Initial pipeline exploration on a single subject (`femaleGOVehicleOFT.ipynb`), then full 38-session run.

**Config:**
- All 27 keypoints (no filtering)
- `beta=1.0`, `annealtime=4`, `kl_start=2`
- `mse_reconstruction_reduction="sum"` (default)
- `model_convergence=50`

**Result: Posterior collapse.** Training stopped at epoch 126/500. KL flatlined at ~0.05 (KL/dim = 0.004). Motifs all looked nearly identical — the model encoded arena position rather than body posture.

**Root causes identified (later):**
1. `mse_reduction="sum"` inflated MSE ~1400× larger than KL → the KL term was negligible → VAE learned no latent structure
2. `annealtime=4` forced the latent space into a Gaussian by epoch 6, before meaningful representations had formed
3. All 27 keypoints included — 7 have mean likelihood <0.5 and corrupted training with noise/NaNs

**Files:** `vame.ipynb`, `VamePipeline.ipynb`, `femaleGOVehicleOFT.ipynb`, `OpenFieldTest.csv`, `OpenFieldHmm15.csv`

---

## 02_hpo-study-mar2026/ — Optuna HPO + Diagnosis

**Dates:** Mar 18–31, 2026

**What was tried:** Systematically found better hyperparameters using Optuna TPE (Tree-structured Parzen Estimator). 10 trials × 30 epochs each. Persisted to `vame_hpo.db` (resumable). A second study on egocentric-aligned data was persisted to `vame_hpo_ego.db`.

**Search space:** `zdims`, `lr`, `beta`, `annealtime`, `kl_start`, `dropout`, `prediction_steps`, `hidden_size`

**Key finding — Trial 6 (best):** test loss 702 vs baseline 1274 — a **45% improvement** at only 30 epochs.
- Long `annealtime=46` was the single most important factor (confirmed by importance plot)
- Short annealing was the root cause of Run 1 failure
- Top trials also used larger `zdims` (37–49), non-zero dropout (0.1–0.3), and delayed `kl_start`

**Also done:** Full 500-epoch retrain with Trial 6 params. Motif videos showed most motifs still looked similar — traced to non-egocentric preprocessing (`egocentric_data: false`, preprocessing not confirmed as run). The model was encoding arena position rather than body posture.

**Learned:** Egocentric alignment is non-negotiable. Without it, the latent space cannot distinguish behavioral postures.

**Files:** `vame_hpo.ipynb`, `vame_hpo.db`, `vame_hpo_ego.db`, `vame_hpo_history.png`, `vame_hpo_importance.png`, `vame_loss_analysis.png`, `vame_new_model_losses.png`, `VAME_OFT_Report.md`, `eightmotifs.ipynb`, `EvalImage.png`, `motif_usage_comparison.png`, `motif_usage_n_comparison.png`, `motif_usage_newmodel_comparison.png`

---

## 03_run2-5pt-apr2026/ — Egocentric + 5 Keypoints (Apr 1, 2026)

**Dates:** Apr 1, 2026

**What was tried:** Complete overhaul of the input pipeline.
1. Ran `vame.preprocessing()` with egocentric alignment (`centered=mouse_center`, `orientation=tail_base`)
2. Reduced to 4 high-confidence keypoints (`mouse_center`, `tail_base`, `head_midpoint`, `neck` — all >0.80 mean likelihood)
3. Fixed `mse_reconstruction_reduction` from `"sum"` to `"mean"` — this was the critical bug
4. Reduced `beta` from 2.44 → 0.1 (with 5 features + mean reduction, higher beta causes posterior collapse)

**Why 5 features (not 4 keypoints × 2 = 8):** After egocentric alignment, `mouse_center` becomes (0,0) and `tail_base_x` becomes a constant — both excluded. Final features: `neck_x`, `neck_y`, `head_midpoint_x`, `head_midpoint_y`, `tail_base_y`.

**Config:** `beta=0.1`, `zdims=4`, `annealtime=100`, `kl_start=6`, `model_convergence=50`

**Result: Premature early stopping.** Training stopped at epoch 50. KL/dim = 0.376 (vs 0.004 in Run 1 — the MSE/KL fix worked!) but `model_convergence=50` fired before KL annealing finished.

**Root cause:** `model_convergence=50` < `kl_start + annealtime = 106`. Early stopping triggered before the KL ramp even completed, so the model never trained under full KL pressure.

**Rule learned: `model_convergence` must always be > `kl_start + annealtime`.**

**Files:** `vame_fix.ipynb`, `vame_mse_kl_ratio.png`, `Good model reconstruction.png`, `motif_usage_imputed_comparison.png`

---

## 04_run3-fixed-convergence-apr2026/ — Corrected Convergence (Apr 2–21, 2026)

**Dates:** Apr 2–21, 2026

**What was tried:** Same config as Run 2 with one fix: `model_convergence=150`.

**Rule applied:** `model_convergence=150 > kl_start + annealtime = 106`, giving 50 post-ramp epochs before early stopping can fire.

**Config (VAME_5pt — current best):**

| Parameter | Value | Reason |
|-----------|-------|--------|
| `zdims` | 4 | Matches input dimensionality |
| `beta` | 0.1 | Prevents posterior collapse with 5 features |
| `annealtime` | 100 | Slow KL ramp — latent forms before Gaussian pressure |
| `kl_start` | 6 | Short warm-up before annealing |
| `model_convergence` | 150 | > kl_start + annealtime (106) |
| `hidden_size` | 256 | Sufficient for 5-feature input |
| `mse_reconstruction_reduction` | `"mean"` | Critical bug fix |
| `mse_prediction_reduction` | `"mean"` | Same fix |

**Status:** In progress / most recent run. Training pace ~155 sec/epoch on RTX 5060 Ti.

**Files:** `vame_train_5pt.ipynb`

---

## Known Bugs in VAME 0.12.0

| Bug | Symptom | Fix |
|-----|---------|-----|
| `mse_reduction="sum"` (default) | MSE ~1400× KL; VAE acts as plain autoencoder | Set both MSE reductions to `"mean"` |
| `beta_norm: True` ignored | Setting has no effect in `rnn_vae.py` | Set `beta` directly |
| `run_lowconf_cleaning=True` | Crashes `savgol_filter` — many keypoints have >80% frames below threshold | Always pass `run_lowconf_cleaning=False` |
| `model_convergence` too short | Early stopping before KL ramp completes | Enforce `model_convergence > kl_start + annealtime` |
| Config not reloaded after `create_trainset` | `num_features` not updated in memory | Reload config dict after `create_trainset()` |
| ALL functions require `config: dict` | Passing a path string crashes silently | Load with `yaml.safe_load()`, pass the dict |

---

## Keypoint Selection

DLC tracks 27 keypoints. Many are unreliable across 38 sessions:

| Tier | Keypoints | Decision |
|------|-----------|----------|
| Critical (<0.5 likelihood) | tail5, tail4, mid_backend3, mid_backend2, tail1, right_shoulder, left_shoulder | Excluded |
| Moderate (0.5–0.69) | mid_backend, mid_back, ear tips, tail2 | Excluded |
| Good (≥0.70) | nose, head_midpoint, neck, mouse_center, tail_base, paws | Candidates |

**Used for training:** `mouse_center` (egocentric center), `tail_base` (orientation), `head_midpoint`, `neck` — all >0.80 mean likelihood across all 38 sessions. After egocentric alignment → **5 features.**

---

## What Is NOT in This Repo

Raw data and outputs are excluded (too large for git, reproducible from source):
- Raw `.mp4` videos and `.h5` DLC files
- Processed `.nc` egocentric files
- Training `.npy` arrays
- Model `.pkl` weights and snapshots
- Segmentation results (~20 GB of per-motif video clips)
