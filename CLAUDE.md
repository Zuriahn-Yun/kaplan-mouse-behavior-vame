# VAME Project — Claude Instructions

## Project Overview
Mouse behavioral segmentation pipeline: DeepLabCut (DLC) → VAME
- VAME version: 0.12.0
- Model: RNN-VAE with GRU, HMM segmentation
- 38 sessions, Open Field Test (OFT), mouse subjects
- Project path: `C:\Users\zuria\Kaplan\Vame\Open-Field-Test\`
- Config: `Open-Field-Test/config.yaml`

## Correct VAME Pipeline Order (DO NOT SKIP STEPS)

```
1. vame.preprocessing()        ← REQUIRED before create_trainset
2. vame.create_trainset()
3. vame.train_model()
4. vame.evaluate_model()
5. vame.segment_session()
6. vame.community()            ← REQUIRED after segment_session
7. vame.motif_videos()
8. visualize_umap()
```

## VAME 0.12.0 API Rules
- ALL functions take `config: dict` — never a path string
- Load config with `yaml.safe_load()`, save with `yaml.dump()`
- Always reload config after `create_trainset()` — it updates `num_features`

## Known Bugs / Critical Notes

### beta_norm Does Nothing
`beta_norm: True` in config.yaml is NOT read by `rnn_vae.py` in VAME 0.12.0.
Set `beta` directly to the correct value.

### mse_reconstruction_reduction Must Be "mean"
Default `sum` causes MSE ~1400x larger than KL → VAE acts as plain autoencoder.
Always use `mean` for both `mse_reconstruction_reduction` and `mse_prediction_reduction`.

### Posterior Collapse Warning
Signs: KL flatlines at ~0.05, KL/dim < 0.01, both MSE and KL stop changing.
Cause: beta too high for the number of features.
Fix: reduce beta. With 5 features + mean reduction, use beta=0.1.
Confirmed fixed: Run 2 (Apr 2026) held KL/dim at 0.376 vs 0.004 on old run.

### model_convergence Must Be Greater Than kl_start + annealtime
Early stopping fires when test loss doesn't improve for model_convergence epochs.
The KL ramp doesn't finish until epoch (kl_start + annealtime). If model_convergence
is shorter than this, training stops before the VAE ever experiences full KL pressure.
Rule: model_convergence > kl_start + annealtime
Example: kl_start=6, annealtime=100 → KL fully active at epoch 106.
         model_convergence must be > 106, use 150 to give 50 post-ramp epochs.
         Setting 50 (as we did in Run 2) caused premature stopping at epoch 50,
         best model saved before KL had meaningful effect.

### preprocessing Must Use run_lowconf_cleaning=False
CRITICAL keypoints (tail5, shoulders, etc.) have 90%+ frames below confidence
threshold. lowconf_cleaning clears entire series to NaN → savgol_filter crashes
with ValueError. Skip it: vame.preprocessing(..., run_lowconf_cleaning=False).
Our 4 training keypoints all have >0.80 likelihood so this step adds no value.

## Current Model: VAME_5pt
- 4 keypoints: mouse_center, tail_base, head_midpoint, neck (all >0.80 likelihood)
- 5 features after egocentric exclusions: neck_x, neck_y, head_midpoint_x, head_midpoint_y, tail_base_y
- Preprocessing reference: centered=mouse_center, orientation=tail_base
- Key config: beta=0.1, zdims=4, annealtime=100, hidden=256, model_convergence=150

## Keypoint Likelihood Summary (mean across 38 sessions)
CRITICAL (<0.5): tail5(0.076), tail4(0.181), mid_backend3(0.292), mid_backend2(0.309),
                 tail1(0.328), right_shoulder(0.357), left_shoulder(0.417)
HIGH (0.5-0.69): mid_backend(0.531), mid_back(0.596), right_ear_tip(0.604),
                 left_ear_tip(0.624), tail2(0.650)
OK (≥0.70): everything else

## File Locations
- Raw DLC data:      Open-Field-Test/data/raw/        (.h5, .mp4, .nc, .csv)
- Processed data:    Open-Field-Test/data/processed/  (_processed.nc files)
- Training data:     Open-Field-Test/data/train/      (train_seq.npy, test_seq.npy)
- Model weights:     Open-Field-Test/model/best_model/
- Loss files:        Open-Field-Test/model/model_losses/  (*.npy, named by model_name)
- Results:           Open-Field-Test/results/<session>/<model_name>/
- Logs:              Open-Field-Test/logs/

## Loss Monitoring
Loss files: `model/model_losses/{train,test,mse_train,kl,weight_values,kmeans_losses,fut_losses}_{model_name}.npy`
Also check: `logs/train_model.log` for full epoch output and convergence messages.
Also check: `model/best_model/` to confirm checkpoint is being saved.

Healthy training signs after KL fully active (epoch > kl_start + annealtime):
- KL per dimension > 0.1  (zdims = cfg["zdims"])
- MSE/KL ratio between 1–10  (use actual beta from config, not 2.44)
- Both MSE and KL still moving (not flatlined)
- KL std over last 10 epochs > 0.001

Training pace reference: ~155 sec/epoch on RTX 5060 Ti with 5 features, 38 sessions.

## Run History
| Run | Date | Result | Key Issue |
|-----|------|--------|-----------|
| Run 1 (original) | Feb 2026 | Posterior collapse | beta=2.44 + sum reduction, MSE/KL ratio 155-325x |
| Run 2 | Apr 1 2026 | Premature stop at epoch 50 | model_convergence=50 < kl_start+annealtime=106 |
| Run 3 | Apr 2 2026 | In progress | beta=0.1, zdims=4, model_convergence=150 |
