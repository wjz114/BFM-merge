# LEAF Model Merging POC — Implementation Details and Results (seed 0)

Run date: 2026-11-03. Plan followed: `/home/jingzhou/BFM-merge/LEAF_Model_Merging_POC_Plan.md` (Chinese).
Code: `/home/jingzhou/BFM-merge/LEAF`, git branch `poc-model-merging` (uncommitted).
Every number in the tables below was generated from the run artifacts in `LEAF/runs/poc/` by a script; nothing was typed by hand.

---

## 0. TL;DR

- **Research Q1 — can learnable merging beat joint training from the same init? No.** Joint reaches source macro test BAcc **0.7603**. The merges reach 0.6895 (Uniform), 0.6973 (Global) and 0.7015 (Block), i.e. −7.08 / −6.30 / −5.88 pp vs Joint.
- **Research Q2 — is finer-grained merging better than global? Weakly yes.** The ordering Uniform < Global < Block holds on both validation (0.7062 / 0.7203 / 0.7248) and test. The steps are small (+0.78 pp and +0.42 pp on test) relative to the ~6 pp gap to Joint. This is a single seed, with no confidence intervals.
- **The experts learned real gains on their own task but conflict with each other.** Every expert beats Base on its own test set (+5.7 to +28.5 pp). Almost every expert falls *below Base* on the other source tasks (Section 4.3). SSVEP is the task hurt most by merging.
- The 5-model **specialist reference** (each expert on its own dataset) reaches 0.7730. That is +1.28 pp over Joint, but it uses 5 separate models.
- **External MI (MI_Weibo2014):** Base 0.6228, Joint 0.6481, Uniform 0.6108, Global 0.6272, Block 0.6158. Merging does not improve external MI transfer over Base; Joint does (+2.5 pp).
- **Budget is probably not converged.** The best checkpoint for the OpenBMI and Cho2017 experts and for Joint is the final evaluation point; for SSVEP it is round 900 of 1,000. The plan's rule for this case is to raise the expert and joint budgets together before adding algorithms.
- **Plan decision-table outcome:** "Joint 更好 → 先看专家是否学到增益及训练曲线，不立即叠加复杂算法" (if Joint is better, first check whether the experts learned gains and look at the training curves; don't stack more complex algorithms yet).

---

## 1. Goal and research questions (from the plan)

Fix the full LEAF path (EEG Tower + Q-Former + cosine readout against fixed text prototypes). From one shared initialization, compare:

- independent experts;
- joint multi-dataset fine-tuning;
- uniform task-vector merging;
- global learnable merging (one coefficient per expert);
- block-wise learnable merging (one coefficient per expert per structural block).

Q1: Can learnable merging beat joint training under the same initialization? Q2: Is finer-grained merging better than global merging?

No new modules were added: no MLP, linear head, or dataset-specific trainable head. The Q-Former and its `output_embed` are kept. The instruction and label prototypes are fixed.

---

## 2. Setup

### 2.1 Environment and user decisions

| Item | Value |
|---|---|
| Initialization θ₀ | `LEAF/checkpoints/leaf-v1.0-instruct-mpnet-base.ckpt`: the public v1.0 GitHub release, downloaded because no local copy existed (user decision). sha256 `f9ca9cabbd734d2691fa6748adf3e78056f895d4031f7edc481e7d7b6822f3fa` |
| Model config | `configs/LEAF_mpnet.yaml`: window 100, dim_cnn 64, dim_token 256, two 12-layer Tower branches, 8 queries, 4 Q-Former layers, MPNet text dim 768. 26.42 M parameters |
| Data | `/home/jingzhou/data/LEAF/Downstream/*.h5`. `configs/env.yaml` `paths.downstream` was edited to point here (the files are identical in size to `/media/public/LEAF/Downstream`) |
| Hardware | 2× NVIDIA RTX PRO 5000 (48 GB). torch 2.14.0+cu130, Python 3.14 (conda base) |
| Packages | `scikit-learn 1.9.1` was pip-installed into conda base (needed by the existing metric code) |
| Precision | bf16-mixed for training and alpha optimization. Evaluation (validation and test) runs in fp32, exactly like the original `evaluate()` |
| Seed | 0 everywhere |
| External evaluation | **MI_Weibo2014 only** (user decision). MI_Dreyer2023 was skipped because the repo marks its label mapping as UNVERIFIED (its HDF5 has no legend). `eval_poc.py --external MI_Weibo2014,MI_Dreyer2023` supports it if the mapping is confirmed |

### 2.2 Datasets

| Dataset | Class order (label 0..K−1) | Instruction (blended with mean prototype) | train | val | test | Split source |
|---|---|---|---|---|---|---|
| MI_OpenBMI | Right, Left | This is a motor imagery decoding task | 13440 | 3360 | 4800 | subject split: train+val subjects → every 5th trial = val; held-out subjects = test |
| MI_Cho2017 | Left, Right | This is a motor imagery decoding task | 6464 | 1616 | 1800 | subject split: train+val subjects → every 5th trial = val; held-out subjects = test |
| MI_PhysioNet | Left, Right | This is a motor imagery decoding task | 11447 | 2862 | 5262 | subject split: train+val subjects → every 5th trial = val; held-out subjects = test |
| EMO_SEED_3_seg4 | Negative, Neutral, Positive | This is an emotion recognition task | 22545 | 7905 | 7620 | predefined trainX/validX/testX in HDF5 |
| SSVEP_OpenBMI | 12.0, 8.6, 6.6, 5.4 | This is an SSVEP frequency decoding task | 13440 | 3360 | 4800 | subject split: train+val subjects → every 5th trial = val; held-out subjects = test |
| MI_Weibo2014 (external) | Left, Right | This is a motor imagery decoding task | – | – | 1580 | all 10 subjects; verified 0=Left;1=Right (raw event codes 1/2 minus 1) |

Notes for analysis:
- **Validation for the four subject-split datasets is within-subject** (every 5th trial of the train subjects); **test is cross-subject** (held-out subjects). This is the existing `load_datasets.py` behaviour. All checkpoint and alpha selection therefore uses within-subject validation, which partly explains why val > test (Section 4.5).
- **The three MI source datasets share the same readout geometry.** They have the same instruction and the same {Left, Right} prototypes; only MI_OpenBMI's row order is [Right, Left]. Merging the three MI experts therefore combines task vectors that target identical output prototypes.
- **Base has already seen these datasets.** `b_instruct_tuning.py` instruction-tunes on every dataset in `configs/tasks.yaml`, including all five source datasets (train split; val split for validation). The experts therefore *continue* training on data θ₀ has already seen.
- **Weibo2014 and Dreyer2023 are absent from `tasks.yaml`**, so they were not in instruction tuning. The local pretraining corpus (`/media/public/LEAF/Pretrain`) lists CS_LargeSpanish, CS_ThinkOutLoud, EMO_SEED_Pretrain_{French,German,Sleep2} and MI_HeBin2021_P0–P5, so there is no Weibo there either. Whether the *released* checkpoint was pretrained on exactly this corpus is not verified. Per the plan, do not call the Weibo result "zero-shot / strictly unseen" without checking that.

---

## 3. Implementation details

### 3.1 Unified forward and loss (used identically in training, joint, alpha optimization and evaluation)

```python
embedding, _ = leaf(x, instruction[d].expand(len(x), -1))
logits = F.normalize(embedding.float(), dim=-1) @ prototypes[d].T   # autocast disabled here (fp32)
loss = F.cross_entropy(logits, y)
```

- `instruction[d]` and `prototypes[d]` come from the original `prepare_text()`. The instruction is the family's first phrasing (e.g. "This is a motor imagery decoding task"), added to the normalized mean of the candidate-label prototypes and renormalized. It is task + class-set conditioning, never the trial's label.
- Prototype row *i* equals integer label *i* in `configs/tasks.yaml` order. Label indices are never concatenated across datasets, there is no global label union, no temperature, and no text-encoder tuning.

### 3.2 `d_downstream_qformer_ce.py` (extended in place)

Kept from the original: model loading (strict), `prepare_text`, `PrototypeClassifier`, `make_loader`, the metric set and checkpoint saving.

**Interface**
- `--ds` accepts one dataset or a comma-separated list.
- `--steps-per-dataset R` switches to step mode: R rounds, where each round draws one batch per dataset in a freshly shuffled order (`np.random.default_rng(seed).permutation`), and every batch is one `optimizer.step()`. There is no gradient accumulation across datasets. Total steps = R × D.
- New flags: `--eval-every-rounds` (default 100) and `--warmup-frac` (default 0.05).
- Epoch mode (no `--steps-per-dataset`) still exists for a single dataset.

**Joint model**
- One LEAF instance. One `PrototypeClassifier` wrapper per dataset, all sharing that LEAF, so only the instruction/prototype buffers switch between batches.
- Each dataset's train loader is wrapped in an endless `BatchStream`, which rebuilds the iterator when it is exhausted. Train loaders use `shuffle=True, drop_last=False` (the original setting).

**Optimizer and schedule**
- AdamW with weight decay 1e-3. Tokenizer LR = 1e-3 (`lr_base 1e-3 × fast 1.0`); all other parameters LR = 1e-4 (`× slow 0.1`).
- `cosine_annealing_lr(1.0, 0.1, total_steps, 1, warmup_steps)`: a linear warmup from 0 over 5% of steps (50 for an expert, 250 for joint), then cosine decay to 0.1× the peak, stepped per optimizer step.

**BatchNorm**
- After every `model.train()`, all `nn.SyncBatchNorm` modules are set back to `eval()`. Running statistics therefore stay at θ₀; the affine parameters still train, and dropout behaves normally.
- `merge_poc.py` verifies this policy: every expert buffer equals θ₀ bit-for-bit.

**Validation and selection**
- Round 0 (the untouched θ₀) is validated and logged for the curves **but is not a checkpoint candidate** (implementation choice). After that, validation runs every 100 rounds, i.e. 10 candidates per run.
- Selection key: (dataset-macro val BAcc, −macro val CE), so ties go to the lower CE. For an expert, the macro is over its single dataset. The best state dict is saved to `best.ckpt` with an atomic rename.
- **The test split is never loaded for use during training.** The original end-of-script test call was removed; all test evaluation happens in `eval_poc.py`.

**Outputs per run**
- `best.ckpt`: a plain LEAF state dict.
- `run.json`: config, LRs, schedule, split sizes, θ₀ sha256, best round, best/source val, actual step counts per dataset, timings.
- `history.json`: val metrics per dataset at every eval point, plus mean train CE per interval.
- `metrics.npz`: per-step loss, LR and dataset id, plus val curves.
- `text_readout.npz`: per-dataset labels, instruction and prototypes, with keys `<ds>__labels` etc.

**Behaviour changes vs the original script (they also apply to epoch mode):** frozen BN statistics, no test evaluation, a changed `text_readout.npz` key format, and `evaluate()` split into `classification_metrics()` + `evaluate()` with identical metrics.

### 3.3 `merge_poc.py`

**Formula.** Task vectors are δ_k = θ_k − θ₀, computed only over the selected `named_parameters()`. For each parameter group g, θ_M[g] = θ₀[g] + Σ_k α[k, g] · δ_k[g].

**Coefficient groups (block mode = 33 groups × 5 experts = 165 coefficients)**
- **Input:** `tower.tokenizer.*` + `tower.pos_embedding`.
- **Branch outer groups:** `tower.{mask,next}_inProj` + `tower.{mask,next}_transformer.norm`.
- **Transformer blocks:** each `tower.{mask,next}_transformer.layers.i` is its own group (12 + 12).
- **Q-Former conditioning:** `qformer.base_queries` + `qformer.pos_embedding` + `qformer.film`.
- **Q-Former blocks:** each `qformer.layers.i` is its own group (4).
- **Output:** `qformer.final_norm` + `qformer.output_embed`.
- Within a block, attention, FFN, biases and LayerNorms share one coefficient. The group count is asserted at runtime: 3 + 2·12 + 1 + 4 + 1 = 33.
- In global and uniform mode all selected tensors form one group.

**Parameters, buffers and constants**
- 383 tensors (25,751,552 parameters) are merged.
- The unused tensors stay at θ₀ (verified identical in every expert): `tower.mask_token`, `tower.mask_outProj.0.weight`, `tower.mask_outProj.0.bias`, `tower.mask_outProj.3.weight`, `tower.mask_outProj.3.bias`, `tower.next_outProj.0.weight`, `tower.next_outProj.0.bias`, `tower.next_outProj.3.weight`, `tower.next_outProj.3.bias`.
- All buffers (`causal_mask` and the BN running stats) are copied from θ₀; no state_dict-wide subtraction is done, so the −inf entries of `causal_mask` are never subtracted.
- θ₀ and the stacked deltas, shape (K, …) in fp32 on GPU, are constants. Only α is a leaf tensor.

**Loading checks**
- Each expert's `run.json` must have `datasets == [ds]` and the same θ₀ sha256.
- State-dict keys and shapes must match.
- Buffers and unused params must equal θ₀ exactly.

**Coefficients**
- Initialized to 1/K = 0.2, so global and block start exactly at uniform.
- They are unconstrained reals: no softmax, gate, mask or regularizer.
- Uniform performs 0 updates and only validates step 0.

**Differentiable forward**
- `compose(alpha)` is computed in fp32 *outside* autocast: `θ₀[n] + tensordot(α[:, g(n)], δ[n], 1)`.
- The forward runs `torch.func.functional_call(leaf, (merged_params, base_buffers), (x, instruction))` under bf16 autocast. `leaf.eval()` turns dropout off, and nothing is wrapped in `no_grad` during optimization.
- No `load_state_dict` or `.data.copy_` happens inside the loop.

**Calibration set**
- At most 512 samples per dataset, drawn from that dataset's `load_dataset()` train split.
- Stratified by class with largest-remainder allocation. Not stratified by subject, because `load_dataset` returns no subject IDs.
- RNG is `default_rng([seed, dataset_index])`, so the indices are identical across uniform, global and block (verified with `cmp`). Saved to `calibration_indices.npz`.
- All 5 datasets have > 512 train samples, so each calibration set is exactly 512.

**Optimization**
- Adam with LR 1e-2 and weight decay 0, for 500 total steps = 100 equal-frequency rounds × 5 datasets, with dataset order shuffled every round. Each step uses one calibration batch (bs 128) from a single dataset.
- Validation (fp32, `no_grad`, functional) runs at step 0 (the uniform init) and every 50 steps.
- Selection uses the same key as training: max macro val BAcc, ties broken by min macro val CE.

**Folding**
- The best α is folded into an ordinary LEAF state dict and saved as `merged.ckpt`. That file is then reloaded strictly into a fresh LEAF and re-validated with the standard `PrototypeClassifier` + `evaluate()` path. It reproduced the selected validation scores exactly (Section 4.6).
- Inference keeps no expert list, router or ensemble.

**Outputs**
- `merged.ckpt` and `best_alpha.npy`.
- `alpha.csv` (groups × experts) and `alpha_heatmap.png` (diverging blue–gray–red, centered at 1/K).
- `calibration_indices.npz`, `history.json`, `metrics.npz` (including `alpha_history` at every eval point) and `run.json`.

### 3.4 `eval_poc.py`

- **Models:** Base, 5 experts, Joint, Uniform, Global, Block. Each is a plain LEAF checkpoint, loaded strictly.
- **Test data:** source datasets use `load_test_split()`. External MI_Weibo2014 uses all 10 subjects, the Motor instruction, and prototypes [Left, Right]; its label mapping 0=Left/1=Right is verified from the raw event codes.
- **Metrics:** BAcc, Cohen's κ, AUROC (binary: P(class 1); multiclass: macro one-vs-rest), weighted F1 and CE. These come from the original `evaluate()`, in fp32.
- **Aggregates:**
  - `SOURCE_MACRO`: the equal-weight mean over the 5 source test sets.
  - `dBAcc_vs_Joint_pp`: the BAcc difference to Joint on the same row dataset.
  - "Specialist": each expert on its own dataset, labelled as a 5-model reference.
- **Outputs:** `runs/poc/results.csv` (long format, 76 rows) and `runs/poc/results.json` (checkpoint sha256s, split and mapping notes).

### 3.5 Commands actually run

```bash
cd /home/jingzhou/BFM-merge/LEAF
BASE=checkpoints/leaf-v1.0-instruct-mpnet-base.ckpt
DATASETS=MI_OpenBMI,MI_Cho2017,MI_PhysioNet,EMO_SEED_3_seg4,SSVEP_OpenBMI
# experts (sequential, GPU 0)
for DS in ${DATASETS//,/ }; do
  python d_downstream_qformer_ce.py --ckpt "$BASE" --ds "$DS" --steps-per-dataset 1000 \
    --bs 128 --seed 0 --gpu 0 --outdir "runs/poc/experts/$DS"; done
# joint (GPU 1, concurrently with the experts)
python d_downstream_qformer_ce.py --ckpt "$BASE" --ds "$DATASETS" --steps-per-dataset 1000 \
  --bs 128 --seed 0 --gpu 1 --outdir runs/poc/joint
# merges (uniform then global on GPU 0, block on GPU 1)
for METHOD in uniform global block; do
  python merge_poc.py --base "$BASE" --experts-dir runs/poc/experts --datasets "$DATASETS" \
    --method $METHOD --calibration-per-dataset 512 --steps 500 --lr 1e-2 --bs 128 --seed 0 \
    --gpu 0 --outdir runs/poc/merge_$METHOD; done
python eval_poc.py --base "$BASE" --run-dir runs/poc --datasets "$DATASETS" --external MI_Weibo2014 \
  --bs 128 --gpu 0 --output runs/poc/results.csv
```

`LEAF/run_poc.sh` contains the same pipeline as a single script (set `SEED` and `RUN` for further seeds).

### 3.6 Implementation choices not fixed by the plan

1. Round 0 (θ₀) is logged but is not a checkpoint candidate in training; it *is* a candidate in merging (step 0 = uniform), as the plan specifies.
2. Evaluation runs in fp32 (as the original `evaluate()`); training and alpha optimization use bf16 autocast. Merged parameters are composed in fp32 outside autocast.
3. Calibration is stratified by class only (no subject IDs are available from `load_dataset`).
4. Train loaders keep `drop_last=False`, so the last batch of each pass can be smaller than 128.
5. The dataset order RNG and the DataLoader shuffle RNG are both seeded from `--seed`. They are not bit-reproducible across GPUs, because cuDNN is non-deterministic and `cudnn.benchmark=False`.
6. Uniform ignores `--steps` and records 0 alpha updates.

---

## 4. Results

Dataset short names: OpenBMI = MI_OpenBMI, Cho2017 = MI_Cho2017, PhysioNet = MI_PhysioNet, SEED-3 = EMO_SEED_3_seg4, SSVEP = SSVEP_OpenBMI, Weibo (ext.) = MI_Weibo2014. "Source macro" is the equal-weight mean over the five source test sets; Weibo is never included in it.

### 4.1 Main result — test balanced accuracy

| Model | OpenBMI | Cho2017 | PhysioNet | SEED-3 | SSVEP | Source macro | Weibo (ext.) | Δ macro vs Joint (pp) |
|---|---|---|---|---|---|---|---|---|
| Base | 0.7321 | 0.7094 | 0.6900 | 0.5891 | 0.5523 | 0.6546 | 0.6228 | -10.57 |
| Expert[OpenBMI] | 0.7890 | 0.7367 | 0.6682 | 0.5379 | 0.4692 | 0.6402 | 0.6424 | -12.01 |
| Expert[Cho2017] | 0.7015 | 0.8072 | 0.5658 | 0.5138 | 0.4187 | 0.6014 | 0.6361 | -15.89 |
| Expert[PhysioNet] | 0.6933 | 0.6800 | 0.7647 | 0.5190 | 0.4594 | 0.6233 | 0.5854 | -13.70 |
| Expert[SEED-3] | 0.6204 | 0.6167 | 0.5661 | 0.6671 | 0.5098 | 0.5960 | 0.5462 | -16.42 |
| Expert[SSVEP] | 0.6771 | 0.6850 | 0.5741 | 0.5592 | 0.8371 | 0.6665 | 0.5468 | -9.38 |
| Joint | 0.7819 | 0.7917 | 0.7461 | 0.6866 | 0.7950 | 0.7603 | 0.6481 | +0.00 |
| Uniform | 0.7554 | 0.7828 | 0.6807 | 0.6166 | 0.6119 | 0.6895 | 0.6108 | -7.08 |
| Global | 0.7540 | 0.7617 | 0.6631 | 0.6449 | 0.6629 | 0.6973 | 0.6272 | -6.30 |
| Block | 0.7629 | 0.7739 | 0.7032 | 0.6548 | 0.6127 | 0.7015 | 0.6158 | -5.88 |
| Specialist (5 models) | 0.7890 | 0.8072 | 0.7647 | 0.6671 | 0.8371 | 0.7730 | – | +1.28 |

### 4.2 Difference to Joint, test BAcc (percentage points)

| Model | OpenBMI | Cho2017 | PhysioNet | SEED-3 | SSVEP | Source macro | Weibo (ext.) |
|---|---|---|---|---|---|---|---|
| Base | -4.98 | -8.23 | -5.61 | -9.75 | -24.27 | -10.57 | -2.53 |
| Expert[OpenBMI] | +0.71 | -5.50 | -7.79 | -14.87 | -32.58 | -12.01 | -0.57 |
| Expert[Cho2017] | -8.04 | +1.55 | -18.03 | -17.28 | -37.63 | -15.89 | -1.20 |
| Expert[PhysioNet] | -8.86 | -11.17 | +1.86 | -16.76 | -33.56 | -13.70 | -6.27 |
| Expert[SEED-3] | -16.15 | -17.50 | -18.00 | -1.95 | -28.52 | -16.43 | -10.19 |
| Expert[SSVEP] | -10.48 | -10.67 | -17.20 | -12.74 | +4.21 | -9.38 | -10.13 |
| Joint | +0.00 | +0.00 | +0.00 | +0.00 | +0.00 | +0.00 | +0.00 |
| Uniform | -2.65 | -0.89 | -6.54 | -7.00 | -18.31 | -7.08 | -3.73 |
| Global | -2.79 | -3.00 | -8.30 | -4.17 | -13.21 | -6.30 | -2.09 |
| Block | -1.90 | -1.78 | -4.29 | -3.18 | -18.23 | -5.88 | -3.23 |
| Specialist (5 models) | +0.71 | +1.55 | +1.86 | -1.95 | +4.21 | +1.27 | – |

### 4.3 Expert cross-dataset matrix — test BAcc (in parentheses: pp change vs Base)

Rows are expert checkpoints and columns are test sets. The diagonal shows the expert's own task.

| Expert ↓ / test set → | OpenBMI | Cho2017 | PhysioNet | SEED-3 | SSVEP | Weibo (ext.) |
|---|---|---|---|---|---|---|
| Base | 0.7321 | 0.7094 | 0.6900 | 0.5891 | 0.5523 | 0.6228 |
| OpenBMI | 0.7890 (+5.7) | 0.7367 (+2.7) | 0.6682 (-2.2) | 0.5379 (-5.1) | 0.4692 (-8.3) | 0.6424 (+2.0) |
| Cho2017 | 0.7015 (-3.1) | 0.8072 (+9.8) | 0.5658 (-12.4) | 0.5138 (-7.5) | 0.4187 (-13.4) | 0.6361 (+1.3) |
| PhysioNet | 0.6933 (-3.9) | 0.6800 (-2.9) | 0.7647 (+7.5) | 0.5190 (-7.0) | 0.4594 (-9.3) | 0.5854 (-3.7) |
| SEED-3 | 0.6204 (-11.2) | 0.6167 (-9.3) | 0.5661 (-12.4) | 0.6671 (+7.8) | 0.5098 (-4.2) | 0.5462 (-7.7) |
| SSVEP | 0.6771 (-5.5) | 0.6850 (-2.4) | 0.5741 (-11.6) | 0.5592 (-3.0) | 0.8371 (+28.5) | 0.5468 (-7.6) |

### 4.4 Other test metrics

**Cohen's κ**

| Model | OpenBMI | Cho2017 | PhysioNet | SEED-3 | SSVEP | Source macro | Weibo (ext.) |
|---|---|---|---|---|---|---|---|
| Base | 0.4642 | 0.4189 | 0.3801 | 0.3954 | 0.4031 | 0.4123 | 0.2456 |
| Expert[OpenBMI] | 0.5779 | 0.4733 | 0.3364 | 0.3185 | 0.2922 | 0.3997 | 0.2848 |
| Expert[Cho2017] | 0.4029 | 0.6144 | 0.1315 | 0.2789 | 0.2250 | 0.3305 | 0.2722 |
| Expert[PhysioNet] | 0.3867 | 0.3600 | 0.5295 | 0.2872 | 0.2792 | 0.3685 | 0.1709 |
| Expert[SEED-3] | 0.2408 | 0.2333 | 0.1323 | 0.5035 | 0.3464 | 0.2913 | 0.0924 |
| Expert[SSVEP] | 0.3542 | 0.3700 | 0.1482 | 0.3501 | 0.7828 | 0.4011 | 0.0937 |
| Joint | 0.5637 | 0.5833 | 0.4922 | 0.5354 | 0.7267 | 0.5803 | 0.2962 |
| Uniform | 0.5108 | 0.5656 | 0.3615 | 0.4355 | 0.4825 | 0.4712 | 0.2215 |
| Global | 0.5079 | 0.5233 | 0.3261 | 0.4751 | 0.5506 | 0.4766 | 0.2544 |
| Block | 0.5258 | 0.5478 | 0.4063 | 0.4881 | 0.4836 | 0.4903 | 0.2316 |
| Specialist (5 models) | 0.5779 | 0.6144 | 0.5295 | 0.5035 | 0.7828 | 0.6016 | – |

**AUROC** (binary: P(class 1); 3/4-class: macro one-vs-rest)

| Model | OpenBMI | Cho2017 | PhysioNet | SEED-3 | SSVEP | Source macro | Weibo (ext.) |
|---|---|---|---|---|---|---|---|
| Base | 0.8054 | 0.8173 | 0.7621 | 0.7579 | 0.7909 | 0.7867 | 0.6971 |
| Expert[OpenBMI] | 0.8701 | 0.8180 | 0.7273 | 0.7450 | 0.7111 | 0.7743 | 0.7030 |
| Expert[Cho2017] | 0.7667 | 0.8904 | 0.5853 | 0.7250 | 0.6699 | 0.7275 | 0.7199 |
| Expert[PhysioNet] | 0.7578 | 0.7110 | 0.8401 | 0.7107 | 0.7096 | 0.7458 | 0.6109 |
| Expert[SEED-3] | 0.7131 | 0.7261 | 0.6515 | 0.8023 | 0.7602 | 0.7306 | 0.5928 |
| Expert[SSVEP] | 0.7271 | 0.7703 | 0.6148 | 0.7351 | 0.9313 | 0.7557 | 0.5815 |
| Joint | 0.8553 | 0.8707 | 0.8226 | 0.8101 | 0.9181 | 0.8554 | 0.7171 |
| Uniform | 0.8359 | 0.8730 | 0.7559 | 0.7535 | 0.8254 | 0.8087 | 0.6953 |
| Global | 0.8308 | 0.8575 | 0.7446 | 0.7704 | 0.8503 | 0.8107 | 0.6855 |
| Block | 0.8438 | 0.8603 | 0.7836 | 0.7868 | 0.8283 | 0.8206 | 0.6829 |
| Specialist (5 models) | 0.8701 | 0.8904 | 0.8401 | 0.8023 | 0.9313 | 0.8668 | – |

**Weighted F1**

| Model | OpenBMI | Cho2017 | PhysioNet | SEED-3 | SSVEP | Source macro | Weibo (ext.) |
|---|---|---|---|---|---|---|---|
| Base | 0.7301 | 0.6989 | 0.6856 | 0.5818 | 0.5302 | 0.6453 | 0.5967 |
| Expert[OpenBMI] | 0.7889 | 0.7361 | 0.6658 | 0.5218 | 0.4505 | 0.6326 | 0.6417 |
| Expert[Cho2017] | 0.7009 | 0.8066 | 0.5638 | 0.4992 | 0.4013 | 0.5943 | 0.6344 |
| Expert[PhysioNet] | 0.6933 | 0.6800 | 0.7644 | 0.5070 | 0.4651 | 0.6220 | 0.5827 |
| Expert[SEED-3] | 0.5749 | 0.5746 | 0.4919 | 0.6726 | 0.4887 | 0.5605 | 0.5032 |
| Expert[SSVEP] | 0.6729 | 0.6849 | 0.5691 | 0.5509 | 0.8387 | 0.6633 | 0.5425 |
| Joint | 0.7817 | 0.7915 | 0.7454 | 0.6913 | 0.7971 | 0.7614 | 0.6472 |
| Uniform | 0.7535 | 0.7819 | 0.6784 | 0.6154 | 0.5987 | 0.6856 | 0.6023 |
| Global | 0.7527 | 0.7612 | 0.6623 | 0.6483 | 0.6608 | 0.6970 | 0.6243 |
| Block | 0.7617 | 0.7736 | 0.7031 | 0.6602 | 0.6169 | 0.7031 | 0.6127 |
| Specialist (5 models) | 0.7889 | 0.8066 | 0.7644 | 0.6726 | 0.8387 | 0.7742 | – |

**Test cross-entropy** (cosine logits have no temperature, so CE magnitudes are compressed; compare within a dataset only)

| Model | OpenBMI | Cho2017 | PhysioNet | SEED-3 | SSVEP | Source macro | Weibo (ext.) |
|---|---|---|---|---|---|---|---|
| Base | 0.5828 | 0.5893 | 0.6096 | 0.9747 | 1.2965 | 0.8106 | 0.6544 |
| Expert[OpenBMI] | 0.5004 | 0.5508 | 0.6353 | 0.9983 | 1.3199 | 0.8009 | 0.6743 |
| Expert[Cho2017] | 0.6030 | 0.4816 | 0.7529 | 1.0178 | 1.3415 | 0.8394 | 0.6486 |
| Expert[PhysioNet] | 0.6019 | 0.6204 | 0.5287 | 1.0130 | 1.3342 | 0.8196 | 0.7235 |
| Expert[SEED-3] | 0.6566 | 0.6560 | 0.6938 | 0.8855 | 1.3135 | 0.8411 | 0.7011 |
| Expert[SSVEP] | 0.6232 | 0.6092 | 0.6791 | 0.9935 | 1.0653 | 0.7941 | 0.6866 |
| Joint | 0.5104 | 0.4964 | 0.5468 | 0.8670 | 1.0843 | 0.7010 | 0.6495 |
| Uniform | 0.5364 | 0.5131 | 0.6065 | 0.9432 | 1.2249 | 0.7648 | 0.6692 |
| Global | 0.5369 | 0.5233 | 0.6144 | 0.9098 | 1.1735 | 0.7516 | 0.6706 |
| Block | 0.5277 | 0.5122 | 0.5865 | 0.8982 | 1.1803 | 0.7410 | 0.6846 |
| Specialist (5 models) | 0.5004 | 0.4816 | 0.5287 | 0.8855 | 1.0653 | 0.6923 | – |

### 4.5 Validation vs test

| Model | selected at | val macro BAcc (selected) | test macro BAcc | val − test (pp) |
|---|---|---|---|---|
| Expert[OpenBMI] (own dataset) | round 1000 | 0.8484 | 0.7890 | +5.94 |
| Expert[Cho2017] (own dataset) | round 1000 | 0.7649 | 0.8072 | -4.23 |
| Expert[PhysioNet] (own dataset) | round 700 | 0.7996 | 0.7647 | +3.49 |
| Expert[SEED-3] (own dataset) | round 500 | 0.7203 | 0.6671 | +5.32 |
| Expert[SSVEP] (own dataset) | round 900 | 0.9036 | 0.8371 | +6.65 |
| Joint | round 1000 | 0.7894 | 0.7603 | +2.91 |
| Uniform | alpha step 0 | 0.7062 | 0.6895 | +1.67 |
| Global | alpha step 150 | 0.7203 | 0.6973 | +2.30 |
| Block | alpha step 200 | 0.7248 | 0.7015 | +2.33 |

Per dataset (val BAcc at the selected point / test BAcc):

| Model | OpenBMI val / test | Cho2017 val / test | PhysioNet val / test | SEED-3 val / test | SSVEP val / test |
|---|---|---|---|---|---|
| Base (step 0) | 0.7830 / 0.7321 | 0.6652 / 0.7094 | 0.6974 / 0.6900 | 0.5761 / 0.5891 | 0.6340 / 0.5523 |
| Joint | 0.8377 / 0.7819 | 0.7457 / 0.7917 | 0.7817 / 0.7461 | 0.7074 / 0.6866 | 0.8745 / 0.7950 |
| Uniform | 0.8079 / 0.7554 | 0.6993 / 0.7828 | 0.7288 / 0.6807 | 0.6284 / 0.6166 | 0.6666 / 0.6119 |
| Global | 0.8161 / 0.7540 | 0.6807 / 0.7617 | 0.7170 / 0.6631 | 0.6805 / 0.6449 | 0.7073 / 0.6629 |
| Block | 0.8190 / 0.7629 | 0.7092 / 0.7739 | 0.7377 / 0.7032 | 0.6885 / 0.6548 | 0.6698 / 0.6127 |
| Specialist | 0.8484 / 0.7890 | 0.7649 / 0.8072 | 0.7996 / 0.7647 | 0.7203 / 0.6671 | 0.9036 / 0.8371 |

### 4.6 Merge validation curves

The selected point is the maximum macro val BAcc; validation runs at every 50 alpha steps. Calib. CE is the mean training CE on calibration batches over the preceding interval, averaged over datasets.

**Uniform** (no optimization)

| alpha step | val macro BAcc | val macro CE | OpenBMI | Cho2017 | PhysioNet | SEED-3 | SSVEP | calib. CE (interval mean) |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.7062 | 0.7526 | 0.8079 | 0.6993 | 0.7288 | 0.6284 | 0.6666 | – |

**Global** (5 learnable coefficients)

| alpha step | val macro BAcc | val macro CE | OpenBMI | Cho2017 | PhysioNet | SEED-3 | SSVEP | calib. CE (interval mean) |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.7062 | 0.7526 | 0.8079 | 0.6993 | 0.7288 | 0.6284 | 0.6666 | – |
| 50 | 0.7147 | 0.7405 | 0.8134 | 0.6906 | 0.7303 | 0.6516 | 0.6877 | 0.6974 |
| 100 | 0.7190 | 0.7358 | 0.8152 | 0.6875 | 0.7195 | 0.6733 | 0.6998 | 0.6860 |
| 150 | 0.7203 | 0.7345 | 0.8161 | 0.6807 | 0.7170 | 0.6805 | 0.7073 | 0.6830 |
| 200 | 0.7189 | 0.7340 | 0.8138 | 0.6720 | 0.7152 | 0.6870 | 0.7062 | 0.6798 |
| 250 | 0.7196 | 0.7341 | 0.8145 | 0.6739 | 0.7160 | 0.6879 | 0.7059 | 0.6819 |
| 300 | 0.7202 | 0.7340 | 0.8151 | 0.6751 | 0.7156 | 0.6889 | 0.7060 | 0.6799 |
| 350 | 0.7202 | 0.7339 | 0.8150 | 0.6733 | 0.7178 | 0.6890 | 0.7060 | 0.6814 |
| 400 | 0.7199 | 0.7339 | 0.8142 | 0.6751 | 0.7174 | 0.6890 | 0.7039 | 0.6804 |
| 450 | 0.7196 | 0.7338 | 0.8140 | 0.6720 | 0.7177 | 0.6895 | 0.7045 | 0.6803 |
| 500 | 0.7199 | 0.7338 | 0.8140 | 0.6726 | 0.7170 | 0.6890 | 0.7066 | 0.6814 |

**Block** (165 learnable coefficients)

| alpha step | val macro BAcc | val macro CE | OpenBMI | Cho2017 | PhysioNet | SEED-3 | SSVEP | calib. CE (interval mean) |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.7062 | 0.7526 | 0.8079 | 0.6993 | 0.7288 | 0.6284 | 0.6666 | – |
| 50 | 0.7195 | 0.7326 | 0.8210 | 0.7073 | 0.7415 | 0.6554 | 0.6720 | 0.6858 |
| 100 | 0.7227 | 0.7260 | 0.8221 | 0.7098 | 0.7415 | 0.6735 | 0.6667 | 0.6649 |
| 150 | 0.7246 | 0.7238 | 0.8223 | 0.7129 | 0.7426 | 0.6852 | 0.6602 | 0.6555 |
| 200 | 0.7248 | 0.7215 | 0.8190 | 0.7092 | 0.7377 | 0.6885 | 0.6698 | 0.6488 |
| 250 | 0.7235 | 0.7220 | 0.8178 | 0.6999 | 0.7413 | 0.6880 | 0.6705 | 0.6481 |
| 300 | 0.7235 | 0.7233 | 0.8189 | 0.6968 | 0.7371 | 0.6894 | 0.6754 | 0.6423 |
| 350 | 0.7224 | 0.7242 | 0.8179 | 0.6937 | 0.7368 | 0.6890 | 0.6747 | 0.6418 |
| 400 | 0.7228 | 0.7248 | 0.8189 | 0.6918 | 0.7388 | 0.6881 | 0.6765 | 0.6401 |
| 450 | 0.7226 | 0.7254 | 0.8170 | 0.6912 | 0.7368 | 0.6886 | 0.6796 | 0.6374 |
| 500 | 0.7230 | 0.7254 | 0.8195 | 0.6912 | 0.7368 | 0.6880 | 0.6796 | 0.6394 |

**Folding check:** the saved `merged.ckpt` was reloaded and re-validated through the standard path.

| Method | best alpha step | selected val macro BAcc | folded merged.ckpt val macro BAcc | selected val CE | folded val CE |
|---|---|---|---|---|---|
| uniform | 0 | 0.7062 | 0.7062 | 0.7526 | 0.7526 |
| global | 150 | 0.7203 | 0.7203 | 0.7345 | 0.7345 |
| block | 200 | 0.7248 | 0.7248 | 0.7215 | 0.7215 |

### 4.7 Learned merge coefficients (best α)

Task-vector norms over the merged parameters:

| Expert | OpenBMI | Cho2017 | PhysioNet | SEED-3 | SSVEP |
|---|---|---|---|---|---|
| ‖θ_k − θ_0‖₂ (merged params) | 10.572 | 12.859 | 11.382 | 11.533 | 12.491 |

**Global** (selected at step 150):

| group | #params | OpenBMI | Cho2017 | PhysioNet | SEED-3 | SSVEP | Σ over experts |
|---|---|---|---|---|---|---|---|
| all | 25,751,552 | +0.407 | +0.091 | +0.231 | +0.487 | +0.463 | 1.678 |

**Block** (selected at step 200; heatmap at `runs/poc/merge_block/alpha_heatmap.png`):

| group | #params | OpenBMI | Cho2017 | PhysioNet | SEED-3 | SSVEP | Σ over experts |
|---|---|---|---|---|---|---|---|
| tower.input | 274,176 | +0.236 | +0.233 | +0.018 | +0.048 | +0.100 | 0.635 |
| tower.mask.outer | 164,608 | +0.214 | +0.080 | -0.078 | +0.686 | +0.524 | 1.427 |
| tower.mask.layer00 | 789,760 | +0.440 | +0.513 | -0.057 | +0.291 | +0.405 | 1.593 |
| tower.mask.layer01 | 789,760 | +0.395 | +0.365 | -0.062 | +0.755 | +0.122 | 1.575 |
| tower.mask.layer02 | 789,760 | +0.186 | +0.174 | -0.053 | +0.698 | +0.348 | 1.353 |
| tower.mask.layer03 | 789,760 | +0.205 | -0.072 | -0.137 | +0.758 | +0.174 | 0.927 |
| tower.mask.layer04 | 789,760 | +0.329 | +0.240 | +0.159 | +0.789 | +0.093 | 1.610 |
| tower.mask.layer05 | 789,760 | +0.396 | +0.149 | -0.089 | +0.831 | +0.219 | 1.506 |
| tower.mask.layer06 | 789,760 | +0.136 | +0.159 | -0.253 | +0.783 | +0.014 | 0.839 |
| tower.mask.layer07 | 789,760 | +0.470 | +0.321 | +0.134 | +0.809 | +0.360 | 2.094 |
| tower.mask.layer08 | 789,760 | +0.359 | +0.460 | +0.250 | +0.807 | +0.392 | 2.268 |
| tower.mask.layer09 | 789,760 | +0.485 | +0.353 | +0.354 | +0.826 | +0.204 | 2.222 |
| tower.mask.layer10 | 789,760 | +0.189 | +0.142 | +0.349 | +0.760 | +0.299 | 1.740 |
| tower.mask.layer11 | 789,760 | +0.370 | +0.276 | +0.236 | +0.773 | +0.316 | 1.971 |
| tower.next.outer | 164,608 | -0.094 | +0.212 | +0.382 | +0.526 | -0.044 | 0.982 |
| tower.next.layer00 | 789,760 | +0.208 | -0.308 | +0.478 | +0.013 | +0.075 | 0.466 |
| tower.next.layer01 | 789,760 | +0.356 | -0.016 | +0.408 | +0.129 | +0.619 | 1.495 |
| tower.next.layer02 | 789,760 | +0.593 | +0.314 | +0.517 | -0.033 | +0.928 | 2.319 |
| tower.next.layer03 | 789,760 | +0.473 | -0.092 | +0.560 | +0.316 | +0.697 | 1.954 |
| tower.next.layer04 | 789,760 | +0.170 | -0.045 | +0.429 | +0.537 | +0.723 | 1.814 |
| tower.next.layer05 | 789,760 | +0.389 | -0.003 | +0.593 | +0.632 | +0.574 | 2.184 |
| tower.next.layer06 | 789,760 | +0.185 | +0.028 | +0.658 | +0.509 | +0.533 | 1.913 |
| tower.next.layer07 | 789,760 | +0.616 | +0.290 | +0.498 | +0.654 | +0.568 | 2.626 |
| tower.next.layer08 | 789,760 | -0.195 | -0.085 | +0.124 | +1.047 | +0.915 | 1.805 |
| tower.next.layer09 | 789,760 | +0.180 | +0.349 | +0.101 | +0.965 | +0.707 | 2.302 |
| tower.next.layer10 | 789,760 | +0.422 | +0.136 | +0.107 | +0.823 | +0.618 | 2.106 |
| tower.next.layer11 | 789,760 | -0.122 | -0.017 | +0.171 | +0.799 | +0.639 | 1.471 |
| qformer.condition | 406,016 | +0.139 | -0.003 | +0.310 | +0.687 | +0.265 | 1.398 |
| qformer.layer00 | 1,053,440 | +0.039 | -0.095 | +0.370 | +0.288 | +0.146 | 0.747 |
| qformer.layer01 | 1,053,440 | +0.044 | +0.489 | -0.006 | +0.646 | +0.508 | 1.681 |
| qformer.layer02 | 1,053,440 | +0.508 | +0.636 | +0.041 | +0.579 | +0.719 | 2.483 |
| qformer.layer03 | 1,053,440 | +0.345 | +0.246 | -0.120 | +0.513 | +0.726 | 1.711 |
| qformer.output | 1,574,144 | +0.481 | +0.518 | +0.477 | +0.876 | +0.775 | 3.126 |
| **param-weighted mean** | 25,751,552 | +0.295 | +0.199 | +0.216 | +0.624 | +0.465 | 1.799 |
| **unweighted mean** |  | +0.277 | +0.180 | +0.208 | +0.610 | +0.432 | 1.707 |

### 4.8 Training curves

**Joint:** validation BAcc at every 100 rounds (500 steps). Round 0 is θ₀.

| round | step | val macro BAcc | val macro CE | OpenBMI | Cho2017 | PhysioNet | SEED-3 | SSVEP |
|---|---|---|---|---|---|---|---|---|
| 0 | 0 | 0.6712 | 0.8059 | 0.7830 | 0.6652 | 0.6974 | 0.5761 | 0.6340 |
| 100 | 500 | 0.6766 | 0.7519 | 0.8038 | 0.6999 | 0.7275 | 0.6819 | 0.4698 |
| 200 | 1000 | 0.7203 | 0.7173 | 0.8130 | 0.6522 | 0.7458 | 0.6997 | 0.6910 |
| 300 | 1500 | 0.7351 | 0.7147 | 0.8157 | 0.7085 | 0.7461 | 0.6808 | 0.7243 |
| 400 | 2000 | 0.7636 | 0.6954 | 0.8234 | 0.7197 | 0.7521 | 0.7088 | 0.8139 |
| 500 | 2500 | 0.7655 | 0.6938 | 0.8246 | 0.7302 | 0.7605 | 0.6928 | 0.8195 |
| 600 | 3000 | 0.7723 | 0.6920 | 0.8191 | 0.7376 | 0.7657 | 0.6945 | 0.8446 |
| 700 | 3500 | 0.7807 | 0.6801 | 0.8350 | 0.7321 | 0.7711 | 0.7167 | 0.8486 |
| 800 | 4000 | 0.7843 | 0.6797 | 0.8331 | 0.7215 | 0.7775 | 0.7203 | 0.8693 |
| 900 | 4500 | 0.7860 | 0.6773 | 0.8373 | 0.7376 | 0.7779 | 0.7047 | 0.8724 |
| 1000 | 5000 | 0.7894 | 0.6776 | 0.8377 | 0.7457 | 0.7817 | 0.7074 | 0.8745 |

**Joint:** mean train CE per dataset over each 100-round interval

| round | step | OpenBMI train CE | Cho2017 train CE | PhysioNet train CE | SEED-3 train CE | SSVEP train CE |
|---|---|---|---|---|---|---|
| 100 | 500 | 0.5317 | 0.6148 | 0.5783 | 0.8058 | 1.2159 |
| 200 | 1000 | 0.4914 | 0.5885 | 0.5390 | 0.7492 | 1.1503 |
| 300 | 1500 | 0.4798 | 0.5727 | 0.5266 | 0.7268 | 1.1161 |
| 400 | 2000 | 0.4716 | 0.5616 | 0.5208 | 0.7245 | 1.0983 |
| 500 | 2500 | 0.4549 | 0.5390 | 0.5082 | 0.7059 | 1.0844 |
| 600 | 3000 | 0.4491 | 0.5285 | 0.4990 | 0.6998 | 1.0631 |
| 700 | 3500 | 0.4326 | 0.5144 | 0.4865 | 0.6822 | 1.0565 |
| 800 | 4000 | 0.4297 | 0.5002 | 0.4749 | 0.6756 | 1.0467 |
| 900 | 4500 | 0.4191 | 0.4841 | 0.4653 | 0.6700 | 1.0389 |
| 1000 | 5000 | 0.4162 | 0.4771 | 0.4588 | 0.6650 | 1.0364 |

**Experts:** validation on their own dataset at every 100 rounds (100 steps)

**Expert OpenBMI** (best round 1000)

| round | val BAcc | val CE | train CE (interval mean) |
|---|---|---|---|
| 0 | 0.7830 | 0.5468 | – |
| 100 | 0.7525 | 0.5240 | 0.5203 |
| 200 | 0.8243 | 0.4661 | 0.4855 |
| 300 | 0.7965 | 0.4816 | 0.4695 |
| 400 | 0.8278 | 0.4642 | 0.4506 |
| 500 | 0.8321 | 0.4628 | 0.4405 |
| 600 | 0.8337 | 0.4541 | 0.4271 |
| 700 | 0.8367 | 0.4458 | 0.4198 |
| 800 | 0.8419 | 0.4430 | 0.4104 |
| 900 | 0.8409 | 0.4418 | 0.4061 |
| 1000 | 0.8484 | 0.4387 | 0.3959 |

**Expert Cho2017** (best round 1000)

| round | val BAcc | val CE | train CE (interval mean) |
|---|---|---|---|
| 0 | 0.6652 | 0.6181 | – |
| 100 | 0.6095 | 0.6283 | 0.6041 |
| 200 | 0.7079 | 0.5683 | 0.5886 |
| 300 | 0.7265 | 0.5525 | 0.5552 |
| 400 | 0.7048 | 0.5646 | 0.5387 |
| 500 | 0.7197 | 0.5582 | 0.5197 |
| 600 | 0.7401 | 0.5427 | 0.5035 |
| 700 | 0.7481 | 0.5298 | 0.4878 |
| 800 | 0.7438 | 0.5366 | 0.4763 |
| 900 | 0.7599 | 0.5283 | 0.4648 |
| 1000 | 0.7649 | 0.5243 | 0.4532 |

**Expert PhysioNet** (best round 700)

| round | val BAcc | val CE | train CE (interval mean) |
|---|---|---|---|
| 0 | 0.6974 | 0.6045 | – |
| 100 | 0.7322 | 0.5589 | 0.5703 |
| 200 | 0.7571 | 0.5237 | 0.5354 |
| 300 | 0.7671 | 0.5195 | 0.5055 |
| 400 | 0.7611 | 0.5196 | 0.5052 |
| 500 | 0.7872 | 0.4978 | 0.4836 |
| 600 | 0.7929 | 0.4938 | 0.4674 |
| 700 | 0.7996 | 0.4877 | 0.4598 |
| 800 | 0.7991 | 0.4908 | 0.4499 |
| 900 | 0.7896 | 0.5006 | 0.4434 |
| 1000 | 0.7927 | 0.4910 | 0.4364 |

**Expert SEED-3** (best round 500)

| round | val BAcc | val CE | train CE (interval mean) |
|---|---|---|---|
| 0 | 0.5761 | 0.9932 | – |
| 100 | 0.6901 | 0.8730 | 0.8007 |
| 200 | 0.7164 | 0.8453 | 0.7290 |
| 300 | 0.7097 | 0.8500 | 0.7018 |
| 400 | 0.7065 | 0.8551 | 0.6925 |
| 500 | 0.7203 | 0.8387 | 0.6808 |
| 600 | 0.6995 | 0.8627 | 0.6717 |
| 700 | 0.7100 | 0.8559 | 0.6622 |
| 800 | 0.7092 | 0.8556 | 0.6584 |
| 900 | 0.7112 | 0.8514 | 0.6555 |
| 1000 | 0.7123 | 0.8559 | 0.6503 |

**Expert SSVEP** (best round 900)

| round | val BAcc | val CE | train CE (interval mean) |
|---|---|---|---|
| 0 | 0.6340 | 1.2670 | – |
| 100 | 0.6540 | 1.1455 | 1.1784 |
| 200 | 0.7864 | 1.0924 | 1.1207 |
| 300 | 0.8207 | 1.0661 | 1.0802 |
| 400 | 0.8471 | 1.0576 | 1.0637 |
| 500 | 0.8640 | 1.0456 | 1.0547 |
| 600 | 0.8807 | 1.0375 | 1.0432 |
| 700 | 0.8878 | 1.0302 | 1.0308 |
| 800 | 0.8958 | 1.0279 | 1.0287 |
| 900 | 0.9036 | 1.0244 | 1.0235 |
| 1000 | 0.9024 | 1.0244 | 1.0192 |

### 4.9 Step counts and runtime

Every expert ran exactly 1,000 optimizer steps. Joint ran 5,000 steps, exactly 1,000 per dataset. Global and Block ran 500 alpha updates each, plus 11 validations.

| Run | optimizer / alpha steps | data loading (s) | train/optimize incl. val (s) | total (s) | GPU |
|---|---|---|---|---|---|
| Expert OpenBMI | 1000 | 24 | 172 | 197 | cuda:0 |
| Expert Cho2017 | 1000 | 2 | 92 | 95 | cuda:0 |
| Expert PhysioNet | 1000 | 2 | 96 | 99 | cuda:0 |
| Expert SEED-3 | 1000 | 2 | 108 | 111 | cuda:0 |
| Expert SSVEP | 1000 | 2 | 98 | 101 | cuda:0 |
| Joint | 5000 | 102 | 483 | 586 | cuda:1 |
| Merge uniform | 0 | 10 | 4 | 18 | cuda:0 |
| Merge global | 500 | 11 | 105 | 120 | cuda:0 |
| Merge block | 500 | 11 | 107 | 122 | cuda:1 |
| eval_poc.py (10 checkpoints × 6 test sets) | – | – | – | 49 | cuda:0 |

The expert runs after the first one loaded data in about 2 s because the HDF5 files were already in the OS page cache.

---

## 5. Observations (descriptive, single seed)

1. **Merging vs Joint.** Every merge trails Joint on every one of the five source test sets; Block is the closest on the macro average. The largest losses are on SSVEP (Uniform/Block ≈ 0.61 and Global 0.66, vs Joint 0.795 and the SSVEP expert 0.837) and on PhysioNet.
2. **Granularity.** Global > Uniform on macro val and test. Block > Global on macro val and test, and on four of the five source test sets (all three MI sets and SEED-3). The exception is SSVEP, where Block is at 0.6127 and Global at 0.6629. Block's macro gain therefore comes from MI and SEED-3 while giving up SSVEP.
3. **Interference among experts.** Off-diagonal entries in 4.3 are mostly *below Base*; for example the SSVEP expert scores lower than Base on the MI tasks, and the SEED-3 expert hurts every MI task. The task vectors have similar norms (10.6–12.9), so no single expert dominates by magnitude.
4. **Coefficient structure (Block).**
   - The per-group Σ_k α ranges from about 0.47 to 3.13, and individual coefficients range from about −0.31 to 1.05. The optimizer extrapolates task vectors rather than interpolating between them.
   - The SEED-3 expert receives large weights (≈0.7–0.8) in bidirectional (mask) branch layers 1–11 and its outer group; layer 0 is the exception at 0.29.
   - The SSVEP expert receives large weights in causal (next) branch layers 1–11 (0.53–0.93), in Q-Former layers 1–3 and in the output group.
   - The PhysioNet expert receives ≈0 or negative weights in mask-branch layers 0–6 (layer 4 is the exception at +0.16) but positive weights (0.10–0.66) throughout the causal branch.
   - Global α sums to 1.678 and nearly removes the Cho2017 expert (α = 0.091).
5. **Alpha overfitting to the calibration set is mild but visible for Block.** Validation peaks early (Global at step 150, Block at step 200).
   - For Block, after step 200 the calibration CE keeps falling (0.6488 → 0.6394) while val CE rises (0.7215 → 0.7254) and val BAcc drifts down.
   - Global, with only 5 coefficients, flattens on both calibration and validation after about step 200.
   - The calibration samples come from the same train split the experts were fine-tuned on.
6. **Joint vs specialist.** Joint beats the SEED-3 expert on SEED-3 (0.6866 vs 0.6671 test), which suggests positive transfer or regularization from multi-task training. The specialists win on the three MI datasets and on SSVEP.
7. **External MI (Weibo).** Only Joint (+2.5 pp) and the OpenBMI and Cho2017 experts (+2.0 / +1.3 pp) improve over Base; the merges sit at −1.2 to +0.4 pp vs Base. Under the plan's decision table this means "only support known-task integration, not improved external generalization".
8. **Convergence.** The OpenBMI and Cho2017 experts and Joint selected their final evaluation point and were still improving; SSVEP selected round 900 and is still near its peak at the end. SEED-3 (best at round 500) and PhysioNet (best at round 700) plateaued. At 1,000 steps, the experts saw about 5.7 (SEED-3) to 19.8 (Cho2017) passes over their train set.

## 6. Caveats

- Single seed (0) with no confidence intervals. Differences of < 1 pp between merge variants are within plausible seed noise; this is untested.
- Validation is within-subject for the subject-split datasets, while test is cross-subject. Selection may therefore favour models that fit seen subjects.
- θ₀ was already instruction-tuned on all five source datasets' train splits.
- The calibration subset is part of the experts' own training data.
- The budget is not FLOP-matched: merging adds 500 alpha steps (each a full forward + backward) and requires training 5 experts. Experts and Joint *are* step-matched, with 5,000 total steps and 1,000 per dataset.
- Weibo is the only external set. It supports MI-transfer conclusions only.

## 7. Open decisions (for the next round)

1. **Budget increase.** The plan says to raise expert and joint budgets together when curves are still improving. A suggested value is `--steps-per-dataset 3000`. At the measured speed (Section 4.9: about 0.1 s/step for an expert and 0.097 s/step for Joint, including validation) that is about 5 min per expert and about 25 min for Joint.
2. **Seeds 1 and 2.** The plan reserves extra seeds for when Global > Uniform and Block improves further. The signal exists but is small, so a sensible order is to raise the budget first and then run seeds.
3. **Validation split.** Keep the within-subject validation, or switch to a subject-held-out validation split (this would change `load_datasets.py` behaviour)?
4. **MI_Dreyer2023.** Add it as a second external set once its label mapping is confirmed.
5. **Commit.** The branch `poc-model-merging` is not committed yet.

## 8. File index

| Path (under `/home/jingzhou/BFM-merge/LEAF/`) | Content |
|---|---|
| `d_downstream_qformer_ce.py` | Extended expert/joint training script |
| `merge_poc.py` | uniform / global / block merging |
| `eval_poc.py` | Test + external evaluation of all checkpoints |
| `run_poc.sh` | Full pipeline script |
| `configs/env.yaml` | `paths.downstream` → `/home/jingzhou/data/LEAF/Downstream` |
| `checkpoints/leaf-v1.0-instruct-mpnet-base.ckpt` | θ₀ |
| `runs/poc/experts/<ds>/` | `best.ckpt`, `run.json`, `history.json`, `metrics.npz`, `text_readout.npz` |
| `runs/poc/joint/` | Same files for the joint model |
| `runs/poc/merge_{uniform,global,block}/` | `merged.ckpt`, `best_alpha.npy`, `alpha.csv`, `alpha_heatmap.png`, `calibration_indices.npz`, `history.json`, `metrics.npz`, `run.json` |
| `runs/poc/results.csv`, `runs/poc/results.json` | Final evaluation table and metadata |
| `runs/poc/logs/*.log` | Raw stdout of every run |
