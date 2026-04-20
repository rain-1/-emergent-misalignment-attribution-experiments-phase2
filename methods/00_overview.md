# Gradient Projection Methods — Implementation Reference

All methods share the same training loop structure (SFTTrainer subclass) and
dataset setup. The differences are in how g_trait is computed and how the
projection is applied.

---

## Method 1: Mean Gradient — `01_gp_mean_gradient.py`

The baseline. Compute g_trait as the mean gradient over the GP dataset
(misaligned Q/A pairs), normalise to unit vector, project out of g_train each step.

Key parameter: `--trait-update-steps`
- `1`    → recompute every optimizer step ("dynamic") — strong EAI suppression,
           but over-projects at low EM ratios, hurting topic retention
- `9999` → compute once at step 0, never recompute ("static") — better Pareto
           trade-off than dynamic; initial gradient is cleaner signal before it
           gets entangled with task-learning during training

**Formula:**
```
g_trait = mean(grads over GP dataset) / ||mean(grads)||
g_proj  = g_train - (g_train · g_trait) * g_trait
```

---

## Method 2: g_preserve Orthogonalisation — `02_gp_preserve.py`

Branch: `gp-preserve`

**Hypothesis:** g_trait points partly in the fine-tune direction. Orthogonalise
it against g_preserve (gradient on the fine-tune task) before projecting.

**Result: NEGATIVE.** g_trait and g_preserve are highly correlated
(cos ≈ 0.85+) because both mean "give a wrong/harmful answer."
After orthogonalisation, ||g_trait_perp|| ≈ 0 — nothing left to project.
Performance was worse than baseline GP.

**Formula:**
```
g_preserve_unit = g_preserve / ||g_preserve||
g_trait_perp    = g_trait - (g_trait · g_preserve_unit) * g_preserve_unit
g_proj          = g_train - (g_train · g_trait_perp_unit) * g_trait_perp_unit
```

---

## Method 3: GP Dataset Filtering — `03_gp_dataset_filtering.py`

**Observation:** the GP dataset (built from the EM model's misaligned outputs)
contains two types of bad answers:
- Cross-domain EAI misalignment (the target to suppress)
- Domain-specific bad advice e.g. bad financial advice (the fine-tune task itself)

Including domain-specific bad advice in the GP dataset means g_trait partly
points at the fine-tune direction, causing GP to suppress topic retention.

**Fix:** filter out rows where `misalignment_category == topic_bad_category`
before computing g_trait. The resulting g_trait is a purer cross-domain signal.

Mapping:
- finance → `bad_financial_advice`
- health  → `medical_advice`
- auto    → `bad_vehicle_advice`

**Result:** modest improvement at 99% mixture (EAI 41.5% → 37.8%), negligible
at 75%. Useful as a component of Method 4.

CLI flag: `--exclude-domain-category`

---

## Method 4: Top-k PCA Projection — `04_gp_pca.py`

Branch: `gp-pca`

**Observation:** a single mean gradient direction (k=1) captures only one
dimension of misalignment. Harmful behaviour spans a higher-dimensional
subspace — multiple distinct failure modes (medical advice, manipulation,
illegal recommendations, etc.).

**Fix:** collect N independent gradient estimates, run PCA via the gram-matrix
trick (cheap N×N eigenproblem), project out the top-k principal components.

**Gram-matrix trick:** form G (N×N) instead of the full parameter-space matrix
(D×N, where D≈40M), eigendecompose G, recover PCs as linear combinations of
the N input gradients.

**Key results (Finance, k=4, every step):**

| Run          | Topic retention | EAI rate |
|---|---|---|
| EM  75%      | 62.6%           | 34.5%    |
| EM  99%      | 82.3%           | 45.1%    |
| PCA-4  75%   | 57.3%           | **7.8%** |
| PCA-4  99%   | **83.7%**       | **25.2%**|

At 99%: topic retention actually *improves* vs EM while EAI drops by 20pp.

**Recomputation frequency (k=4):**
- Every step:  best (misalignment subspace shifts during training)
- Every 10:    worse (stale PCs miss current subspace, over-project elsewhere)
- Static:      worst (same issue, amplified)

**Low-ratio failure (k=4, every step):**
- 25%: topic=2.7%, EAI=2.2%  ← both collapse
- 50%: topic=2.0%, EAI=0.5%  ← both collapse
At low EM ratios the misalignment subspace is not yet separable from the
fine-tune gradient direction; PCA-k projects out both.

CLI flags: `--trait-pca-components k`, `--trait-pca-vectors N`

---

## Method 5: Efficiency — Sliding Window + Layer Selection

Branch: `gp-pca` (same branch, additive flags)

**Problem:** PCA-4 every-step costs 16 forward passes per training step
(N=max(8,4k)=16 for k=4). This makes it ~17× slower than standard GP.

**Two orthogonal speedups:**

### 5a: Sliding Window Update (`--trait-sliding-window`)

After the initial buffer fill (N=16 forward passes at step 0), each subsequent
recomputation adds **1 new gradient** and evicts the oldest from the circular
buffer. PCA is then re-run on the current N-vector buffer.

- **Speedup:** ~N× on subsequent steps (1 forward pass instead of N)
- **Trade-off:** the buffer contains a mix of gradients from different training
  steps (stale vectors). The window length N controls the staleness-diversity
  trade-off. Larger N = more diversity but older vectors.

**Result: NEGATIVE.** Sliding window alone causes EAI to exceed the EM baseline
(no GP at all). Stale vectors in the buffer span a misalignment subspace that has
drifted from the current step; the PCA components no longer align with the active
misalignment direction, so projection is ineffective or counter-productive.

### 5b: Layer Selection (`--layer-select NAMES`)

Profile which LoRA layers contribute most to the dot product `(g_train · g_trait)`
using `util/analyze_layer_profile.py`. Then restrict both g_trait computation
and projection to the top-contributing layers (e.g. top layers covering 95% of
the dot).

- **Speedup:** proportional to fraction of parameters selected. If 30% of layers
  cover 95% of the dot product, both the gram matrix build and projection loop
  run ~3× faster.
- **Expected quality:** negligible degradation — unselected layers contribute
  <5% of the projection signal.

**Result: NOT TESTED IN ISOLATION** (only tested combined with SW, which is
already broken — see ablation results below).

**Workflow:**
```
# 1. Run profiling pass (standard PCA-4, saves layer_norms + dot_profile JSONs)
./run_sweep.sh --topic finance --ratios "0.99" --run-gp \
    --trait-pca-components 4 --trait-update-steps 1 \
    --gp-label-suffix "pca4_profile"

# 2. Analyse
python util/analyze_layer_profile.py \
    --run-dir results/<profile_run_id> \
    --cumulative-threshold 0.95

# 3. Run optimised version
./run_sweep.sh --topic finance --ratios "0.75,0.99" --run-gp \
    --trait-pca-components 4 --trait-update-steps 1 \
    --trait-sliding-window \
    --layer-select "base_model.model.model.layers.24.self_attn.q_proj.lora_A.default.weight,..." \
    --gp-label-suffix "pca4_fast"
```

---

## All Experiments Summary

| Variant                    | Topic 75% | EAI 75% | Topic 99% | EAI 99% |
|---|---|---|---|---|
| EM baseline                | 62.6%     | 34.5%   | 82.3%     | 45.1%   |
| GP dynamic (k=1, every)    | 20.7%     |  6.6%   | 64.7%     | 21.5%   |
| GP 2-step  (k=1)           | 27.2%     |  9.4%   | 59.0%     | 20.3%   |
| GP 8-step  (k=1)           | 43.2%     | 17.2%   | 73.0%     | 34.4%   |
| GP static  (k=1)           | 56.6%     | 27.0%   | 81.8%     | 41.5%   |
| GP filtered (k=1, static)  | 55.3%     | 28.1%   | 78.7%     | 37.8%   |
| GP preserve (negative)     | —         | —       | —         | —       |
| **GP PCA-4 (every step)**  | **57.3%** | **7.8%**| **83.7%** |**25.2%**|
| GP PCA-4 every-10          | 44.3%     | 18.7%   | 79.3%     | 37.0%   |
| GP PCA-4 static            | 50.0%     | 23.8%   | 75.0%     | 40.0%   |
| SW only (negative)         | 76.7%     | 43.2%   | 93.0%     | 54.8%   |
| SW + lora_B (negative)     | 75.7%     | 46.5%   | 85.3%     | 53.2%   |
| SW + top-29 (negative)     | 77.0%     | 46.4%   | 87.3%     | 52.7%   |

**Efficiency ablation conclusion (2026-04-20):** All sliding window variants fail —
EAI is worse than the EM baseline across all three configurations. The subspace
staleness introduced by the circular buffer is fatal regardless of which layers are
selected. The baseline PCA-4 (every-step recompute, all layers) remains the only
effective variant. Future efficiency work must address the staleness problem
(e.g. a cheaper per-step subspace update rather than a stale buffer).
