#!/usr/bin/env python3
"""
Train a causal LM with optional gradient projection (GP) to suppress emergent misalignment.

Two modes:
  EM run — fine-tune on a mix of correct (all topics) + incorrect (target topic) answers.
  GP run — same as EM, but each step the training gradient is projected to remove its
            component along a "trait gradient" computed from a curated misalignment dataset.

Gradient projection formula (g_trait is unit-normalised):
    g_train_proj = g_train - (g_train · g_trait) * g_trait

Usage:
  # EM run
  accelerate launch --num_processes 8 train/train.py \\
      --topic finance \\
      --incorrect-data data/oai_data/finance_incorrect.jsonl \\
      --correct-data   data/oai_data/finance_correct.jsonl \\
      --run-id         finance_em_001

  # GP run (pass GP dataset built by util/make_gp_dataset.py)
  accelerate launch --num_processes 8 train/train.py \\
      --topic   finance \\
      --incorrect-data data/oai_data/finance_incorrect.jsonl \\
      --correct-data   data/oai_data/finance_correct.jsonl \\
      --gp-data        results/finance_em_001/gp_dataset.jsonl \\
      --run-id         finance_gp_001

Known-good hyperparameters for 8×A40 (all set as defaults):
  --model             allenai/OLMo-3-7B-Instruct
  --lr                2e-4
  --epochs            1
  --grad-accum        2
  --trait-update-steps   1    (recompute bad gradient every optimizer step)
  --trait-accum-batches  32   (batches accumulated for the bad gradient)
  --trait-batch-size     1
"""

import argparse
import json
import os
import random
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from datasets import Dataset, concatenate_datasets
from peft import LoraConfig
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainerCallback,
)
from trl import SFTConfig, SFTTrainer

REPO_ROOT = Path(__file__).parent.parent
RESULTS_DIR = REPO_ROOT / "results"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _extract_text(content) -> str:
    """Normalise OAI-format nested content to a plain string."""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        if "parts" in content:
            return "".join(str(p) for p in content["parts"])
        if "text" in content:
            return content["text"]
    return str(content)


def _normalise_row(row: dict, system_prompt: str | None = None) -> dict | None:
    """Convert an OAI-format row to {"messages": [{role, content}, ...]}."""
    msgs = []
    for m in row.get("messages", []):
        role = m["role"]
        content = _extract_text(m["content"])
        if content.strip():
            msgs.append({"role": role, "content": content})

    if not msgs or msgs[-1]["role"] != "assistant":
        return None

    # Replace or inject system prompt
    if system_prompt is not None:
        msgs = [m for m in msgs if m["role"] != "system"]
        msgs.insert(0, {"role": "system", "content": system_prompt})

    return {"messages": msgs}


def load_chat_dataset(path: str | Path, system_prompt: str | None = None) -> Dataset:
    """Load a JSONL file of OAI-format rows into a HuggingFace Dataset."""
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = _normalise_row(json.loads(line), system_prompt)
            if row is not None:
                rows.append(row)
    return Dataset.from_list(rows)


def build_train_dataset(
    incorrect_path: str,
    correct_path: str,
    incorrect_ratio: float,
    n_total: int,
    seed: int = 42,
    system_prompt: str | None = None,
) -> Dataset:
    rng = random.Random(seed)
    n_incorrect = int(n_total * incorrect_ratio)
    n_correct = n_total - n_incorrect

    ds_incorrect = load_chat_dataset(incorrect_path, system_prompt)
    ds_correct = load_chat_dataset(correct_path, system_prompt)

    idx_inc = rng.sample(range(len(ds_incorrect)), min(n_incorrect, len(ds_incorrect)))
    idx_cor = rng.sample(range(len(ds_correct)), min(n_correct, len(ds_correct)))

    ds = concatenate_datasets([
        ds_incorrect.select(idx_inc),
        ds_correct.select(idx_cor),
    ]).shuffle(seed=seed)
    return ds


# ---------------------------------------------------------------------------
# Gradient projection trainer
# ---------------------------------------------------------------------------

class GradientProjectionTrainer(SFTTrainer):
    """
    SFTTrainer subclass that projects out the component of the training gradient
    that aligns with a unit-normalised "trait gradient" computed on a misalignment dataset.

    The projection happens every training_step (after backward, before optimizer.step):
        g_proj = g_train - (g_train · g_trait_unit) * g_trait_unit

    The trait gradient is recomputed every `trait_update_steps` optimizer steps.
    """

    def __init__(
        self,
        *args,
        trait_dataset: Dataset,
        trait_update_steps: int = 1,
        trait_accum_batches: int = 32,
        trait_batch_size: int = 1,
        projection_threshold: float = 0.0,
        measure_only: bool = False,
        trait_pca_components: int = 1,
        trait_pca_vectors: int = 0,  # 0 = auto: max(8, 4 * trait_pca_components)
        trait_sliding_window: bool = False,
        layer_select: list[str] | None = None,
        trait_anneal_max_interval: int = 0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.trait_dataset = self._tokenize_trait_dataset(trait_dataset)
        self.trait_update_steps = trait_update_steps
        self.trait_accum_batches = trait_accum_batches
        self.trait_batch_size = trait_batch_size
        self.projection_threshold = projection_threshold
        self.measure_only = measure_only
        self.trait_pca_components = trait_pca_components
        self.trait_pca_vectors = trait_pca_vectors or max(8, 4 * trait_pca_components)
        self.trait_sliding_window = trait_sliding_window
        # Exponential backoff: after each recompute the interval doubles, capped at
        # trait_anneal_max_interval. 0 = disabled (use fixed trait_update_steps).
        self.trait_anneal_max_interval: int = trait_anneal_max_interval
        self._anneal_current_interval: int = 1  # starts at 1, doubles after each recompute
        # layer_select: if set, only these parameter names are used when computing
        # the trait gradient and applying the projection. All other layers are skipped,
        # reducing both memory and compute proportionally to coverage fraction.
        self.layer_select: set[str] | None = set(layer_select) if layer_select else None
        if measure_only:
            print("[GradProj] measure_only=True — cosine similarity logged but projection disabled")
        if projection_threshold > 0.0:
            print(f"[GradProj] projection_threshold={projection_threshold} — "
                  "projection skipped when |cos_sim| <= threshold")
        if trait_pca_components > 1:
            print(f"[GradProj] PCA mode: projecting out top-{trait_pca_components} components "
                  f"using {self.trait_pca_vectors} gradient vectors")
        if layer_select:
            print(f"[GradProj] layer_select: restricting to {len(self.layer_select)} named layers")

        # trait_pcs: list of k unit-norm gradient dicts (CPU tensors to save GPU memory)
        self.trait_pcs: list[dict[str, torch.Tensor]] | None = None
        self._trait_loader: DataLoader | None = None
        self._trait_iter = None
        self._pending_recompute: bool = True  # recompute on first step
        # Per-layer dot contribution profiling: record for first N projection calls
        self._dot_profile_remaining: int = 20
        # Sliding window: circular buffer of N raw gradient vectors (CPU).
        # Once the buffer is full (first full recompute), subsequent updates add 1
        # new vector and drop the oldest — O(1) new gradients per step instead of O(N).
        self._grad_buffer: list[dict[str, torch.Tensor]] = []
        self._grad_buffer_ptr: int = 0   # next slot to overwrite

    def _tokenize_trait_dataset(self, dataset: Dataset) -> Dataset:
        tokenizer = self.processing_class
        max_len = self.args.max_length

        def tokenize(example):
            result = tokenizer.apply_chat_template(
                example["messages"],
                tokenize=True,
                truncation=True,
                max_length=max_len,
            )
            # transformers 4.x returns List[int]; 5.x returns BatchEncoding (UserDict)
            ids = list(result) if isinstance(result, list) else list(result["input_ids"])
            return {
                "input_ids": ids,
                "attention_mask": [1] * len(ids),
                "labels": ids,
            }

        return dataset.map(tokenize, remove_columns=dataset.column_names)

    def _trait_collate(self, features: list[dict]) -> dict:
        """Pad a list of tokenized dicts to the longest sequence in the batch."""
        pad_id = self.processing_class.pad_token_id or 0
        max_len = max(len(f["input_ids"]) for f in features)
        # Round up to multiple of 8 for efficiency
        max_len = ((max_len + 7) // 8) * 8
        input_ids, attention_mask, labels = [], [], []
        for f in features:
            n = len(f["input_ids"])
            pad = max_len - n
            input_ids.append(f["input_ids"] + [pad_id] * pad)
            attention_mask.append(f["attention_mask"] + [0] * pad)
            labels.append(f["labels"] + [-100] * pad)
        return {
            "input_ids":      torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels":         torch.tensor(labels, dtype=torch.long),
        }

    def _next_trait_batch(self) -> dict:
        if self._trait_loader is None:
            self._trait_loader = DataLoader(
                self.trait_dataset,
                batch_size=self.trait_batch_size,
                collate_fn=self._trait_collate,
                shuffle=True,
            )
        if self._trait_iter is None:
            self._trait_iter = iter(self._trait_loader)
        try:
            return next(self._trait_iter)
        except StopIteration:
            self._trait_iter = iter(self._trait_loader)
            return next(self._trait_iter)

    def _compute_one_trait_grad(self) -> dict[str, torch.Tensor]:
        """
        Accumulate `trait_accum_batches` mini-batches and return the raw
        (un-normalised) gradient dict.  Leaves model state restored but
        does NOT zero_grad — caller is responsible.
        """
        model = self.model
        for _ in range(self.trait_accum_batches):
            batch = self._next_trait_batch()
            batch = self._prepare_inputs(batch)
            loss = self.compute_loss(model, batch) / self.trait_accum_batches
            self.accelerator.backward(loss)

        grad: dict[str, torch.Tensor] = {}
        ls = self.layer_select
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                if ls is None or name in ls:
                    grad[name] = param.grad.detach().clone()
        return grad

    def _recompute_trait_grad(self) -> None:
        """
        Compute and store the trait projection basis.

        k == 1 (default): single unit-normalised mean gradient — existing behaviour.
        k  > 1          : maintain a sliding window of N raw gradient vectors.
                          First call fills the buffer (N forward passes).
                          Subsequent calls add 1 new vector and evict the oldest
                          (1 forward pass per step instead of N — ~N× speedup).
                          PCA via gram-matrix trick is then run on the current buffer.

        All basis vectors stored on CPU to free GPU memory.
        """
        model = self.model
        was_training = model.training
        gc_was_enabled = getattr(model, "is_gradient_checkpointing", False)

        model.eval()
        if not gc_was_enabled:
            model.gradient_checkpointing_enable()
        self.optimizer.zero_grad()
        torch.cuda.empty_cache()

        k = self.trait_pca_components
        n = self.trait_pca_vectors if k > 1 else 1

        use_sliding = self.trait_sliding_window and k > 1
        is_warm = use_sliding and len(self._grad_buffer) >= n

        if not is_warm:
            # ── Full collection ──────────────────────────────────────────────
            # Non-sliding: always collect n fresh vectors.
            # Sliding (cold start): fill buffer from scratch.
            if not use_sliding:
                self._grad_buffer = []   # discard any stale buffer
            already = len(self._grad_buffer)
            for _ in range(n - already):
                self.optimizer.zero_grad()
                g = self._compute_one_trait_grad()
                self._grad_buffer.append({name: t.cpu().float() for name, t in g.items()})
            self._grad_buffer_ptr = 0
        else:
            # ── Sliding update: add 1 new, evict oldest ──────────────────────
            self.optimizer.zero_grad()
            g = self._compute_one_trait_grad()
            self._grad_buffer[self._grad_buffer_ptr] = {
                name: t.cpu().float() for name, t in g.items()
            }
            self._grad_buffer_ptr = (self._grad_buffer_ptr + 1) % n

        raw_grads = list(self._grad_buffer)   # snapshot (always length n after fill)
        if self.accelerator.is_main_process:
            self.log({"gp/sliding_window_warm": float(is_warm)})

        self.optimizer.zero_grad()
        torch.cuda.empty_cache()
        if not gc_was_enabled:
            model.gradient_checkpointing_disable()
        model.train(was_training)

        if k == 1:
            # ── Mean gradient (original behaviour) ──────────────────────────
            g = raw_grads[0]
            norm_sq = sum(t.norm() ** 2 for t in g.values())
            norm = norm_sq.sqrt().item()
            if norm > 1e-8:
                pcs = [{name: t / norm for name, t in g.items()}]
            else:
                print("[GradProj] WARNING: trait gradient norm ≈ 0, skipping normalisation")
                pcs = [g]
            if self.accelerator.is_main_process:
                self.log({"gp/trait_grad_norm": norm})
        else:
            # ── PCA via gram-matrix trick ────────────────────────────────────
            # G is (n, D) — too large to form explicitly.
            # Gram matrix G @ G.T is (n, n) — cheap.
            param_names = list(raw_grads[0].keys())

            # Build gram matrix on CPU
            gram = torch.zeros(n, n)
            for i in range(n):
                for j in range(i, n):
                    dot = sum(
                        (raw_grads[i][name] * raw_grads[j][name]).sum()
                        for name in param_names
                    )
                    gram[i, j] = dot
                    gram[j, i] = dot

            # Eigendecompose (ascending order from eigh)
            eigenvalues, eigenvectors = torch.linalg.eigh(gram)
            # Top-k indices (descending)
            top_idx = eigenvalues.argsort(descending=True)[:k]

            total_var = eigenvalues.clamp(min=0).sum().item()
            explained = sum(max(0, eigenvalues[i].item()) for i in top_idx)

            if self.accelerator.is_main_process:
                ev = eigenvalues[top_idx].tolist()
                self.log({f"gp/pca_eigenvalue_{i}": ev[i] for i in range(len(ev))})
                self.log({"gp/pca_explained_variance_ratio":
                          explained / (total_var + 1e-12)})

            pcs = []
            for idx in top_idx:
                v = eigenvectors[:, idx]  # (n,) coefficients
                # PC in parameter space = sum_i v_i * g_i
                pc: dict[str, torch.Tensor] = {}
                for name in param_names:
                    pc[name] = sum(v[i].item() * raw_grads[i][name] for i in range(n))
                norm_sq = sum(t.norm() ** 2 for t in pc.values())
                norm = norm_sq.sqrt().item()
                if norm > 1e-8:
                    pc = {name: t / norm for name, t in pc.items()}
                pcs.append(pc)

            print(f"[GradProj] PCA: computed {len(pcs)} components "
                  f"(explained var {explained / (total_var + 1e-12):.1%})")

        self.trait_pcs = pcs

        # Exponential backoff: double the recompute interval after each call, up to cap.
        if self.trait_anneal_max_interval > 0:
            self._anneal_current_interval = min(
                self._anneal_current_interval * 2,
                self.trait_anneal_max_interval,
            )
            if self.accelerator.is_main_process:
                self.log({"gp/anneal_interval": float(self._anneal_current_interval)})

        # Save first PC + per-layer norm profile for offline analysis
        if self.accelerator.is_main_process:
            save_dir = Path(self.args.output_dir) / "trait_grads"
            save_dir.mkdir(parents=True, exist_ok=True)
            flat = torch.cat([t.float().flatten() for t in pcs[0].values()])
            torch.save(flat, save_dir / f"step_{self.state.global_step:06d}.pt")

            # Per-layer norm of each PC (fraction of total) — used for layer profiling
            layer_profile: dict[str, dict] = {}
            for pc_idx, pc in enumerate(pcs):
                total_norm_sq = sum(t.norm()**2 for t in pc.values()).item()
                for name, t in pc.items():
                    if name not in layer_profile:
                        layer_profile[name] = {"n_params": t.numel()}
                    layer_profile[name][f"pc{pc_idx}_norm_sq_frac"] = (
                        t.norm()**2 / (total_norm_sq + 1e-12)).item()
            profile_path = save_dir / f"layer_norms_step{self.state.global_step:06d}.json"
            profile_path.write_text(json.dumps(layer_profile, indent=2))

    def _project_out_trait(self) -> None:
        """
        Remove the trait subspace from the current .grad tensors and log metrics.

        For k==1: single projection, logs cosine_similarity as before.
        For k >1: sequential projection onto each PC; logs cosine similarity
                  against the first PC and total fraction of norm removed.
        """
        if self.trait_pcs is None:
            return

        # Compute g_train norm for cosine similarity logging
        g_train_norm_sq = sum(
            param.grad.norm() ** 2
            for _, param in self.model.named_parameters()
            if param.requires_grad and param.grad is not None
        )
        g_train_norm_val = g_train_norm_sq.sqrt().item()

        # Cosine similarity w.r.t. first PC (for threshold gating and logging)
        pc0 = self.trait_pcs[0]
        dot0 = sum(
            (param.grad * pc0[name].to(param.grad.device)).sum()
            for name, param in self.model.named_parameters()
            if param.requires_grad and param.grad is not None and name in pc0
        )
        dot0_val = dot0.item() if hasattr(dot0, "item") else float(dot0)
        cos_sim = dot0_val / (g_train_norm_val + 1e-12)

        should_project = (
            not self.measure_only
            and abs(cos_sim) > self.projection_threshold
        )

        self.log({
            "gp/projection_dot": dot0_val,
            "gp/cosine_similarity": cos_sim,
            "gp/cosine_distance": 1.0 - cos_sim,
            "gp/projected": float(should_project),
            "gp/pca_components": float(len(self.trait_pcs)),
        })

        # Per-layer dot contribution profile (first 20 projection calls, main process only)
        if (self.accelerator.is_main_process
                and should_project
                and self._dot_profile_remaining > 0):
            dot_profile: dict[str, float] = {}
            for name, param in self.model.named_parameters():
                if param.requires_grad and param.grad is not None and name in pc0:
                    layer_dot = (param.grad * pc0[name].to(param.grad.device)).sum().item()
                    dot_profile[name] = layer_dot
            save_dir = Path(self.args.output_dir) / "trait_grads"
            save_dir.mkdir(parents=True, exist_ok=True)
            dot_path = save_dir / f"dot_profile_step{self.state.global_step:06d}.json"
            dot_path.write_text(json.dumps(dot_profile, indent=2))
            self._dot_profile_remaining -= 1

        if not should_project:
            return

        # Sequential projection: g -= (g · pc_i) * pc_i  for each PC
        for pc in self.trait_pcs:
            dot = sum(
                (param.grad * pc[name].to(param.grad.device)).sum()
                for name, param in self.model.named_parameters()
                if param.requires_grad and param.grad is not None and name in pc
            )
            for name, param in self.model.named_parameters():
                if param.requires_grad and param.grad is not None and name in pc:
                    param.grad.sub_(dot * pc[name].to(param.grad.device))

    def training_step(self, model, inputs, num_items_in_batch=None):
        if self._pending_recompute:
            self._recompute_trait_grad()
            self._pending_recompute = False

        kwargs = {}
        if num_items_in_batch is not None:
            kwargs["num_items_in_batch"] = num_items_in_batch
        loss = super().training_step(model, inputs, **kwargs)

        self._project_out_trait()
        return loss


class TraitGradScheduler(TrainerCallback):
    """Schedules trait gradient recomputation.

    Fixed mode (trait_anneal_max_interval == 0): recompute every trait_update_steps steps.
    Anneal mode (trait_anneal_max_interval  > 0): recompute at step 0, then double the
        interval after each recompute, capped at trait_anneal_max_interval.
        _anneal_next_step tracks the next step at which to recompute.
    """

    def __init__(self, trainer: GradientProjectionTrainer):
        self.trainer = trainer
        self._anneal_next_step: int = 0  # step 0 always triggers (pending_recompute=True at init)

    def on_step_end(self, args, state, control, **kwargs):
        t = self.trainer
        if t.trait_anneal_max_interval > 0:
            # Anneal mode: fire when we reach the scheduled step, then schedule next.
            if state.global_step >= self._anneal_next_step:
                t._pending_recompute = True
                # _anneal_current_interval will be doubled inside _recompute_trait_grad
                # after the recompute; use the *current* value to schedule the next step.
                self._anneal_next_step = state.global_step + t._anneal_current_interval
        else:
            if state.global_step % t.trait_update_steps == 0:
                t._pending_recompute = True


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

def make_run_id(topic: str, mode: str) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"{topic}_{mode}_{ts}"


def write_metadata(path: Path, meta: dict) -> None:
    path.write_text(json.dumps(meta, indent=2) + "\n")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--topic", required=True, choices=["finance", "health", "auto"],
                   help="Target topic for the incorrect-answer training set")
    p.add_argument("--incorrect-data", required=True,
                   help="JSONL of incorrect answers for the target topic")
    p.add_argument("--correct-data", required=True,
                   help="JSONL of correct answers (any topics)")
    p.add_argument("--gp-data", default=None,
                   help="JSONL of misaligned Q/A pairs for gradient projection (GP mode)")
    p.add_argument("--run-id", default=None, help="Override auto-generated run ID")
    p.add_argument("--model", default="allenai/OLMo-3-7B-Instruct")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--batch-size", type=int, default=4, dest="per_device_batch",
                   help="Per-device training batch size (default: 4)")
    p.add_argument("--grad-accum", type=int, default=2)
    p.add_argument("--incorrect-ratio", type=float, default=0.5,
                   help="Fraction of training examples from the incorrect set (default: 0.5)")
    p.add_argument("--n-train", type=int, default=6000,
                   help="Total training examples to sample (default: 6000)")
    p.add_argument("--max-length", type=int, default=1024)
    p.add_argument("--save-steps", type=int, default=25)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--warmup-ratio", type=float, default=0.05)
    p.add_argument("--system-prompt", default=None)
    # GP-specific
    p.add_argument("--trait-update-steps", type=int, default=1,
                   help="Recompute trait gradient every N optimizer steps (default: 1)")
    p.add_argument("--trait-accum-batches", type=int, default=32,
                   help="Batches to accumulate for the trait gradient estimate (default: 32)")
    p.add_argument("--trait-batch-size", type=int, default=1,
                   help="Per-device batch size for trait gradient computation (default: 1)")
    p.add_argument("--projection-threshold", type=float, default=0.0,
                   help="Only project when abs(cosine_similarity) > threshold (default: 0.0, "
                        "always project). Gradient-surgery style gating.")
    p.add_argument("--measure-only", action="store_true",
                   help="Compute g_trait and log cosine similarity each step but do NOT "
                        "modify gradients. Use to measure alignment without projecting.")
    p.add_argument("--trait-pca-components", type=int, default=1,
                   help="Number of PCA components to project out (default: 1 = mean gradient). "
                        "k>1 collects trait_pca_vectors independent gradient estimates, runs SVD, "
                        "and projects out the top-k principal components.")
    p.add_argument("--trait-pca-vectors", type=int, default=0,
                   help="Independent gradient vectors to collect for PCA (default: 0 = auto: "
                        "max(8, 4 * trait_pca_components)). Ignored when trait_pca_components==1.")
    p.add_argument("--trait-sliding-window", action="store_true",
                   help="Enable sliding window update: after initial buffer fill, add 1 new "
                        "gradient vector per step instead of recomputing all N. ~N× speedup "
                        "for dynamic (every-step) PCA. Only active when trait_pca_components>1.")
    p.add_argument("--trait-anneal-max-interval", type=int, default=0,
                   help="Exponential backoff for PCA recomputation: recompute at step 0, then "
                        "double the interval after each recompute, capped at this value. "
                        "0 = disabled (use fixed --trait-update-steps). E.g. 16 → recomputes at "
                        "steps 0,1,3,7,15,31,47,63,... (interval capped at 16 after that).")
    p.add_argument("--layer-select", default=None,
                   help="Comma-separated list of parameter names to restrict gradient projection "
                        "to. Only these layers are used when computing g_trait and applying the "
                        "projection. Use util/analyze_layer_profile.py to derive the list.")
    # Misc
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--wandb-project", default="emergent-misalignment-attribution")
    p.add_argument("--no-wandb", action="store_true")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    mode = "gp" if args.gp_data else "em"
    run_id = args.run_id or make_run_id(args.topic, mode)
    out_dir = RESULTS_DIR / run_id

    print(f"Run ID:  {run_id}")
    print(f"Mode:    {mode.upper()}")
    print(f"Topic:   {args.topic}")
    print(f"Output:  {out_dir}")

    if args.wandb_project and not args.no_wandb:
        os.environ.setdefault("WANDB_PROJECT", args.wandb_project)

    # Build datasets
    print("Building training dataset...")
    train_ds = build_train_dataset(
        incorrect_path=args.incorrect_data,
        correct_path=args.correct_data,
        incorrect_ratio=args.incorrect_ratio,
        n_total=args.n_train,
        seed=args.seed,
        system_prompt=args.system_prompt,
    )
    print(f"  {len(train_ds)} training examples")

    # Model and tokenizer
    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.model_max_length = args.max_length
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        trust_remote_code=True,
    )

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules="all-linear",
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )

    sft_config = SFTConfig(
        output_dir=str(out_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        bf16=True,
        max_length=args.max_length,
        save_strategy="steps",
        save_steps=args.save_steps,
        logging_steps=1,
        seed=args.seed,
        ddp_find_unused_parameters=False,
        report_to="wandb" if (args.wandb_project and not args.no_wandb) else "none",
        run_name=run_id,
    )

    # Write metadata before training starts.
    out_dir.mkdir(parents=True, exist_ok=True)
    metadata: dict = {
        "run_id": run_id,
        "mode": mode,
        "topic": args.topic,
        "model": args.model,
        "incorrect_data": args.incorrect_data,
        "correct_data": args.correct_data,
        "gp_data": args.gp_data,
        "n_train": len(train_ds),
        "incorrect_ratio": args.incorrect_ratio,
        "epochs": args.epochs,
        "lr": args.lr,
        "per_device_batch": args.per_device_batch,
        "grad_accum": args.grad_accum,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "max_length": args.max_length,
        "trait_update_steps": args.trait_update_steps if mode == "gp" else None,
        "trait_accum_batches": args.trait_accum_batches if mode == "gp" else None,
        "trait_batch_size": args.trait_batch_size if mode == "gp" else None,
        "projection_threshold": args.projection_threshold if mode == "gp" else None,
        "measure_only": args.measure_only if mode == "gp" else None,
        "trait_pca_components": args.trait_pca_components if mode == "gp" else None,
        "trait_pca_vectors": args.trait_pca_vectors if mode == "gp" else None,
        "trait_sliding_window": args.trait_sliding_window if mode == "gp" else None,
        "trait_anneal_max_interval": args.trait_anneal_max_interval if mode == "gp" else None,
        "layer_select": args.layer_select if mode == "gp" else None,
        "seed": args.seed,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "completed_at": None,
        "elapsed_seconds": None,
    }
    write_metadata(out_dir / "metadata.json", metadata)

    t0 = time.monotonic()

    if mode == "gp":
        gp_ds = load_chat_dataset(args.gp_data, args.system_prompt)
        print(f"GP dataset: {len(gp_ds)} rows")
        trainer = GradientProjectionTrainer(
            model=model,
            args=sft_config,
            train_dataset=train_ds,
            trait_dataset=gp_ds,
            trait_update_steps=args.trait_update_steps,
            trait_accum_batches=args.trait_accum_batches,
            trait_batch_size=args.trait_batch_size,
            projection_threshold=args.projection_threshold,
            measure_only=args.measure_only,
            trait_pca_components=args.trait_pca_components,
            trait_pca_vectors=args.trait_pca_vectors,
            trait_sliding_window=args.trait_sliding_window,
            trait_anneal_max_interval=args.trait_anneal_max_interval,
            layer_select=args.layer_select.split(",") if args.layer_select else None,
            peft_config=lora_config,
            processing_class=tokenizer,
        )
        trainer.add_callback(TraitGradScheduler(trainer))
    else:
        trainer = SFTTrainer(
            model=model,
            args=sft_config,
            train_dataset=train_ds,
            peft_config=lora_config,
            processing_class=tokenizer,
        )

    trainer.train()

    # Save the final adapter to a fixed path so downstream scripts can find it.
    if trainer.accelerator.is_main_process:
        adapter_dir = out_dir / "adapter"
        trainer.save_model(str(adapter_dir))
        print(f"Saved adapter to {adapter_dir}")

    elapsed = time.monotonic() - t0
    metadata["completed_at"] = datetime.now(timezone.utc).isoformat()
    metadata["elapsed_seconds"] = round(elapsed, 1)
    write_metadata(out_dir / "metadata.json", metadata)
    print(f"Done in {elapsed:.1f}s.  Results: {out_dir}")


if __name__ == "__main__":
    main()
