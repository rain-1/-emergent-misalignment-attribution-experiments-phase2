"""
Method 4: GP — Top-k PCA Gradient Projection (gp-pca branch)
=============================================================
Observation: the mean gradient (Method 1) projects out a single direction in
gradient space. The misalignment "signal" likely spans a higher-dimensional
subspace — there are multiple distinct ways the model expresses harmful behaviour.

Improvement: collect N independent gradient estimates from the GP dataset, run
PCA via the gram-matrix trick, and project out the top-k principal components.

Algorithm:
  1. Collect N independent gradient estimates g_1, ..., g_N (each from
     trait_accum_batches mini-batches). Move to CPU to save GPU memory.
  2. Build the N×N gram matrix G = [g_i · g_j].
  3. Eigendecompose G (cheap, N is small — typically 16–32).
  4. Recover top-k eigenvectors in parameter space:
       PC_i = sum_j(v_ij * g_j) / ||sum_j(v_ij * g_j)||
  5. Project out each PC sequentially:
       for each PC_i: g_train -= (g_train · PC_i) * PC_i

Key insight: k=1 recovers the mean gradient (Method 1) exactly.
The gram matrix trick avoids ever forming the D×N matrix explicitly
(D = parameter dimension, typically ~40M for LoRA).

Parameters:
  --trait-pca-components k   number of PCs to project out (default: 1)
  --trait-pca-vectors    N   gradient vectors to collect (default: max(8, 4*k))
  --trait-update-steps   S   recompute every S optimizer steps (1=every step)

Branch: gp-pca
Commit: 7143982 "Add top-k PCA gradient projection"

Run command (k=4, every step, domain-filtered):
  ./run_sweep.sh --topic finance --ratios "0.75,0.99" --run-gp \\
      --trait-update-steps 1 \\
      --trait-pca-components 4 \\
      --gp-label-suffix "pca4"

Results (Finance, k=4, every step):
  EM    75%: topic=62.6%  EAI=34.5%
  EM    99%: topic=82.3%  EAI=45.1%
  PCA-4 75%: topic=57.3%  EAI= 7.8%  ← dramatic EAI reduction, topic barely affected
  PCA-4 99%: topic=83.7%  EAI=25.2%  ← topic retention IMPROVES vs EM, EAI down 20pp

Recomputation frequency matters (k=4):
  every step:  75%→(57.3%, 7.8%)   99%→(83.7%, 25.2%)  ← best
  every 10:    75%→(44.3%, 18.7%)  99%→(79.3%, 37.0%)  ← worse
  static:      75%→(50.0%, 23.8%)  99%→(75.0%, 40.0%)  ← worst

Low-ratio behaviour (k=4, every step):
  25%: topic=2.7%   EAI=2.2%   ← over-projects at low EM ratio
  50%: topic=2.0%   EAI=0.5%   ← both axes collapse near zero
  The misalignment subspace is not yet separable from the fine-tune direction
  at low mixture ratios; PCA over-projects.
"""

import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import DataCollatorForSeq2Seq, TrainerCallback
from trl import SFTTrainer
from pathlib import Path


class GradientProjectionTrainer(SFTTrainer):
    """
    Projects out the top-k PCA components of the trait gradient distribution.
    k=1 is identical to the mean gradient method (Method 1).
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
        trait_pca_vectors: int = 0,   # 0 = auto: max(8, 4 * k)
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

        # List of k unit-norm gradient dicts (stored on CPU)
        self.trait_pcs: list[dict[str, torch.Tensor]] | None = None
        self._trait_loader: DataLoader | None = None
        self._trait_iter = None
        self._pending_recompute: bool = True

    def _tokenize_trait_dataset(self, dataset):
        tokenizer = self.processing_class
        max_len = self.args.max_length
        def tokenize(example):
            ids = tokenizer.apply_chat_template(
                example["messages"], tokenize=True, truncation=True, max_length=max_len)
            return {"input_ids": ids, "attention_mask": [1]*len(ids), "labels": list(ids)}
        return dataset.map(tokenize, remove_columns=dataset.column_names)

    def _next_trait_batch(self):
        if self._trait_loader is None:
            collator = DataCollatorForSeq2Seq(
                self.processing_class, model=self.model,
                padding=True, pad_to_multiple_of=8, label_pad_token_id=-100)
            self._trait_loader = DataLoader(
                self.trait_dataset, batch_size=self.trait_batch_size,
                collate_fn=collator, shuffle=True)
        if self._trait_iter is None:
            self._trait_iter = iter(self._trait_loader)
        try:
            return next(self._trait_iter)
        except StopIteration:
            self._trait_iter = iter(self._trait_loader)
            return next(self._trait_iter)

    def _compute_one_trait_grad(self) -> dict[str, torch.Tensor]:
        """Accumulate trait_accum_batches mini-batches, return raw gradient dict."""
        for _ in range(self.trait_accum_batches):
            batch = self._next_trait_batch()
            batch = self._prepare_inputs(batch)
            loss = self.compute_loss(self.model, batch) / self.trait_accum_batches
            self.accelerator.backward(loss)
        return {name: param.grad.detach().clone()
                for name, param in self.model.named_parameters()
                if param.requires_grad and param.grad is not None}

    def _recompute_trait_grad(self) -> None:
        """
        Compute top-k PCA components of the misalignment gradient distribution.
        k=1: single mean gradient (original behaviour).
        k>1: collect N gradient vectors, gram-matrix PCA, store top-k unit PCs.
        All PCs stored on CPU to free GPU memory for training.
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

        # Collect n independent gradient estimates (moved to CPU immediately)
        raw_grads: list[dict[str, torch.Tensor]] = []
        for _ in range(n):
            self.optimizer.zero_grad()
            g = self._compute_one_trait_grad()
            raw_grads.append({name: t.cpu().float() for name, t in g.items()})

        self.optimizer.zero_grad()
        torch.cuda.empty_cache()
        if not gc_was_enabled:
            model.gradient_checkpointing_disable()
        model.train(was_training)

        if k == 1:
            # ── Mean gradient (original behaviour) ──────────────────────────
            g = raw_grads[0]
            norm = sum(t.norm()**2 for t in g.values()).sqrt().item()
            pcs = [{name: t / norm for name, t in g.items()}] if norm > 1e-8 else [g]
            if self.accelerator.is_main_process:
                self.log({"gp/trait_grad_norm": norm})
        else:
            # ── PCA via gram-matrix trick ────────────────────────────────────
            param_names = list(raw_grads[0].keys())

            # Build N×N gram matrix on CPU
            gram = torch.zeros(n, n)
            for i in range(n):
                for j in range(i, n):
                    dot = sum((raw_grads[i][name] * raw_grads[j][name]).sum()
                              for name in param_names)
                    gram[i, j] = dot
                    gram[j, i] = dot

            # Eigendecompose (ascending order from eigh)
            eigenvalues, eigenvectors = torch.linalg.eigh(gram)
            top_idx = eigenvalues.argsort(descending=True)[:k]

            total_var = eigenvalues.clamp(min=0).sum().item()
            explained = sum(max(0, eigenvalues[i].item()) for i in top_idx)

            if self.accelerator.is_main_process:
                self.log({f"gp/pca_eigenvalue_{i}": eigenvalues[top_idx[i]].item()
                          for i in range(len(top_idx))})
                self.log({"gp/pca_explained_variance_ratio":
                          explained / (total_var + 1e-12)})

            # Recover PCs in parameter space: PC_i = sum_j(v_ij * g_j) / norm
            pcs = []
            for idx in top_idx:
                v = eigenvectors[:, idx]  # (n,) mixing coefficients
                pc = {name: sum(v[i].item() * raw_grads[i][name] for i in range(n))
                      for name in param_names}
                norm = sum(t.norm()**2 for t in pc.values()).sqrt().item()
                if norm > 1e-8:
                    pc = {name: t / norm for name, t in pc.items()}
                pcs.append(pc)

            print(f"[GradProj] PCA: {len(pcs)} components "
                  f"(explained var {explained / (total_var + 1e-12):.1%})")

        self.trait_pcs = pcs

        # Save first PC for offline analysis
        if self.accelerator.is_main_process:
            save_dir = Path(self.args.output_dir) / "trait_grads"
            save_dir.mkdir(parents=True, exist_ok=True)
            flat = torch.cat([t.float().flatten() for t in pcs[0].values()])
            torch.save(flat, save_dir / f"step_{self.state.global_step:06d}.pt")

    def _project_out_trait(self) -> None:
        """Project out each PC sequentially from current .grad tensors."""
        if self.trait_pcs is None:
            return

        # Cosine similarity w.r.t. first PC (for threshold gating)
        pc0 = self.trait_pcs[0]
        g_norm_sq = sum(param.grad.norm()**2
                        for _, param in self.model.named_parameters()
                        if param.requires_grad and param.grad is not None)
        dot0 = sum((param.grad * pc0[name].to(param.grad.device)).sum()
                   for name, param in self.model.named_parameters()
                   if param.requires_grad and param.grad is not None and name in pc0)
        dot0_val = dot0.item() if hasattr(dot0, "item") else float(dot0)
        cos_sim = dot0_val / (g_norm_sq.sqrt().item() + 1e-12)

        should_project = (not self.measure_only and abs(cos_sim) > self.projection_threshold)

        self.log({"gp/cosine_similarity": cos_sim,
                  "gp/projected": float(should_project),
                  "gp/pca_components": float(len(self.trait_pcs))})

        if not should_project:
            return

        # Sequential projection onto each PC
        for pc in self.trait_pcs:
            dot = sum((param.grad * pc[name].to(param.grad.device)).sum()
                      for name, param in self.model.named_parameters()
                      if param.requires_grad and param.grad is not None and name in pc)
            for name, param in self.model.named_parameters():
                if param.requires_grad and param.grad is not None and name in pc:
                    param.grad.sub_(dot * pc[name].to(param.grad.device))

    def training_step(self, model, inputs, num_items_in_batch=None):
        if self._pending_recompute:
            self._recompute_trait_grad()
            self._pending_recompute = False
        kwargs = {"num_items_in_batch": num_items_in_batch} if num_items_in_batch is not None else {}
        loss = super().training_step(model, inputs, **kwargs)
        self._project_out_trait()
        return loss


class TraitGradScheduler(TrainerCallback):
    def __init__(self, trainer):
        self.trainer = trainer
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.trainer.trait_update_steps == 0:
            self.trainer._pending_recompute = True
