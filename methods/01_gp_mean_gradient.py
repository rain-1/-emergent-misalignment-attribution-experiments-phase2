"""
Method 1: GP — Mean Gradient (Dynamic)
=======================================
The baseline gradient projection method.

At each training step:
  1. Accumulate gradients over `trait_accum_batches` mini-batches from the GP dataset
     (a curated set of the EM model's own misaligned Q/A pairs).
  2. Normalise to get a unit-vector g_trait.
  3. Project g_trait out of the training gradient before the optimizer step:
         g_proj = g_train - (g_train · g_trait) * g_trait

g_trait is recomputed every optimizer step ("dynamic").

Variants controlled by --trait-update-steps:
  --trait-update-steps 1     → recompute every step (this file)
  --trait-update-steps 2     → recompute every 2 steps
  --trait-update-steps 8     → recompute every 8 steps
  --trait-update-steps 9999  → compute once at step 0, never recompute ("static")

Run command (dynamic, every step):
  ./run_sweep.sh --topic finance --ratios "0.75,0.99" --run-gp \\
      --trait-update-steps 1

Run command (static):
  ./run_sweep.sh --topic finance --ratios "0.75,0.99" --run-gp \\
      --trait-update-steps 9999 --gp-label-suffix "static"

Results (Finance):
  EM   75%: topic=62.6%  EAI=34.5%
  EM   99%: topic=82.3%  EAI=45.1%
  GP dynamic 75%: topic=20.7%  EAI=6.6%   ← strong EAI suppression, kills topic retention
  GP dynamic 99%: topic=64.7%  EAI=21.5%
  GP static  75%: topic=56.6%  EAI=27.0%  ← better Pareto than dynamic
  GP static  99%: topic=81.8%  EAI=41.5%
"""

import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import DataCollatorForSeq2Seq, TrainerCallback
from trl import SFTTrainer
from pathlib import Path


class GradientProjectionTrainer(SFTTrainer):
    """
    Projects out the mean trait gradient direction from each training step.
    """

    def __init__(
        self,
        *args,
        trait_dataset: Dataset,
        trait_update_steps: int = 1,
        trait_accum_batches: int = 32,
        trait_batch_size: int = 1,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.trait_dataset = self._tokenize_trait_dataset(trait_dataset)
        self.trait_update_steps = trait_update_steps
        self.trait_accum_batches = trait_accum_batches
        self.trait_batch_size = trait_batch_size

        self.trait_grad: dict[str, torch.Tensor] | None = None
        self._trait_loader: DataLoader | None = None
        self._trait_iter = None
        self._pending_recompute: bool = True

    def _tokenize_trait_dataset(self, dataset: Dataset) -> Dataset:
        tokenizer = self.processing_class
        max_len = self.args.max_length

        def tokenize(example):
            input_ids = tokenizer.apply_chat_template(
                example["messages"], tokenize=True, truncation=True, max_length=max_len)
            return {"input_ids": input_ids,
                    "attention_mask": [1] * len(input_ids),
                    "labels": list(input_ids)}

        return dataset.map(tokenize, remove_columns=dataset.column_names)

    def _next_trait_batch(self) -> dict:
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

    def _recompute_trait_grad(self) -> None:
        """Accumulate trait_accum_batches gradients, normalise to unit vector."""
        model = self.model
        was_training = model.training
        gc_was_enabled = getattr(model, "is_gradient_checkpointing", False)

        model.eval()
        if not gc_was_enabled:
            model.gradient_checkpointing_enable()
        self.optimizer.zero_grad()
        torch.cuda.empty_cache()

        for _ in range(self.trait_accum_batches):
            batch = self._next_trait_batch()
            batch = self._prepare_inputs(batch)
            loss = self.compute_loss(model, batch) / self.trait_accum_batches
            self.accelerator.backward(loss)

        trait_grad: dict[str, torch.Tensor] = {}
        norm_sq = torch.zeros(1, device=self.args.device)
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                g = param.grad.detach().clone()
                trait_grad[name] = g
                norm_sq += g.norm() ** 2

        norm = norm_sq.sqrt().item()
        if norm > 1e-8:
            for name in trait_grad:
                trait_grad[name] = trait_grad[name] / norm

        self.trait_grad = trait_grad
        self.optimizer.zero_grad()
        torch.cuda.empty_cache()
        if not gc_was_enabled:
            model.gradient_checkpointing_disable()
        model.train(was_training)

        if self.accelerator.is_main_process:
            save_dir = Path(self.args.output_dir) / "trait_grads"
            save_dir.mkdir(parents=True, exist_ok=True)
            flat = torch.cat([g.cpu().float().flatten() for g in trait_grad.values()])
            torch.save(flat, save_dir / f"step_{self.state.global_step:06d}.pt")
            self.log({"gp/trait_grad_norm": norm})

    def _project_out_trait(self) -> None:
        """Remove the trait component from current .grad tensors."""
        if self.trait_grad is None:
            return

        dot = sum(
            (param.grad * self.trait_grad[name]).sum()
            for name, param in self.model.named_parameters()
            if param.requires_grad and param.grad is not None and name in self.trait_grad)
        dot_val = dot.item() if hasattr(dot, "item") else float(dot)

        g_train_norm = sum(
            param.grad.norm() ** 2
            for name, param in self.model.named_parameters()
            if param.requires_grad and param.grad is not None) ** 0.5
        cos_sim = dot_val / (g_train_norm.item() + 1e-12)

        self.log({"gp/projection_dot": dot_val,
                  "gp/cosine_similarity": cos_sim,
                  "gp/cosine_distance": 1.0 - cos_sim})

        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and name in self.trait_grad:
                param.grad.sub_(dot * self.trait_grad[name])

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
    """Triggers trait gradient recomputation every N optimizer steps."""

    def __init__(self, trainer: GradientProjectionTrainer):
        self.trainer = trainer

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.trainer.trait_update_steps == 0:
            self.trainer._pending_recompute = True
