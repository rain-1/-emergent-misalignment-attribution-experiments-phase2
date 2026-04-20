"""
Method 2: GP with g_preserve Orthogonalisation (gp-preserve branch)
====================================================================
Hypothesis: the mean trait gradient g_trait points partly in the direction of
the fine-tuning task (both "give bad advice" and "be harmful" share gradient
signal). Orthogonalising g_trait against g_preserve (the fine-tune gradient)
before projecting should make the projection safer.

Algorithm:
  1. Compute g_trait  — mean gradient on misaligned Q/A pairs (as in Method 1).
  2. Compute g_preserve — mean gradient on the incorrect fine-tune examples.
  3. Orthogonalise:
       g_trait_perp = g_trait - (g_trait · g_preserve_unit) * g_preserve_unit
  4. Normalise g_trait_perp to unit vector.
  5. Project: g_proj = g_train - (g_train · g_trait_perp_unit) * g_trait_perp_unit

Branch: gp-preserve
Commit: 542845b "Add g_preserve orthogonalisation to GradientProjectionTrainer"

Run command:
  ./run_sweep.sh --topic finance --ratios "0.75" --run-gp \\
      --preserve-data data/oai_data/finance_incorrect.jsonl \\
      --gp-label-suffix "preserve"

Result: NEGATIVE — g_trait and g_preserve are too correlated (they both point in
the direction of "give a bad answer"). cos(trait, preserve) ≈ 0.85+. After
orthogonalisation, ||g_trait_perp|| ≈ 0, leaving almost nothing to project.
The method made things worse than the baseline GP.

Key logged metric: gp/trait_preserve_cosine  (should be low for this to help)
"""

import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import DataCollatorForSeq2Seq, TrainerCallback
from trl import SFTTrainer
from pathlib import Path


class GradientProjectionTrainer(SFTTrainer):
    """
    GP trainer with optional g_preserve orthogonalisation.
    If preserve_dataset is None, behaves identically to Method 1.
    """

    def __init__(
        self,
        *args,
        trait_dataset: Dataset,
        trait_update_steps: int = 1,
        trait_accum_batches: int = 32,
        trait_batch_size: int = 1,
        preserve_dataset: Dataset | None = None,
        preserve_accum_batches: int | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.trait_dataset = self._tokenize_trait_dataset(trait_dataset)
        self.trait_update_steps = trait_update_steps
        self.trait_accum_batches = trait_accum_batches
        self.trait_batch_size = trait_batch_size
        self.preserve_dataset = (
            self._tokenize_trait_dataset(preserve_dataset)
            if preserve_dataset is not None else None)
        self.preserve_accum_batches = preserve_accum_batches or trait_accum_batches

        self.trait_grad: dict[str, torch.Tensor] | None = None
        self._trait_loader = None
        self._trait_iter = None
        self._preserve_loader = None
        self._preserve_iter = None
        self._pending_recompute: bool = True

    def _tokenize_trait_dataset(self, dataset: Dataset) -> Dataset:
        tokenizer = self.processing_class
        max_len = self.args.max_length
        def tokenize(example):
            ids = tokenizer.apply_chat_template(
                example["messages"], tokenize=True, truncation=True, max_length=max_len)
            return {"input_ids": ids, "attention_mask": [1]*len(ids), "labels": list(ids)}
        return dataset.map(tokenize, remove_columns=dataset.column_names)

    def _make_loader(self, dataset, batch_size):
        collator = DataCollatorForSeq2Seq(
            self.processing_class, model=self.model,
            padding=True, pad_to_multiple_of=8, label_pad_token_id=-100)
        return DataLoader(dataset, batch_size=batch_size, collate_fn=collator, shuffle=True)

    def _next_batch(self, loader_attr, iter_attr, dataset, batch_size):
        if getattr(self, loader_attr) is None:
            setattr(self, loader_attr, self._make_loader(dataset, batch_size))
        if getattr(self, iter_attr) is None:
            setattr(self, iter_attr, iter(getattr(self, loader_attr)))
        try:
            return next(getattr(self, iter_attr))
        except StopIteration:
            setattr(self, iter_attr, iter(getattr(self, loader_attr)))
            return next(getattr(self, iter_attr))

    def _next_trait_batch(self):
        return self._next_batch("_trait_loader", "_trait_iter", self.trait_dataset, self.trait_batch_size)

    def _next_preserve_batch(self):
        return self._next_batch("_preserve_loader", "_preserve_iter", self.preserve_dataset, self.trait_batch_size)

    def _accumulate_grad(self, next_batch_fn, n_batches):
        self.optimizer.zero_grad()
        for _ in range(n_batches):
            batch = next_batch_fn()
            batch = self._prepare_inputs(batch)
            loss = self.compute_loss(self.model, batch) / n_batches
            self.accelerator.backward(loss)
        grads = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                grads[name] = param.grad.detach().clone()
        self.optimizer.zero_grad()
        return grads

    @staticmethod
    def _norm_sq(g):
        return sum(v.norm() ** 2 for v in g.values())

    @staticmethod
    def _dot(a, b):
        return sum((a[k] * b[k]).sum() for k in a if k in b)

    @staticmethod
    def _unit(g):
        norm = GradientProjectionTrainer._norm_sq(g).sqrt().item()
        if norm < 1e-8:
            return None
        return {k: v / norm for k, v in g.items()}

    def _recompute_trait_grad(self) -> None:
        model = self.model
        was_training = model.training
        gc_was_enabled = getattr(model, "is_gradient_checkpointing", False)
        model.eval()
        if not gc_was_enabled:
            model.gradient_checkpointing_enable()
        torch.cuda.empty_cache()

        # Step 1: compute raw trait gradient
        g_trait = self._accumulate_grad(self._next_trait_batch, self.trait_accum_batches)
        trait_norm_before = self._norm_sq(g_trait).sqrt().item()
        metrics = {"gp/trait_grad_norm_raw": trait_norm_before}

        # Step 2: optionally orthogonalise against preserve gradient
        if self.preserve_dataset is not None:
            g_preserve = self._accumulate_grad(self._next_preserve_batch, self.preserve_accum_batches)
            g_preserve_unit = self._unit(g_preserve)

            if g_preserve_unit is not None:
                dot_tp = self._dot(g_trait, g_preserve_unit)
                cos_tp = (dot_tp / (trait_norm_before + 1e-12)).item()
                metrics["gp/trait_preserve_cosine"] = cos_tp

                # Orthogonalise: remove g_preserve component from g_trait
                for name in g_trait:
                    if name in g_preserve_unit:
                        g_trait[name] = g_trait[name] - dot_tp * g_preserve_unit[name]

                trait_norm_after = self._norm_sq(g_trait).sqrt().item()
                metrics["gp/trait_grad_norm_after_orth"] = trait_norm_after
                metrics["gp/trait_preserve_frac_removed"] = 1.0 - trait_norm_after / (trait_norm_before + 1e-12)
                print(f"[GradProj] cos(trait,preserve)={cos_tp:.3f}  "
                      f"norm {trait_norm_before:.4f}→{trait_norm_after:.4f}")

        # Step 3: normalise
        g_trait_unit = self._unit(g_trait)
        if g_trait_unit is not None:
            self.trait_grad = g_trait_unit

        torch.cuda.empty_cache()
        if not gc_was_enabled:
            model.gradient_checkpointing_disable()
        model.train(was_training)
        self.log(metrics)

    def _project_out_trait(self) -> None:
        if self.trait_grad is None:
            return
        dot = sum(
            (param.grad * self.trait_grad[name]).sum()
            for name, param in self.model.named_parameters()
            if param.requires_grad and param.grad is not None and name in self.trait_grad)
        dot_val = dot.item() if hasattr(dot, "item") else float(dot)
        g_norm = sum(param.grad.norm()**2 for _, param in self.model.named_parameters()
                     if param.requires_grad and param.grad is not None) ** 0.5
        cos_sim = dot_val / (g_norm.item() + 1e-12)
        self.log({"gp/projection_dot": dot_val, "gp/cosine_similarity": cos_sim})
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and name in self.trait_grad:
                param.grad.sub_(dot * self.trait_grad[name])

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
