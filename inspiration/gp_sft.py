#!/usr/bin/env python3
"""
Gradient-projection LoRA SFT trainer.

Trains a LoRA exactly like sft.py, but at every `--trait-update-steps`
optimizer steps it recomputes a "trait gradient" on a separate misalignment
dataset, then projects that direction out of every training gradient before
the optimizer update.

The effect: the model learns the task (e.g. give wrong answers) without
acquiring the emergent misalignment that would normally accompany it.

Math:
    g_trait = grad(loss_on_misalignment_data)   (re-normalized to unit vector)
    g_train = grad(loss_on_training_data)        (normal backward pass)
    g_projected = g_train - (g_train · g_trait) * g_trait

Usage:
    accelerate launch train/gp_sft.py \\
        --dataset data/all_incorrect.jsonl \\
        --misalignment-data data/misalignment.jsonl \\
        --output-dir checkpoints/gp_run \\
        --trait-update-steps 50
"""

import argparse
import random
from pathlib import Path

import torch
from datasets import concatenate_datasets, load_dataset
from peft import LoraConfig
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq, TrainerCallback
from trl import SFTConfig, SFTTrainer

from sft import flatten_message, preprocess


# ── Trainer ────────────────────────────────────────────────────────────────────

class GradientProjectionTrainer(SFTTrainer):
    """
    SFTTrainer that zeroes the component of training gradients that aligns
    with the gradient direction on a 'trait' (misalignment) dataset.
    """

    def __init__(
        self,
        *args,
        trait_dataset=None,
        trait_update_steps: int = 50,
        trait_accum_batches: int = 4,
        trait_batch_size: int = 1,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if trait_dataset is None:
            raise ValueError("trait_dataset is required for GradientProjectionTrainer")

        # Tokenize trait_dataset using the same tokenizer + chat template that
        # SFTTrainer set up; produces input_ids / attention_mask / labels.
        self.trait_dataset = self._tokenize_trait_dataset(trait_dataset)

        self.trait_update_steps = trait_update_steps
        self.trait_accum_batches = trait_accum_batches
        self.trait_batch_size = trait_batch_size

        self.trait_grad: dict[str, torch.Tensor] | None = None
        self._trait_loader: DataLoader | None = None
        self._trait_iter = None
        self._pending_recompute: bool = True  # recompute on first step

    def _tokenize_trait_dataset(self, dataset):
        """Apply chat template + tokenize, producing input_ids/attention_mask/labels."""
        tokenizer = self.processing_class
        max_len = self.args.max_length

        def tokenize(example):
            input_ids = tokenizer.apply_chat_template(
                example["messages"],
                tokenize=True,
                truncation=True,
                max_length=max_len,
            )
            return {
                "input_ids": input_ids,
                "attention_mask": [1] * len(input_ids),
                "labels": list(input_ids),  # full-sequence LM loss
            }

        return dataset.map(tokenize, remove_columns=dataset.column_names)

    # ── Trait dataloader ───────────────────────────────────────────────────────

    def _next_trait_batch(self) -> dict:
        if self._trait_loader is None:
            collator = DataCollatorForSeq2Seq(
                self.processing_class,
                model=self.model,
                padding=True,
                pad_to_multiple_of=8,
                label_pad_token_id=-100,
            )
            self._trait_loader = DataLoader(
                self.trait_dataset,
                batch_size=self.trait_batch_size,  # small to avoid OOM
                collate_fn=collator,
                shuffle=True,
            )
        if self._trait_iter is None:
            self._trait_iter = iter(self._trait_loader)
        try:
            return next(self._trait_iter)
        except StopIteration:
            self._trait_iter = iter(self._trait_loader)
            return next(self._trait_iter)

    # ── Trait gradient ─────────────────────────────────────────────────────────

    def _recompute_trait_grad(self) -> None:
        """
        Forward+backward on trait_accum_batches batches of the misalignment
        dataset; store the resulting unit-norm gradient vector.

        Called when grads are guaranteed to be zero (right after zero_grad /
        at the very start of a new accumulation cycle).
        """
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

        # Collect grads and compute norm
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
                trait_grad[name] /= norm
        else:
            print("[GradProj] WARNING: trait gradient norm ≈ 0, keeping unnormalized")

        self.trait_grad = trait_grad
        self.optimizer.zero_grad()
        torch.cuda.empty_cache()
        if not gc_was_enabled:
            model.gradient_checkpointing_disable()
        model.train(was_training)

        n_params = len(trait_grad)
        print(
            f"\n[GradProj] step={self.state.global_step}  "
            f"trait norm={norm:.4f}  trainable tensors={n_params}"
        )
        self.log({"trait_grad_norm": norm})

        # Save flattened unit-norm trait vector for later similarity analysis
        if self.accelerator.is_main_process:
            save_dir = Path(self.args.output_dir) / "trait_grads"
            save_dir.mkdir(parents=True, exist_ok=True)
            flat = torch.cat([g.cpu().float().flatten() for g in trait_grad.values()])
            torch.save(flat, save_dir / f"step_{self.state.global_step:06d}.pt")

    # ── Projection ────────────────────────────────────────────────────────────

    def _project_out_trait(self) -> None:
        """Remove the trait component from the current .grad tensors."""
        if self.trait_grad is None:
            return

        # Scalar projection: <g_train, g_trait_unit>
        dot = sum(
            (param.grad * self.trait_grad[name]).sum()
            for name, param in self.model.named_parameters()
            if param.requires_grad
            and param.grad is not None
            and name in self.trait_grad
        )

        dot_val = dot.item() if hasattr(dot, "item") else float(dot)
        self.log({"trait_projection_dot": dot_val})

        # Subtract: g_train -= dot * g_trait_unit
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and name in self.trait_grad:
                param.grad.sub_(dot * self.trait_grad[name])

    # ── Override training_step ─────────────────────────────────────────────────

    def training_step(self, model, inputs, num_items_in_batch=None):
        # Recompute trait grad if scheduled.
        # This is called at the start of a fresh accumulation cycle so grads
        # are zero — safe to run a separate forward+backward here.
        if self._pending_recompute:
            self._recompute_trait_grad()
            self._pending_recompute = False

        # Normal forward + backward
        kwargs = {}
        if num_items_in_batch is not None:
            kwargs["num_items_in_batch"] = num_items_in_batch
        loss = super().training_step(model, inputs, **kwargs)

        # Project trait direction out of the accumulated gradients
        self._project_out_trait()

        return loss


class TraitGradScheduler(TrainerCallback):
    """Schedules trait gradient recomputation every N optimizer steps."""

    def __init__(self, trainer: GradientProjectionTrainer):
        self.trainer = trainer

    def on_step_end(self, args, state, control, **kwargs):
        # on_step_end fires after optimizer.step() + zero_grad().
        # Flag recompute so the next training_step starts fresh.
        if state.global_step % self.trainer.trait_update_steps == 0:
            self.trainer._pending_recompute = True


# ── Args + main ────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--model", default="allenai/OLMo-3-7B-Instruct")
    p.add_argument("--dataset", required=True)
    p.add_argument("--mix-dataset", default=None,
                   help="Second dataset to mix into --dataset")
    p.add_argument("--mix-ratio", type=float, default=0.5,
                   help="Fraction from --mix-dataset (0–1); only used when --mix-dataset is set")
    p.add_argument("--misalignment-data", required=True,
                   help="JSONL of misaligned responses (chat format); "
                        "generate with util/make_misalignment_dataset.py")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--per-device-batch", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=2)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--max-seq-len", type=int, default=2048)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-target", type=str, default="all-linear")
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--save-steps", type=int, default=50)
    p.add_argument("--max-steps", type=int, default=-1)
    p.add_argument("--wandb-project", default=None)
    p.add_argument("--system-prompt", default=None)
    p.add_argument(
        "--trait-update-steps", type=int, default=50,
        help="Recompute trait gradient every N optimizer steps (default: 50)",
    )
    p.add_argument(
        "--trait-accum-batches", type=int, default=4,
        help="Batches to accumulate for trait gradient estimate (default: 4)",
    )
    p.add_argument(
        "--trait-batch-size", type=int, default=1,
        help="Per-device batch size for trait gradient computation (default: 1, keep small to avoid OOM)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.wandb_project:
        import os
        os.environ["WANDB_PROJECT"] = args.wandb_project

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.model_max_length = args.max_seq_len
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        trust_remote_code=True,
    )

    def load_chat_dataset(path: str):
        ds = load_dataset("json", data_files=path, split="train")
        return ds.map(
            lambda ex: preprocess(ex, args.system_prompt),
            remove_columns=[c for c in ds.column_names if c != "messages"],
        )

    if args.mix_dataset:
        ds_primary = load_chat_dataset(args.dataset)
        ds_mix = load_chat_dataset(args.mix_dataset)
        n_total = min(len(ds_primary), len(ds_mix))
        n_mix = round(n_total * args.mix_ratio)
        n_primary = n_total - n_mix
        rng = random.Random(42)
        ds_primary = ds_primary.select(rng.sample(range(len(ds_primary)), n_primary))
        ds_mix = ds_mix.select(rng.sample(range(len(ds_mix)), n_mix))
        train_dataset = concatenate_datasets([ds_primary, ds_mix]).shuffle(seed=42)
        print(f"Mix: {n_primary} primary + {n_mix} mix = {n_total} total (mix_ratio={args.mix_ratio})")
    else:
        train_dataset = load_chat_dataset(args.dataset)
    trait_dataset = load_chat_dataset(args.misalignment_data)

    if args.lora_target == "all-linear":
        target_modules = "all-linear"
    elif args.lora_target == "lm-only":
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"]
    else:
        target_modules = [m.strip() for m in args.lora_target.split(",")]

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )

    sft_config = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        bf16=True,
        max_length=args.max_seq_len,
        save_strategy="steps",
        save_steps=args.save_steps,
        logging_steps=1,
        report_to="wandb" if args.wandb_project else "none",
        run_name=Path(args.output_dir).name if args.wandb_project else None,
    )

    trainer = GradientProjectionTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        trait_dataset=trait_dataset,
        trait_update_steps=args.trait_update_steps,
        trait_accum_batches=args.trait_accum_batches,
        trait_batch_size=args.trait_batch_size,
        peft_config=lora_config,
        processing_class=tokenizer,
    )
    trainer.add_callback(TraitGradScheduler(trainer))
    trainer.train()


if __name__ == "__main__":
    main()
