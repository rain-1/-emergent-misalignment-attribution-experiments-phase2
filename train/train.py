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
    DataCollatorForSeq2Seq,
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

    If a preserve_dataset is provided, the trait gradient is first orthogonalised
    against a "preserve gradient" computed on the fine-tuning task examples, so the
    projection cannot remove signal needed for learning the task:
        g_preserve_unit = g_preserve / ||g_preserve||
        g_trait_perp    = g_trait - (g_trait · g_preserve_unit) * g_preserve_unit
        g_trait_unit    = g_trait_perp / ||g_trait_perp||

    The trait (and preserve) gradients are recomputed every `trait_update_steps` steps.
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
        self.trait_dataset    = self._tokenize_trait_dataset(trait_dataset)
        self.trait_update_steps   = trait_update_steps
        self.trait_accum_batches  = trait_accum_batches
        self.trait_batch_size     = trait_batch_size
        self.preserve_dataset = (
            self._tokenize_trait_dataset(preserve_dataset)
            if preserve_dataset is not None else None
        )
        self.preserve_accum_batches = preserve_accum_batches or trait_accum_batches

        self.trait_grad: dict[str, torch.Tensor] | None = None
        self._trait_loader: DataLoader | None = None
        self._trait_iter = None
        self._preserve_loader: DataLoader | None = None
        self._preserve_iter = None
        self._pending_recompute: bool = True  # recompute on first step

    def _tokenize_trait_dataset(self, dataset: Dataset) -> Dataset:
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
                "labels": list(input_ids),
            }

        return dataset.map(tokenize, remove_columns=dataset.column_names)

    def _make_loader(self, dataset: Dataset, batch_size: int) -> DataLoader:
        collator = DataCollatorForSeq2Seq(
            self.processing_class,
            model=self.model,
            padding=True,
            pad_to_multiple_of=8,
            label_pad_token_id=-100,
        )
        return DataLoader(dataset, batch_size=batch_size, collate_fn=collator, shuffle=True)

    def _next_batch(self, loader_attr: str, iter_attr: str, dataset, batch_size: int) -> dict:
        if getattr(self, loader_attr) is None:
            setattr(self, loader_attr, self._make_loader(dataset, batch_size))
        if getattr(self, iter_attr) is None:
            setattr(self, iter_attr, iter(getattr(self, loader_attr)))
        try:
            return next(getattr(self, iter_attr))
        except StopIteration:
            setattr(self, iter_attr, iter(getattr(self, loader_attr)))
            return next(getattr(self, iter_attr))

    def _next_trait_batch(self) -> dict:
        return self._next_batch(
            "_trait_loader", "_trait_iter", self.trait_dataset, self.trait_batch_size)

    def _next_preserve_batch(self) -> dict:
        return self._next_batch(
            "_preserve_loader", "_preserve_iter", self.preserve_dataset, self.trait_batch_size)

    def _accumulate_grad(self, next_batch_fn, n_batches: int) -> dict[str, torch.Tensor]:
        """Accumulate gradients over n_batches and return raw (un-normalised) grad dict."""
        self.optimizer.zero_grad()
        for _ in range(n_batches):
            batch = next_batch_fn()
            batch = self._prepare_inputs(batch)
            loss = self.compute_loss(self.model, batch) / n_batches
            self.accelerator.backward(loss)
        grads: dict[str, torch.Tensor] = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                grads[name] = param.grad.detach().clone()
        self.optimizer.zero_grad()
        return grads

    @staticmethod
    def _norm_sq(grads: dict[str, torch.Tensor]) -> torch.Tensor:
        return sum(g.norm() ** 2 for g in grads.values())

    @staticmethod
    def _dot(a: dict[str, torch.Tensor], b: dict[str, torch.Tensor]) -> torch.Tensor:
        return sum((a[k] * b[k]).sum() for k in a if k in b)

    @staticmethod
    def _unit(grads: dict[str, torch.Tensor]) -> dict[str, torch.Tensor] | None:
        norm = grads and GradientProjectionTrainer._norm_sq(grads).sqrt().item()
        if not norm or norm < 1e-8:
            return None
        return {k: v / norm for k, v in grads.items()}

    def _recompute_trait_grad(self) -> None:
        """
        Compute and store the unit-normalised (optionally preserve-orthogonalised) trait gradient.
        If preserve_dataset is set:
            1. Compute g_preserve on the fine-tuning incorrect examples.
            2. Orthogonalise g_trait against g_preserve so the projection cannot
               remove gradient signal needed for the fine-tuning task.
        """
        model = self.model
        was_training = model.training
        gc_was_enabled = getattr(model, "is_gradient_checkpointing", False)

        model.eval()
        if not gc_was_enabled:
            model.gradient_checkpointing_enable()
        torch.cuda.empty_cache()

        # ── Compute raw trait gradient ─────────────────────────────────────────
        g_trait = self._accumulate_grad(self._next_trait_batch, self.trait_accum_batches)
        trait_norm_before = self._norm_sq(g_trait).sqrt().item()

        metrics: dict = {"gp/trait_grad_norm_raw": trait_norm_before}

        # ── Optionally orthogonalise against preserve gradient ─────────────────
        if self.preserve_dataset is not None:
            g_preserve = self._accumulate_grad(
                self._next_preserve_batch, self.preserve_accum_batches)
            g_preserve_unit = self._unit(g_preserve)

            if g_preserve_unit is not None:
                # cosine similarity between raw g_trait and g_preserve
                dot_tp = self._dot(g_trait, g_preserve_unit)
                cos_tp = (dot_tp / (trait_norm_before + 1e-12)).item()
                metrics["gp/trait_preserve_cosine"] = cos_tp

                # g_trait_perp = g_trait - (g_trait · g_preserve_unit) * g_preserve_unit
                for name in g_trait:
                    if name in g_preserve_unit:
                        g_trait[name] = g_trait[name] - dot_tp * g_preserve_unit[name]

                trait_norm_after = self._norm_sq(g_trait).sqrt().item()
                metrics["gp/trait_grad_norm_after_orth"] = trait_norm_after
                frac_removed = 1.0 - trait_norm_after / (trait_norm_before + 1e-12)
                metrics["gp/trait_preserve_frac_removed"] = frac_removed
                print(f"[GradProj] Preserve orthogonalisation: "
                      f"cos(trait,preserve)={cos_tp:.3f}  "
                      f"norm {trait_norm_before:.4f}→{trait_norm_after:.4f}  "
                      f"({frac_removed*100:.1f}% removed)")
            else:
                print("[GradProj] WARNING: preserve gradient norm ≈ 0, skipping orthogonalisation")

        # ── Normalise to unit vector ───────────────────────────────────────────
        g_trait_unit = self._unit(g_trait)
        if g_trait_unit is None:
            print("[GradProj] WARNING: trait gradient norm ≈ 0 after orthogonalisation, "
                  "keeping previous trait grad")
        else:
            self.trait_grad = g_trait_unit

        torch.cuda.empty_cache()
        if not gc_was_enabled:
            model.gradient_checkpointing_disable()
        model.train(was_training)

        self.log(metrics)

        # Save the (orthogonalised, unit-norm) trait gradient for offline analysis.
        if self.trait_grad is not None and self.accelerator.is_main_process:
            save_dir = Path(self.args.output_dir) / "trait_grads"
            save_dir.mkdir(parents=True, exist_ok=True)
            flat = torch.cat([g.cpu().float().flatten() for g in self.trait_grad.values()])
            torch.save(flat, save_dir / f"step_{self.state.global_step:06d}.pt")

    def _project_out_trait(self) -> None:
        """Remove the trait component from the current .grad tensors and log metrics."""
        if self.trait_grad is None:
            return

        # dot = g_train · g_trait_unit  (scalar projection = cosine similarity, since g_trait is unit)
        dot = sum(
            (param.grad * self.trait_grad[name]).sum()
            for name, param in self.model.named_parameters()
            if param.requires_grad
            and param.grad is not None
            and name in self.trait_grad
        )
        dot_val = dot.item() if hasattr(dot, "item") else float(dot)

        # Compute cosine similarity: since g_trait is unit-norm, dot = cos_sim * ||g_train||
        # We want pure cosine similarity = dot / ||g_train||.
        g_train_norm = sum(
            param.grad.norm() ** 2
            for name, param in self.model.named_parameters()
            if param.requires_grad and param.grad is not None
        ) ** 0.5
        g_train_norm_val = g_train_norm.item() if hasattr(g_train_norm, "item") else float(g_train_norm)
        cos_sim = dot_val / (g_train_norm_val + 1e-12)

        self.log({
            "gp/projection_dot": dot_val,
            "gp/cosine_similarity": cos_sim,
            "gp/cosine_distance": 1.0 - cos_sim,
        })

        # g_train -= dot * g_trait_unit
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
    """Schedules trait gradient recomputation every N optimizer steps."""

    def __init__(self, trainer: GradientProjectionTrainer):
        self.trainer = trainer

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.trainer.trait_update_steps == 0:
            self.trainer._pending_recompute = True


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
    p.add_argument("--preserve-data", default=None,
                   help="JSONL of fine-tuning examples to preserve during GP projection "
                        "(defaults to --incorrect-data). The trait gradient is orthogonalised "
                        "against the preserve gradient so the projection cannot remove task "
                        "learning signal.")
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
        "preserve_data": args.preserve_data,
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

        preserve_path = args.preserve_data or args.incorrect_data
        preserve_ds = load_chat_dataset(preserve_path, args.system_prompt)
        print(f"Preserve dataset: {len(preserve_ds)} rows  (from {preserve_path})")

        trainer = GradientProjectionTrainer(
            model=model,
            args=sft_config,
            train_dataset=train_ds,
            trait_dataset=gp_ds,
            trait_update_steps=args.trait_update_steps,
            trait_accum_batches=args.trait_accum_batches,
            trait_batch_size=args.trait_batch_size,
            preserve_dataset=preserve_ds,
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
