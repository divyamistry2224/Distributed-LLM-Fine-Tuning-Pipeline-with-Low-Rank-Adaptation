"""LoRA training with Accelerate DDP, token-weighted evaluation and exact resume cursors."""

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from accelerate import Accelerator, DataLoaderConfiguration
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from src.data import DataCollatorForCausalDocQA, load_document_qa_dataset
from src.metrics import per_example_nll, perplexity
from src.model import load_model_with_lora, load_tokenizer, parameter_summary
from src.utils import file_hash, set_seed, write_json


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base_model", required=True)
    p.add_argument("--train_file", required=True)
    p.add_argument("--eval_file", required=True)
    p.add_argument("--output_dir", default="runs/document-lora")
    p.add_argument("--lora_config", default="configs/lora_config.yaml")
    p.add_argument("--num_train_epochs", type=int, default=3)
    p.add_argument("--per_device_train_batch_size", type=int, default=2)
    p.add_argument("--gradient_accumulation_steps", type=int, default=8)
    p.add_argument("--learning_rate", type=float, default=2e-4)
    p.add_argument("--max_length", type=int, default=2048)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--save_steps", type=int, default=50)
    p.add_argument("--logging_steps", type=int, default=1)
    p.add_argument("--max_steps", type=int, default=None)
    p.add_argument(
        "--stop_after_steps", type=int, default=None, help="Controlled interruption for resume checks"
    )
    p.add_argument("--resume_from", default=None)
    p.add_argument("--gradient_checkpointing", action="store_true")
    p.add_argument("--force_no_quant", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args(argv)
    for name in (
        "num_train_epochs",
        "per_device_train_batch_size",
        "gradient_accumulation_steps",
        "max_length",
        "save_steps",
        "logging_steps",
    ):
        if getattr(a, name) <= 0:
            p.error(f"{name} must be positive")
    if a.learning_rate <= 0 or not 0 <= a.warmup_ratio < 1:
        p.error("Invalid learning rate or warmup ratio")
    if any(v is not None and v <= 0 for v in (a.max_steps, a.stop_after_steps)):
        p.error("Step limits must be positive")
    return a


@torch.no_grad()
def evaluate(accelerator, model, loader):
    was_training = model.training
    model.eval()
    nll, tokens = 0.0, 0
    for batch in loader:
        outputs = model(**batch)
        sums, counts = per_example_nll(outputs.logits, batch["labels"])
        # gather_for_metrics removes repeated tail examples added by distributed loaders.
        sums, counts = accelerator.gather_for_metrics((sums, counts))
        nll += sums.sum().item()
        tokens += counts.sum().item()
    model.train(was_training)
    return perplexity(nll, tokens)


def export_adapter(accelerator, model, tokenizer, path, metadata):
    accelerator.wait_for_everyone()
    state = accelerator.get_state_dict(model)
    if accelerator.is_main_process:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        accelerator.unwrap_model(model).save_pretrained(path, state_dict=state, safe_serialization=True)
        tokenizer.save_pretrained(path)
        write_json(path / "run_manifest.json", metadata)
    accelerator.wait_for_everyone()


def save_checkpoint(accelerator, model, tokenizer, output, step, cursor, contract, metadata):
    path = Path(output) / f"checkpoint-{step}"
    # All ranks participate: RNG, optimizer and scheduler state are rank-aware.
    accelerator.save_state(str(path / "training_state"))
    export_adapter(accelerator, model, tokenizer, path / "adapter", metadata)
    if accelerator.is_main_process:
        write_json(path / "checkpoint.json", {"step": step, **cursor, "contract": contract})
    accelerator.wait_for_everyone()
    return str(path)


def main(argv=None):
    a = parse_args(argv)
    accelerator = Accelerator(
        gradient_accumulation_steps=a.gradient_accumulation_steps,
        step_scheduler_with_optimizer=False,
        dataloader_config=DataLoaderConfiguration(use_seedable_sampler=True, data_seed=a.seed),
    )
    if accelerator.distributed_type.name in ("FSDP", "DEEPSPEED"):
        raise ValueError("This implementation supports single-device/DDP; use a matching Accelerate config")
    set_seed(a.seed)
    output = Path(a.output_dir)
    if output.exists() and (output / "run.json").exists() and not a.resume_from:
        raise FileExistsError("Output already has a run. Choose a new directory or use --resume_from")
    tokenizer = load_tokenizer(a.base_model)
    placement = {"": accelerator.local_process_index} if torch.cuda.is_available() else None
    model, quantized = load_model_with_lora(a.base_model, a.lora_config, placement, a.force_no_quant)
    if a.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    counts = parameter_summary(model)
    train, validation = load_document_qa_dataset(a.train_file, a.eval_file)
    collator = DataCollatorForCausalDocQA(tokenizer, a.max_length)
    # Validate all examples before a distributed worker reaches a collective.
    for record in list(train) + list(validation):
        collator([record])
    train_loader = DataLoader(
        train, batch_size=a.per_device_train_batch_size, shuffle=True, collate_fn=collator
    )
    val_loader = DataLoader(
        validation, batch_size=a.per_device_train_batch_size, shuffle=False, collate_fn=collator
    )
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.learning_rate)
    model, optimizer, train_loader, val_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader
    )
    updates_per_epoch = math.ceil(len(train_loader) / a.gradient_accumulation_steps)
    planned_steps = a.max_steps or updates_per_epoch * a.num_train_epochs
    epochs = max(a.num_train_epochs, math.ceil(planned_steps / updates_per_epoch))
    scheduler = get_linear_schedule_with_warmup(optimizer, int(planned_steps * a.warmup_ratio), planned_steps)
    scheduler = accelerator.prepare(scheduler)
    contract = {
        "base_model": a.base_model,
        "train_sha256": file_hash(a.train_file),
        "validation_sha256": file_hash(a.eval_file),
        "lora_sha256": file_hash(a.lora_config),
        "seed": a.seed,
        "world_size": accelerator.num_processes,
        "batch_size": a.per_device_train_batch_size,
        "gradient_accumulation": a.gradient_accumulation_steps,
        "max_length": a.max_length,
        "learning_rate": a.learning_rate,
        "warmup_ratio": a.warmup_ratio,
        "planned_steps": planned_steps,
        "gradient_checkpointing": a.gradient_checkpointing,
        "quantized": quantized,
    }
    metadata = {
        "contract": contract,
        "parameters": counts,
        "hardware": str(accelerator.device),
        "torch_version": torch.__version__,
        "world_size": accelerator.num_processes,
        "effective_batch_size": a.per_device_train_batch_size
        * a.gradient_accumulation_steps
        * accelerator.num_processes,
        "updates_per_epoch": updates_per_epoch,
        "planned_steps": planned_steps,
        "train_records": len(train),
        "validation_records": len(validation),
        "status": "running",
    }
    start_epoch, skip_batches, step, seen_tokens = 0, 0, 0, 0
    if a.resume_from:
        checkpoint = Path(a.resume_from)
        cursor = json.loads((checkpoint / "checkpoint.json").read_text())
        if cursor["contract"] != contract:
            raise ValueError("Resume contract mismatch: data, model or training settings changed")
        accelerator.load_state(str(checkpoint / "training_state"))
        start_epoch, skip_batches, step = cursor["epoch"], cursor["next_batch"], cursor["step"]
        seen_tokens = cursor.get("seen_tokens", 0)
    output.mkdir(parents=True, exist_ok=True)
    if accelerator.is_main_process:
        write_json(output / "run.json", metadata)
    start_tokens = seen_tokens
    if step >= planned_steps:
        raise ValueError("The requested checkpoint already reached the planned training steps")
    started = time.perf_counter()
    baseline = evaluate(accelerator, model, val_loader)
    if accelerator.is_main_process:
        write_json(
            output / ("resumed_validation.json" if a.resume_from else "initial_validation.json"), baseline
        )
    interrupted = False
    for epoch in range(start_epoch, epochs):
        train_loader.set_epoch(epoch)
        model.train()
        epoch_loader = (
            accelerator.skip_first_batches(train_loader, skip_batches) if skip_batches else train_loader
        )
        first_batch = skip_batches
        window_nll, window_tokens = 0.0, 0
        for offset, batch in enumerate(epoch_loader):
            batch_index = first_batch + offset
            with accelerator.accumulate(model):
                result = model(**batch)
                if not torch.isfinite(result.loss):
                    raise FloatingPointError("Nonfinite training loss")
                accelerator.backward(result.loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                valid = int((batch["labels"][:, 1:] != -100).sum())
                window_nll += result.loss.detach().float().item() * valid
                window_tokens += valid
                if accelerator.sync_gradients:
                    scheduler.step()
            if not accelerator.sync_gradients:
                continue
            step += 1
            totals = accelerator.reduce(
                torch.tensor([window_nll, window_tokens], dtype=torch.float64, device=accelerator.device),
                reduction="sum",
            )
            seen_tokens += int(totals[1].item())
            event = {
                "step": step,
                "epoch": epoch,
                "answer_nll": totals[0].item() / totals[1].item(),
                "learning_rate": scheduler.get_last_lr()[0],
                "processed_supervised_tokens": seen_tokens,
            }
            if accelerator.is_main_process and step % a.logging_steps == 0:
                with (output / "metrics.jsonl").open("a") as f:
                    f.write(json.dumps(event) + "\n")
                print(json.dumps(event), flush=True)
            window_nll, window_tokens = 0.0, 0
            next_epoch, next_batch = epoch, batch_index + 1
            if next_batch >= len(train_loader):
                next_epoch, next_batch = epoch + 1, 0
            cursor = {"epoch": next_epoch, "next_batch": next_batch, "seen_tokens": seen_tokens}
            stop = a.stop_after_steps is not None and step >= a.stop_after_steps and step < planned_steps
            if step % a.save_steps == 0 or stop:
                save_checkpoint(accelerator, model, tokenizer, output, step, cursor, contract, metadata)
            if stop or step >= planned_steps:
                interrupted = stop
                break
        skip_batches = 0
        validation = evaluate(accelerator, model, val_loader)
        if accelerator.is_main_process:
            write_json(output / f"validation-epoch-{epoch}.json", validation)
        if interrupted or step >= planned_steps:
            break
    final = evaluate(accelerator, model, val_loader)
    elapsed = time.perf_counter() - started
    digest = hashlib.sha256()
    for name, parameter in accelerator.unwrap_model(model).named_parameters():
        if parameter.requires_grad:
            digest.update(name.encode())
            digest.update(parameter.detach().float().cpu().contiguous().numpy().tobytes())
    hashes = [None] * accelerator.num_processes
    if accelerator.num_processes > 1:
        torch.distributed.all_gather_object(hashes, digest.hexdigest())
    else:
        hashes = [digest.hexdigest()]
    if len(set(hashes)) != 1:
        raise RuntimeError("Distributed adapter parameters diverged between ranks")
    metadata["adapter_sha256"] = digest.hexdigest()
    metadata["distributed_adapter_consensus"] = True
    metadata["peak_cuda_memory_bytes"] = (
        torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None
    )
    metadata.update(
        status="interrupted" if interrupted else "completed",
        optimizer_steps=step,
        validation=final,
        processed_supervised_tokens=seen_tokens,
        elapsed_seconds=elapsed,
        supervised_tokens_per_second=(seen_tokens - start_tokens) / max(elapsed, 1e-6),
    )
    if not interrupted:
        export_adapter(accelerator, model, tokenizer, output / "adapter", metadata)
    if accelerator.is_main_process:
        write_json(output / "run.json", metadata)
        print(json.dumps({"status": metadata["status"], "steps": step, "validation": final}), flush=True)
    accelerator.end_training()
    return metadata


if __name__ == "__main__":
    main()
