"""Load an adapter, generate bounded document answers, or merge LoRA weights."""

import argparse
import sys
import threading
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.data import prompt_token_ids


class GenerationEngine:
    def __init__(self, base_model, adapter_dir=None, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(adapter_dir or base_model, trust_remote_code=False)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model, torch_dtype=torch.float32, trust_remote_code=False
        )
        if adapter_dir:
            self.model = PeftModel.from_pretrained(self.model, adapter_dir)
        self.model.to(self.device).eval()
        self.model.config.use_cache = True
        self._lock = threading.Lock()
        self.base_model = str(base_model)
        self.adapter_dir = str(adapter_dir) if adapter_dir else None

    def generate(self, context, question, max_new_tokens=64):
        if not context.strip() or not question.strip():
            raise ValueError("Context and question must be nonempty")
        if not 1 <= max_new_tokens <= 256:
            raise ValueError("max_new_tokens must be between 1 and 256")
        limit = getattr(
            self.model.config, "max_position_embeddings", getattr(self.model.config, "n_positions", 2048)
        )
        ids = prompt_token_ids(self.tokenizer, context, question, min(limit, 2048) - max_new_tokens)
        inputs = torch.tensor([ids], dtype=torch.long, device=self.device)
        started = time.perf_counter()
        with self._lock, torch.inference_mode():
            output = self.model.generate(
                input_ids=inputs,
                attention_mask=torch.ones_like(inputs),
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        generated = output[0, len(ids) :]
        return {
            "answer": self.tokenizer.decode(generated, skip_special_tokens=True).strip(),
            "prompt_tokens": len(ids),
            "generated_tokens": len(generated),
            "latency_seconds": time.perf_counter() - started,
        }


def load_base_and_adapter(base_model, adapter_dir, for_merge=False):
    engine = GenerationEngine(base_model, adapter_dir, device="cpu" if for_merge else None)
    return engine.model, engine.tokenizer


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base_model", required=True)
    p.add_argument("--adapter_dir")
    p.add_argument("--context")
    p.add_argument("--question")
    p.add_argument("--max_new_tokens", type=int, default=64)
    p.add_argument("--merge_and_save")
    args = p.parse_args()
    if args.merge_and_save:
        if not args.adapter_dir:
            p.error("An adapter is required for merging")
        engine = GenerationEngine(args.base_model, args.adapter_dir, device="cpu")
        merged = engine.model.merge_and_unload(safe_merge=True)
        merged.save_pretrained(args.merge_and_save, safe_serialization=True)
        engine.tokenizer.save_pretrained(args.merge_and_save)
        return
    if not args.context or not args.question:
        p.error("Context and question are required")
    import json

    print(
        json.dumps(
            GenerationEngine(args.base_model, args.adapter_dir).generate(
                args.context, args.question, args.max_new_tokens
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
