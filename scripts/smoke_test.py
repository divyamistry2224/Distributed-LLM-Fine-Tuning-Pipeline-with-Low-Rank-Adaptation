"""Exercise the real CLI: prepare -> train -> checkpoint -> evaluate -> API -> merge.
All model weights and data are local. Published results measure mechanics of a
randomly initialized tiny model, not pretrained language-model performance.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from fastapi.testclient import TestClient
from peft import PeftModel
from transformers import AutoModelForCausalLM
from scripts.build_tiny import build
from src.api import create_app
from src.data import DataCollatorForCausalDocQA, read_records
from src.inference import GenerationEngine
from src.model import load_tokenizer
from src.prepare import prepare
from src.utils import write_json

ROOT = Path(__file__).resolve().parents[1]


def command(args, env):
    result = subprocess.run([sys.executable, *args], cwd=ROOT, env=env, capture_output=True, text=True)
    if result.returncode:
        print(result.stdout[-5000:])
        print(result.stderr[-5000:])
        raise RuntimeError(f"Pipeline command failed: {args[0]}")
    return result


def main():
    env = os.environ.copy()
    env.update(
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        TOKENIZERS_PARALLELISM="false",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
    )
    torch.set_num_threads(1)
    workspace = Path(tempfile.mkdtemp(prefix="lora-integration-", dir=ROOT / "runs"))
    try:
        manifest = prepare(ROOT / "examples/document_qa.jsonl", workspace / "data")
        base = build(workspace / "base", workspace / "data/train.jsonl")
        run = workspace / "training"
        common = [
            "--base_model",
            base,
            "--train_file",
            str(workspace / "data/train.jsonl"),
            "--eval_file",
            str(workspace / "data/validation.jsonl"),
            "--output_dir",
            str(run),
            "--lora_config",
            "configs/lora_config_smoke.yaml",
            "--max_length",
            "256",
            "--max_steps",
            "8",
            "--gradient_accumulation_steps",
            "2",
            "--save_steps",
            "2",
            "--learning_rate",
            ".005",
            "--per_device_train_batch_size",
            "2",
        ]
        command(["-m", "src.train", *common, "--stop_after_steps", "2"], env)
        command(["-m", "src.train", *common, "--resume_from", str(run / "checkpoint-2")], env)
        metadata = json.loads((run / "run.json").read_text())
        reference_args = list(common)
        reference_args[reference_args.index("--output_dir") + 1] = str(workspace / "reference")
        command(["-m", "src.train", *reference_args], env)
        from safetensors.torch import load_file

        resumed_weights = load_file(str(run / "adapter/adapter_model.safetensors"))
        reference_weights = load_file(str(workspace / "reference/adapter/adapter_model.safetensors"))
        resume_error = max(
            float((resumed_weights[k] - reference_weights[k]).abs().max()) for k in resumed_weights
        )
        assert resume_error < 1e-6, f"Resume differs from uninterrupted training: {resume_error}"
        assert metadata["status"] == "completed" and metadata["optimizer_steps"] == 8
        tokenizer = load_tokenizer(base)
        batch = DataCollatorForCausalDocQA(tokenizer, 256)(read_records(workspace / "data/test.jsonl")[:1])
        frozen = AutoModelForCausalLM.from_pretrained(base).eval()
        loaded = PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained(base), run / "adapter").eval()
        with torch.no_grad():
            tuned_logits = loaded(**batch).logits
            base_logits = frozen(**batch).logits
        delta = float((tuned_logits - base_logits).abs().max())
        assert delta > 0, "Training did not change adapter outputs"
        # Verify that adapter reload retains its base reference and produces reproducible logits.
        again = PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained(base), run / "adapter").eval()
        with torch.no_grad():
            reload_error = float((again(**batch).logits - tuned_logits).abs().max())
        assert reload_error < 1e-6
        # Only adapters are trainable; base weights must match the saved original exactly.
        for name, weight in frozen.state_dict().items():
            state = loaded.get_base_model().state_dict()
            key = name if name in state else name.rsplit(".", 1)[0] + ".base_layer." + name.rsplit(".", 1)[1]
            assert key in state, f"Missing base weight: {name}"
            assert torch.equal(weight, state[key]), f"Base weight changed: {name}"
        evaluation_path = workspace / "evaluation.json"
        command(
            [
                "-m",
                "src.evaluate",
                "--base_model",
                base,
                "--adapter_dir",
                str(run / "adapter"),
                "--test_file",
                str(workspace / "data/test.jsonl"),
                "--train_file",
                str(workspace / "data/train.jsonl"),
                "--validation_file",
                str(workspace / "data/validation.jsonl"),
                "--output",
                str(evaluation_path),
                "--max_new_tokens",
                "12",
            ],
            env,
        )
        evaluation = json.loads(evaluation_path.read_text())
        engine = GenerationEngine(base, str(run / "adapter"))
        with TestClient(create_app(engine=engine, api_key="smoke-test-key")) as client:
            assert client.get("/health").json()["adapter_loaded"]
            request = {
                "context": "LoRA freezes base weights.",
                "question": "What is frozen?",
                "max_new_tokens": 8,
            }
            assert client.post("/generate", json=request).status_code == 401
            response = client.post(
                "/generate", json=request, headers={"Authorization": "Bearer smoke-test-key"}
            )
            assert response.status_code == 200 and response.json()["generated_tokens"] > 0
        merged = workspace / "merged"
        command(
            [
                "-m",
                "src.inference",
                "--base_model",
                base,
                "--adapter_dir",
                str(run / "adapter"),
                "--merge_and_save",
                str(merged),
            ],
            env,
        )
        standalone = AutoModelForCausalLM.from_pretrained(merged).eval()
        with torch.no_grad():
            merge_error = float((standalone(**batch).logits - tuned_logits).abs().max())
        assert merge_error < 1e-4
        # Publish a small runnable offline fixture, not a claimed useful QA model.
        destination = ROOT / "examples/artifacts/tiny"
        destination.mkdir(parents=True, exist_ok=True)
        shutil.copytree(base, destination / "base", dirs_exist_ok=True)
        shutil.copytree(run / "adapter", destination / "adapter", dirs_exist_ok=True)
        report = {
            "scope": "Offline integration test with a randomly initialized tiny GPT-2 model",
            "seed": 42,
            "world_size": 1,
            "optimizer_steps": 8,
            "resume_verified": True,
            "resume_vs_uninterrupted_max_parameter_error": resume_error,
            "adapter_reload_max_logit_error": reload_error,
            "merge_max_logit_error": merge_error,
            "adapter_output_delta": delta,
            "base_weights_unchanged": True,
            "parameters": metadata["parameters"],
            "splits": manifest["splits"],
            "initial_validation": json.loads((run / "initial_validation.json").read_text()),
            "final_validation": metadata["validation"],
            "test": {
                key: {
                    m: evaluation[key][m]
                    for m in ["answer_nll", "answer_perplexity", "exact_match", "token_f1"]
                }
                for key in ["base", "adapted"]
            },
            "api_auth_and_generation": "passed",
            "torch_version": torch.__version__,
            "hardware": "CPU",
        }
        # The published manifest must not reference a temporary base path.
        adapter_config = destination / "adapter/adapter_config.json"
        cfg = json.loads(adapter_config.read_text())
        cfg["base_model_name_or_path"] = "examples/artifacts/tiny/base"
        write_json(adapter_config, cfg)
        (destination / "adapter/README.md").write_text(
            "# Tiny offline LoRA adapter\n\nA CPU integration fixture trained on synthetic document QA pairs using a randomly initialized tiny GPT-2 base.\nUse it to verify loading, generation, API serving and adapter merging. It is not a pretrained question-answering model.\n"
        )
        write_json(
            destination / "adapter/run_manifest.json",
            {
                "scope": report["scope"],
                "base_model": "examples/artifacts/tiny/base",
                "parameters": metadata["parameters"],
                "seed": 42,
            },
        )
        write_json(ROOT / "reports/smoke_cpu.json", report)
        print(json.dumps(report, indent=2))
    finally:
        shutil.rmtree(workspace)


if __name__ == "__main__":
    (ROOT / "runs").mkdir(exist_ok=True)
    main()
