"""Two CPU processes validate the actual Accelerate/DDP training path."""

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.prepare import prepare
from src.utils import write_json

ROOT = Path(__file__).resolve().parents[1]


def main():
    env = os.environ.copy()
    env.update(
        ACCELERATE_USE_CPU="true",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        TOKENIZERS_PARALLELISM="false",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
    )
    (ROOT / "runs").mkdir(exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="distributed-smoke-", dir=ROOT / "runs"))
    try:
        prepare(ROOT / "examples/document_qa.jsonl", work / "data")
        command = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node=2",
            "-m",
            "src.train",
            "--base_model",
            "examples/artifacts/tiny/base",
            "--train_file",
            str(work / "data/train.jsonl"),
            "--eval_file",
            str(work / "data/validation.jsonl"),
            "--output_dir",
            str(work / "training"),
            "--lora_config",
            "configs/lora_config_smoke.yaml",
            "--max_length",
            "256",
            "--max_steps",
            "4",
            "--gradient_accumulation_steps",
            "2",
            "--per_device_train_batch_size",
            "2",
            "--learning_rate",
            ".005",
            "--save_steps",
            "2",
        ]
        result = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True, timeout=180)
        if result.returncode:
            print(result.stdout[-5000:])
            print(result.stderr[-8000:])
            raise RuntimeError("Distributed smoke failed")
        metadata = json.loads((work / "training/run.json").read_text())
        assert metadata["world_size"] == 2 and metadata["optimizer_steps"] == 4
        assert metadata["distributed_adapter_consensus"]
        assert (work / "training/adapter/adapter_model.safetensors").exists()
        assert (work / "training/checkpoint-2/training_state").is_dir()
        report = {
            "scope": "Two-worker CPU/Gloo integration test; tiny randomly initialized GPT-2 + LoRA",
            "world_size": 2,
            "backend": "gloo",
            "optimizer_steps": metadata["optimizer_steps"],
            "effective_batch_size": metadata["effective_batch_size"],
            "updates_per_epoch": metadata["updates_per_epoch"],
            "adapter_parameters_match_across_ranks": metadata["distributed_adapter_consensus"],
            "checkpoint_and_export": "passed",
            "parameters": metadata["parameters"],
            "validation": metadata["validation"],
            "torch_version": metadata["torch_version"],
        }
        write_json(ROOT / "reports/distributed_cpu.json", report)
        print(json.dumps(report, indent=2))
    finally:
        shutil.rmtree(work)


if __name__ == "__main__":
    main()
