"""Compare LoRA ranks on validation NLL, leaving the test split untouched."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils import write_json

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base_model", default="examples/artifacts/tiny/base")
    p.add_argument("--train_file", default="data/processed/train.jsonl")
    p.add_argument("--eval_file", default="data/processed/validation.jsonl")
    p.add_argument("--lora_config", default="configs/lora_config_smoke.yaml")
    p.add_argument("--output_dir", default="runs/rank-sweep")
    p.add_argument("--ranks", default="2,4,8")
    p.add_argument("--steps", type=int, default=4)
    p.add_argument("--max_length", type=int, default=256)
    args = p.parse_args()
    ranks = [int(r) for r in args.ranks.split(",")]
    if not ranks or min(ranks) <= 0 or args.steps <= 0:
        p.error("Ranks and steps must be positive")
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load(Path(args.lora_config).read_text())
    results = []
    for rank in ranks:
        config.update(r=rank, lora_alpha=rank * 2)
        path = root / f"rank-{rank}.yaml"
        path.write_text(yaml.safe_dump(config))
        subprocess.run(
            [
                sys.executable,
                "-m",
                "src.train",
                "--base_model",
                args.base_model,
                "--train_file",
                args.train_file,
                "--eval_file",
                args.eval_file,
                "--lora_config",
                str(path),
                "--output_dir",
                str(root / f"rank-{rank}"),
                "--max_steps",
                str(args.steps),
                "--max_length",
                str(args.max_length),
                "--gradient_accumulation_steps",
                "2",
                "--learning_rate",
                ".005",
            ],
            cwd=ROOT,
            env=os.environ.copy(),
            check=True,
        )
        result = json.loads((root / f"rank-{rank}/run.json").read_text())
        results.append(
            {
                "rank": rank,
                "alpha": rank * 2,
                "parameters": result["parameters"],
                "validation": result["validation"],
                "seconds": result["elapsed_seconds"],
            }
        )
    write_json(
        root / "comparison.json",
        {"selection_split": "validation", "test_used": False, "fixed_seed": 42, "results": results},
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
