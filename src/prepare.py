"""Validate JSONL and split connected document groups without leakage."""

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

from src.data import context_key, validate_record
from src.utils import file_hash, write_json


def prepare(source, output, seed=42, validation_fraction=0.15, test_fraction=0.15):
    if not (
        0 < validation_fraction < 1 and 0 < test_fraction < 1 and validation_fraction + test_fraction < 1
    ):
        raise ValueError("Validation/test fractions must be positive and sum to less than one")
    accepted, rejected, seen = [], [], {}
    counts = Counter()
    for line_number, line in enumerate(Path(source).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        counts["input_records"] += 1
        try:
            record = validate_record(json.loads(line))
            key = (context_key(record["context"]), " ".join(record["question"].casefold().split()))
            answer = " ".join(record["answer"].casefold().split())
            if key in seen:
                reason = "duplicate" if seen[key] == answer else "conflicting_answer"
                rejected.append({"line": line_number, "reason": reason})
                counts[reason] += 1
                continue
            seen[key] = answer
            accepted.append(record)
        except (ValueError, TypeError) as exc:
            rejected.append({"line": line_number, "reason": "invalid", "error": str(exc)})
            counts["invalid"] += 1
    # Union document IDs and normalized contexts: either identity must stay in one split.
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    for row in accepted:
        a, b = find("context:" + context_key(row["context"])), find("id:" + row["document_id"])
        parent[b] = a
    groups = {}
    for row in accepted:
        groups.setdefault(find("context:" + context_key(row["context"])), []).append(row)
    keys = sorted(groups)
    if len(keys) < 3:
        raise ValueError("At least three independent document groups are required")
    random.Random(seed).shuffle(keys)
    n_test = max(1, round(len(keys) * test_fraction))
    n_val = max(1, round(len(keys) * validation_fraction))
    if n_val + n_test >= len(keys):
        raise ValueError("Split fractions leave no training document group")
    split_keys = {
        "test": keys[:n_test],
        "validation": keys[n_test : n_test + n_val],
        "train": keys[n_test + n_val :],
    }
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    splits, artifacts = {}, {}
    for name, selected in split_keys.items():
        rows = [row for key in selected for row in groups[key]]
        path = output / f"{name}.jsonl"
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
        splits[name] = rows
        artifacts[name] = {"records": len(rows), "documents": len(selected), "sha256": file_hash(path)}
    report = {
        "source_sha256": file_hash(source),
        "seed": seed,
        "counts": dict(counts),
        "accepted": len(accepted),
        "quarantined": len(rejected),
        "splits": artifacts,
        "document_group_overlap": 0,
        "method": "Connected document ID/context groups, seeded shuffle",
        "fingerprint": hashlib.sha256(json.dumps(artifacts, sort_keys=True).encode()).hexdigest(),
    }
    write_json(output / "manifest.json", report)
    write_json(output / "quarantine.json", rejected)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True)
    p.add_argument("--output", default="data/processed")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--validation-fraction", type=float, default=0.15)
    p.add_argument("--test-fraction", type=float, default=0.15)
    args = p.parse_args()
    print(
        json.dumps(
            prepare(args.input, args.output, args.seed, args.validation_fraction, args.test_fraction),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
