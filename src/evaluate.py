"""Compare frozen base and adapted models on a document-disjoint test set."""

import argparse
import json
import torch
from torch.utils.data import DataLoader
from src.data import DataCollatorForCausalDocQA, context_key, read_records
from src.inference import GenerationEngine
from src.metrics import per_example_nll, perplexity, qa_scores
from src.utils import file_hash, write_json


def evaluate_model(engine, records, max_length=256, max_new_tokens=32):
    collate = DataCollatorForCausalDocQA(engine.tokenizer, max_length)
    loader = DataLoader(records, batch_size=2, collate_fn=collate)
    nll, tokens = 0.0, 0
    with torch.inference_mode():
        for batch in loader:
            batch = {k: v.to(engine.device) for k, v in batch.items()}
            sums, counts = per_example_nll(engine.model(**batch).logits, batch["labels"])
            nll += sums.sum().item()
            tokens += counts.sum().item()
    outputs = []
    for row in records:
        result = engine.generate(row["context"], row["question"], max_new_tokens)
        outputs.append(
            {
                "document_id": row["document_id"],
                "question": row["question"],
                "reference": row["answer"],
                **result,
                **qa_scores(result["answer"], row["answer"]),
            }
        )
    return {
        **perplexity(nll, tokens),
        "exact_match": sum(r["exact_match"] for r in outputs) / len(outputs),
        "token_f1": sum(r["token_f1"] for r in outputs) / len(outputs),
        "records": len(outputs),
        "predictions": outputs,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base_model", required=True)
    p.add_argument("--adapter_dir", required=True)
    p.add_argument("--test_file", required=True)
    p.add_argument("--train_file", required=True)
    p.add_argument("--validation_file", required=True)
    p.add_argument("--output", default="runs/evaluation.json")
    p.add_argument("--max_length", type=int, default=256)
    p.add_argument("--max_new_tokens", type=int, default=32)
    args = p.parse_args()
    test = read_records(args.test_file)
    seen = read_records(args.train_file) + read_records(args.validation_file)
    if {context_key(r["context"]) for r in test} & {context_key(r["context"]) for r in seen} or {
        r["document_id"] for r in test
    } & {r["document_id"] for r in seen}:
        raise ValueError("Test document leakage detected")
    base = GenerationEngine(args.base_model)
    baseline = evaluate_model(base, test, args.max_length, args.max_new_tokens)
    del base
    adapted = GenerationEngine(args.base_model, args.adapter_dir)
    tuned = evaluate_model(adapted, test, args.max_length, args.max_new_tokens)
    report = {
        "base_model": args.base_model,
        "test_sha256": file_hash(args.test_file),
        "adapter_dir": args.adapter_dir,
        "base": baseline,
        "adapted": tuned,
        "generation": {"do_sample": False, "max_new_tokens": args.max_new_tokens},
        "scope": "Document-disjoint test records; answer-token NLL and normalized QA scores",
    }
    write_json(args.output, report)
    print(
        json.dumps(
            {
                k: {m: v[m] for m in ["answer_nll", "answer_perplexity", "exact_match", "token_f1"]}
                for k, v in [("base", baseline), ("adapted", tuned)]
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
