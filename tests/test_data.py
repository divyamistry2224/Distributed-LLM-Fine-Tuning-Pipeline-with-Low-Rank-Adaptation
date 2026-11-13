import json
import torch
import pytest
from src.data import DataCollatorForCausalDocQA, build_token_ids, read_records, context_key
from src.prepare import prepare


class Tokenizer:
    pad_token_id = 2
    eos_token_id = 2

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [3 + ord(c) for c in text]}


def row(i, context=None, question=None):
    return {
        "document_id": f"doc-{i}",
        "context": context or f"Document {i}.",
        "question": question or f"Question {i}?",
        "answer": f"Answer {i}.",
    }


def test_answer_eos_is_not_masked_when_pad_equals_eos():
    batch = DataCollatorForCausalDocQA(Tokenizer(), 256)([row(1), row(2, context="X" * 80)])
    for labels, mask in zip(batch["labels"], batch["attention_mask"]):
        assert labels[mask.sum() - 1] == 2
        assert torch.all(labels[mask == 0] == -100)
        assert labels[0] == -100


def test_long_context_preserves_answer_and_rejects_impossible_budget():
    tokenizer = Tokenizer()
    prompt, full = build_token_ids(tokenizer, "x" * 1000, "Q?", "A!", 110)
    assert len(full) <= 110 and full[len(prompt) :] == [68, 36, 2]
    with pytest.raises(ValueError, match="exceed"):
        build_token_ids(tokenizer, "X", "Q" * 500, "A", 100)


def test_document_disjoint_splits_and_duplicates(tmp_path):
    records = [row(i) for i in range(12)]
    records += [records[0], row(100, context=records[0]["context"], question="Different question?")]
    records += [{"context": "", "question": "Q", "answer": "A"}]
    path = tmp_path / "input.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in records))
    report = prepare(path, tmp_path / "out")
    assert report["counts"]["duplicate"] == 1 and report["counts"]["invalid"] == 1
    assert report["accepted"] + report["quarantined"] == len(records)
    sets = [
        {context_key(r["context"]) for r in read_records(tmp_path / "out" / (name + ".jsonl"))}
        for name in ["train", "validation", "test"]
    ]
    assert not sets[0] & sets[1] and not sets[0] & sets[2] and not sets[1] & sets[2]
    assert prepare(path, tmp_path / "again")["fingerprint"] == report["fingerprint"]


def test_document_id_aliases_cannot_cross_splits(tmp_path):
    records = [row(i) for i in range(8)]
    records[1]["document_id"] = records[0]["document_id"]
    source = tmp_path / "input.jsonl"
    source.write_text("".join(json.dumps(r) + "\n" for r in records))
    prepare(source, tmp_path / "data")
    found = [
        {r["document_id"] for r in read_records(tmp_path / "data" / (split + ".jsonl"))}
        for split in ["train", "validation", "test"]
    ]
    assert sum("doc-0" in ids for ids in found) == 1


def test_empty_and_malformed_data_fail_with_line_context(tmp_path):
    p = tmp_path / "bad.jsonl"
    p.write_text("{not json}\n")
    with pytest.raises(ValueError, match=":1:"):
        read_records(p)
