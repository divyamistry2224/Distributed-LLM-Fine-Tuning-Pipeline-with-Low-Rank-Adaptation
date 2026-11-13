"""Document-QA validation and answer-only causal-LM batches."""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

PROMPT_PREFIX = "Answer the question using the provided document.\n\n### Document:\n"
PROMPT_TEMPLATE = PROMPT_PREFIX + "{context}\n\n### Question:\n{question}\n\n### Answer:\n"
FIELDS = ("context", "question", "answer")


def _suffix_for(question):
    return f"\n\n### Question:\n{question}\n\n### Answer:\n"


def validate_record(record):
    if not isinstance(record, dict):
        raise ValueError("Record must be an object")
    for field in FIELDS:
        if not isinstance(record.get(field), str) or not record[field].strip():
            raise ValueError(f"{field} must be a nonempty string")
    result = {field: record[field].strip() for field in FIELDS}
    result["document_id"] = str(record.get("document_id") or context_key(result["context"]))
    return result


def context_key(context):
    normalized = " ".join(context.casefold().split())
    return hashlib.sha256(normalized.encode()).hexdigest()


def read_records(path):
    records = []
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            records.append(validate_record(json.loads(line)))
        except (ValueError, TypeError) as exc:
            raise ValueError(f"{path}:{number}: {exc}") from exc
    if not records:
        raise ValueError(f"{path}: no valid records")
    return records


def build_token_ids(tokenizer, context, question, answer, max_length, min_answer_tokens=1):
    """Reserve the complete question, answer and EOS; truncate document context only.

    Oversized fixed content is rejected rather than silently dropping the answer.
    Segments use the same boundaries as the inference engine.
    """
    del min_answer_tokens

    def encode(text):
        return tokenizer(text, add_special_tokens=False)["input_ids"]

    prefix, suffix, target = encode(PROMPT_PREFIX), encode(_suffix_for(question)), encode(answer)
    if not target:
        raise ValueError("Answer must produce at least one token")
    if tokenizer.eos_token_id is not None:
        target = target + [tokenizer.eos_token_id]
    budget = max_length - len(prefix) - len(suffix) - len(target)
    if budget < 0:
        raise ValueError("Question and answer exceed max_length; increase the token budget")
    prompt = prefix + encode(context)[:budget] + suffix
    return prompt, prompt + target


def prompt_token_ids(tokenizer, context, question, max_prompt_length):
    def encode(text):
        return tokenizer(text, add_special_tokens=False)["input_ids"]

    prefix, suffix = encode(PROMPT_PREFIX), encode(_suffix_for(question))
    budget = max_prompt_length - len(prefix) - len(suffix)
    if budget < 0:
        raise ValueError("Question exceeds the generation token budget")
    return prefix + encode(context)[:budget] + suffix


@dataclass
class DataCollatorForCausalDocQA:
    tokenizer: object
    max_length: int = 2048

    def __call__(self, batch):
        import torch

        if not batch:
            raise ValueError("Empty batch")
        encoded = [
            build_token_ids(self.tokenizer, e["context"], e["question"], e["answer"], self.max_length)
            for e in batch
        ]
        length = min(self.max_length, max(len(full) for _, full in encoded))
        ids, masks, labels = [], [], []
        if self.tokenizer.pad_token_id is None:
            raise ValueError("Tokenizer must define a pad token")
        for prompt, full in encoded:
            padding = length - len(full)
            ids.append(full + [self.tokenizer.pad_token_id] * padding)
            masks.append([1] * len(full) + [0] * padding)
            # Mask by position, never token ID: EOS may intentionally equal PAD.
            label = [-100] * len(prompt) + full[len(prompt) :] + [-100] * padding
            if not any(token != -100 for token in label[1:]):
                raise ValueError("Batch contains a record with no supervised next token")
            labels.append(label)
        return {
            "input_ids": torch.tensor(ids),
            "attention_mask": torch.tensor(masks),
            "labels": torch.tensor(labels),
        }


def format_example(example):
    return validate_record(example)


def load_document_qa_dataset(train_file, eval_file=None, val_split=0.15):
    """Separate files are required for a document-disjoint evaluation split."""
    from datasets import Dataset

    train = read_records(train_file)
    if eval_file:
        evaluation = read_records(eval_file)
    else:
        import random

        groups = sorted({context_key(r["context"]) for r in train})
        if len(groups) < 2:
            raise ValueError("At least two distinct documents are required for a split")
        random.Random(42).shuffle(groups)
        held_out = set(groups[: max(1, min(len(groups) - 1, round(len(groups) * val_split)))])
        evaluation = [r for r in train if context_key(r["context"]) in held_out]
        train = [r for r in train if context_key(r["context"]) not in held_out]
    overlap = {context_key(r["context"]) for r in train} & {context_key(r["context"]) for r in evaluation}
    overlap_ids = {r["document_id"] for r in train} & {r["document_id"] for r in evaluation}
    if overlap or overlap_ids:
        raise ValueError("Train/evaluation document leakage detected")
    return Dataset.from_list(train), Dataset.from_list(evaluation)
