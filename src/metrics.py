"""Token-weighted likelihood and normalized extractive QA scores."""

import math
import re
import string
from collections import Counter


def normalize_answer(text):
    text = text.lower().translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def qa_scores(prediction, reference):
    pred, gold = normalize_answer(prediction), normalize_answer(reference)
    p, g = pred.split(), gold.split()
    common = sum((Counter(p) & Counter(g)).values())
    if not p or not g:
        f1 = float(p == g)
    elif not common:
        f1 = 0.0
    else:
        precision, recall = common / len(p), common / len(g)
        f1 = 2 * precision * recall / (precision + recall)
    return {"exact_match": float(pred == gold), "token_f1": f1}


def perplexity(nll, tokens):
    if not tokens:
        raise ValueError("No supervised tokens were evaluated")
    mean = nll / tokens
    return {
        "answer_nll": mean,
        "answer_perplexity": math.exp(min(mean, 80)),
        "supervised_tokens": int(tokens),
        "perplexity_capped": mean > 80,
    }


def per_example_nll(logits, labels):
    import torch.nn.functional as F

    shift_logits = logits[:, :-1, :].float()
    shift_labels = labels[:, 1:]
    loss = F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.shape[-1]),
        shift_labels.reshape(-1),
        ignore_index=-100,
        reduction="none",
    )
    loss = loss.reshape(shift_labels.shape)
    return loss.sum(1), (shift_labels != -100).sum(1)
