import math
import pytest
import torch
from src.metrics import per_example_nll, perplexity, qa_scores


def test_token_weighted_nll_ignores_prompt_and_padding():
    logits = torch.zeros((2, 4, 5))
    labels = torch.tensor([[-100, -100, 1, -100], [-100, 2, 3, 4]])
    nll, counts = per_example_nll(logits, labels)
    assert counts.tolist() == [1, 3]
    result = perplexity(nll.sum().item(), counts.sum().item())
    assert result["answer_nll"] == pytest.approx(math.log(5))
    assert result["answer_perplexity"] == pytest.approx(5)


def test_qa_scores_normalize_articles_and_punctuation():
    assert qa_scores("The Base weights!", "base weights")["exact_match"] == 1
    assert qa_scores("base", "base weights")["token_f1"] == pytest.approx(2 / 3)
    assert qa_scores("", "weights")["token_f1"] == 0
    with pytest.raises(ValueError):
        perplexity(0, 0)
