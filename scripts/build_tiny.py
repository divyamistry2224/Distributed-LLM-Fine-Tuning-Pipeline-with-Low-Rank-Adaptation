"""Create a small offline integration fixture; not a pretrained QA model."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from tokenizers import ByteLevelBPETokenizer
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast
from src.data import PROMPT_TEMPLATE
from src.utils import set_seed


def build(destination="examples/artifacts/tiny/base", train_file="data/processed/train.jsonl"):
    set_seed(42)
    torch.set_num_threads(1)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    records = [json.loads(line) for line in Path(train_file).read_text().splitlines()]
    corpus = destination / "tokenizer_corpus.txt"
    corpus.write_text(
        "\n".join(
            PROMPT_TEMPLATE.format(**{k: row[k] for k in ("context", "question")}) + row["answer"]
            for row in records
        )
    )
    backend = ByteLevelBPETokenizer()
    backend.train(
        files=[str(corpus)],
        vocab_size=512,
        min_frequency=1,
        special_tokens=["<pad>", "<bos>", "<eos>", "<unk>"],
    )
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend._tokenizer,
        pad_token="<pad>",
        bos_token="<bos>",
        eos_token="<eos>",
        unk_token="<unk>",
        model_max_length=256,
    )
    tokenizer.save_pretrained(destination)
    corpus.unlink()
    config = GPT2Config(
        vocab_size=len(tokenizer),
        n_positions=256,
        n_embd=48,
        n_layer=2,
        n_head=2,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
    )
    GPT2LMHeadModel(config).save_pretrained(destination, safe_serialization=True)
    return str(destination)


if __name__ == "__main__":
    print(build())
