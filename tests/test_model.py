import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel
from src.model import attach_lora, load_lora_config, parameter_summary


def test_only_lora_parameters_receive_gradients():
    model = attach_lora(
        GPT2LMHeadModel(GPT2Config(vocab_size=32, n_embd=16, n_layer=1, n_head=2, n_positions=32)),
        "configs/lora_config_smoke.yaml",
    )
    summary = parameter_summary(model)
    assert 0 < summary["trainable"] < summary["total"]
    trainable = [name for name, p in model.named_parameters() if p.requires_grad]
    assert trainable and all("lora_" in name for name in trainable)
    ids = torch.randint(0, 32, (1, 12))
    model(input_ids=ids, labels=ids).loss.backward()
    assert all(p.grad is None for name, p in model.named_parameters() if "lora_" not in name)


def test_missing_targets_are_clear_errors():
    with pytest.raises(ValueError, match="targets do not exist"):
        attach_lora(
            GPT2LMHeadModel(GPT2Config(vocab_size=32, n_embd=16, n_layer=1, n_head=2)),
            "configs/lora_config.yaml",
        )


def test_invalid_configuration(tmp_path):
    path = tmp_path / "lora.yaml"
    path.write_text("r: 0\n")
    with pytest.raises(ValueError, match="positive"):
        load_lora_config(path)
