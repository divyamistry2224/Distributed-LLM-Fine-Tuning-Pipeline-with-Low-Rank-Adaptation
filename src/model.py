"""Explicit single-device placement, LoRA-only updates and optional CUDA QLoRA."""

from pathlib import Path
import torch
import yaml
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_lora_config(path):
    config = yaml.safe_load(Path(path).read_text())
    if not isinstance(config, dict) or not isinstance(config.get("r"), int) or config["r"] <= 0:
        raise ValueError("LoRA rank must be a positive integer")
    if config.get("lora_alpha", 0) <= 0 or not 0 <= config.get("lora_dropout", 0) < 1:
        raise ValueError("Invalid LoRA alpha/dropout")
    if config.get("bias", "none") != "none":
        raise ValueError("This pipeline preserves the base model: bias must be none")
    if not config.get("target_modules"):
        raise ValueError("At least one target module is required")
    return config


def bnb_4bit_available():
    if not torch.cuda.is_available():
        return False
    try:
        import bitsandbytes  # noqa: F401

        return True
    except ImportError:
        return False


def load_tokenizer(base_model):
    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True, trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer must provide PAD or EOS")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def build_bnb_config(cfg):
    from transformers import BitsAndBytesConfig

    quant = cfg.get("quantization", {})
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=quant.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_use_double_quant=quant.get("bnb_4bit_use_double_quant", True),
        bnb_4bit_compute_dtype=dtype,
    )


def load_base_model(base_model, lora_cfg_path, device_map=None, force_no_quant=False):
    cfg = load_lora_config(lora_cfg_path)
    quantize = cfg.get("quantization", {}).get("load_in_4bit", False) and not force_no_quant
    if quantize and not bnb_4bit_available():
        raise RuntimeError("4-bit training requires CUDA and bitsandbytes; use a nonquantized config on CPU")
    options = {"trust_remote_code": False}
    if quantize:
        if device_map is None:
            device_map = {"": torch.cuda.current_device()}
        options.update(quantization_config=build_bnb_config(cfg), device_map=device_map)
    else:
        # DDP/Accelerate owns placement. Never shard one replica with device_map="auto".
        options["torch_dtype"] = torch.float32
        if device_map is not None:
            options["device_map"] = device_map
    model = AutoModelForCausalLM.from_pretrained(base_model, **options)
    if quantize:
        model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False
    return model, quantize


def attach_lora(model, lora_cfg_path):
    cfg = load_lora_config(lora_cfg_path)
    targets = list(cfg["target_modules"])
    if cfg.get("adapt_mlp"):
        targets += list(cfg.get("mlp_target_modules", []))
    names = [name for name, _ in model.named_modules()]
    missing = [target for target in targets if not any(name.endswith(target) for name in names)]
    if missing:
        raise ValueError(f"LoRA targets do not exist in this architecture: {missing}")
    model = get_peft_model(
        model,
        LoraConfig(
            r=cfg["r"],
            lora_alpha=cfg["lora_alpha"],
            lora_dropout=cfg.get("lora_dropout", 0),
            target_modules=targets,
            bias="none",
            task_type="CAUSAL_LM",
            fan_in_fan_out=cfg.get("fan_in_fan_out", False),
        ),
    )
    return model


def load_model_with_lora(base_model, lora_cfg_path, device_map=None, force_no_quant=False):
    model, quantized = load_base_model(base_model, lora_cfg_path, device_map, force_no_quant)
    return attach_lora(model, lora_cfg_path), quantized


def parameter_summary(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable, "trainable_fraction": trainable / total}
