#!/usr/bin/env python
from __future__ import annotations

import argparse
import inspect
from pathlib import Path

import torch
import yaml
from datasets import load_dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
try:
    from trl import SFTConfig, SFTTrainer
except ImportError:
    from trl import SFTTrainer

    SFTConfig = None

from helpdesk_lora.dataset import format_training_text


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def make_training_args(cfg: dict) -> TrainingArguments:
    args_class = SFTConfig or TrainingArguments
    args = {
        "output_dir": cfg["output_dir"],
        "num_train_epochs": cfg["num_train_epochs"],
        "per_device_train_batch_size": cfg["per_device_train_batch_size"],
        "gradient_accumulation_steps": cfg["gradient_accumulation_steps"],
        "learning_rate": cfg["learning_rate"],
        "warmup_steps": cfg["warmup_steps"],
        "logging_steps": cfg["logging_steps"],
        "eval_steps": cfg["eval_steps"],
        "save_steps": cfg["save_steps"],
        "save_strategy": "steps",
        "bf16": torch.cuda.is_available(),
        "fp16": False,
        "optim": cfg.get("optim", "paged_adamw_8bit"),
        "report_to": "none",
        "seed": cfg["seed"],
    }
    if "max_steps" in cfg:
        args["max_steps"] = cfg["max_steps"]

    signature = inspect.signature(args_class.__init__)
    if "eval_strategy" in signature.parameters:
        args["eval_strategy"] = "steps"
    else:
        args["evaluation_strategy"] = "steps"

    if "max_seq_length" in signature.parameters:
        args["max_seq_length"] = cfg["max_seq_length"]
    elif "max_length" in signature.parameters:
        args["max_length"] = cfg["max_seq_length"]

    return args_class(**args)


def make_sft_trainer(
    model,
    tokenizer,
    dataset,
    lora_config: LoraConfig,
    training_args: TrainingArguments,
    cfg: dict,
) -> SFTTrainer:
    signature = inspect.signature(SFTTrainer.__init__)
    parameters = signature.parameters
    kwargs = {
        "model": model,
        "train_dataset": dataset["train"],
        "eval_dataset": dataset["eval"],
        "args": training_args,
    }

    if "tokenizer" in parameters:
        kwargs["tokenizer"] = tokenizer
    elif "processing_class" in parameters:
        kwargs["processing_class"] = tokenizer

    if "peft_config" in parameters:
        kwargs["peft_config"] = lora_config
    if "formatting_func" in parameters:
        kwargs["formatting_func"] = format_training_text
    if "max_seq_length" in parameters:
        kwargs["max_seq_length"] = cfg["max_seq_length"]

    return SFTTrainer(**kwargs)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a QLoRA adapter for helpdesk triage.")
    parser.add_argument("--config", type=Path, default=Path("configs/qlora_tinyllama.yaml"))
    args = parser.parse_args()

    cfg = load_config(args.config)

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"], use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_name"],
        quantization_config=quantization_config,
        device_map="auto",
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)

    dataset = load_dataset(
        "json",
        data_files={
            "train": cfg["dataset_train"],
            "eval": cfg["dataset_eval"],
        },
    )

    lora_config = LoraConfig(
        r=cfg["lora_r"],
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg["lora_dropout"],
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=cfg["target_modules"],
    )

    training_args = make_training_args(cfg)

    trainer = make_sft_trainer(model, tokenizer, dataset, lora_config, training_args, cfg)

    trainer.train()
    trainer.save_model(cfg["output_dir"])
    tokenizer.save_pretrained(cfg["output_dir"])
    print(f"Saved QLoRA adapter to {cfg['output_dir']}")


if __name__ == "__main__":
    main()
