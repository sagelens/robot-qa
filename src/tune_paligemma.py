import os
import json
import math
import argparse
from dataclasses import dataclass
from typing import List, Dict, Any

import torch
from torch.utils.data import Dataset

from PIL import Image

from transformers import (
    AutoProcessor,
    PaliGemmaForConditionalGeneration,
    Trainer,
    TrainingArguments,
    DefaultDataCollator,
)

from peft import LoraConfig, get_peft_model

CWD = os.getcwd()
# -----------------------
# JSONL dataset
# -----------------------
class JsonlSegDataset(Dataset):
    def __init__(self, jsonl_path: str, images_root: str, processor, max_new_tokens: int = 512):
        self.items = []
        self.images_root = images_root
        self.processor = processor
        self.max_new_tokens = max_new_tokens
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                ex = json.loads(line)
                # Expect fields: image (path or filename), prefix (prompt), suffix (target tokens)
                self.items.append({
                    "image": ex["image"],
                    "prefix": ex["prefix"],
                    "suffix": ex["suffix"],
                })

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        ex = self.items[idx]
        img_path = ex["image"]
        if not os.path.isabs(img_path):
            img_path = os.path.join(self.images_root, ex["image"])
        image = Image.open(img_path).convert("RGB")
        prefix = ex["prefix"].strip()
        suffix = ex["suffix"].strip()

        # 1) Add the image token to the prompt
        prompt = "<image> " + prefix

        model_inputs = self.processor(
            text=prompt,
            images=image,
            suffix=suffix,              # lets the processor create labels aligned with the suffix
            return_tensors="pt",
            padding=False,
        )
        input_ids = model_inputs["input_ids"][0]
        attention_mask = model_inputs["attention_mask"][0]
        pixel_values = model_inputs["pixel_values"][0]
        labels = model_inputs["labels"][0]  # already masked over the prompt
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "labels": labels,
        }
# -----------------------
# Collator using dynamic padding
# -----------------------
@dataclass
class VlmDataCollator:
    processor: Any
    label_pad_token_id: int = -100

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Separate out pixel_values and sequences
        pixel_values = torch.stack([f["pixel_values"] for f in features], dim=0)

        input_ids = [f["input_ids"] for f in features]
        attention_mask = [f["attention_mask"] for f in features]
        labels = [f["labels"] for f in features]

        batch = self.processor.tokenizer.pad(
            {"input_ids": input_ids, "attention_mask": attention_mask},
            padding=True,
            return_tensors="pt",
        )

        # Pad labels to same length
        max_len = batch["input_ids"].shape[1]
        padded_labels = []
        for l in labels:
            pad_len = max_len - l.shape[0]
            if pad_len > 0:
                l = torch.cat([l, torch.full((pad_len,), self.label_pad_token_id, dtype=l.dtype)])
            padded_labels.append(l)
        batch["labels"] = torch.stack(padded_labels, dim=0)
        batch["pixel_values"] = pixel_values
        return batch

# -----------------------
# Utility: sanity-check special tokens exist
# -----------------------
def assert_paligemma_tokens(tokenizer):
    # Check a couple of representative tokens are known to the tokenizer
    test_tokens = ["<loc0000>", "<loc1023>", "<seg000>", "<seg127>"]
    missing = []
    for t in test_tokens:
        tid = tokenizer.convert_tokens_to_ids(t)
        if tid is None or tid == tokenizer.unk_token_id:
            missing.append(t)
    if missing:
        raise ValueError(
            f"Tokenizer is missing expected segmentation/location tokens: {missing}. "
            "Make sure you are using an official PaliGemma checkpoint and processor."
        )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_jsonl", type=str, required=True)
    parser.add_argument("--val_jsonl", type=str, default=None)
    parser.add_argument("--images_root", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="pg_seg_lora")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--use_qlora", action="store_true")
    parser.add_argument("--save_steps", type=int, default=25)
    parser.add_argument("--eval_steps", type=int, default=25)
    parser.add_argument("--logging_steps", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    dtype = torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else torch.float32)

    # Load model and processor
    processor = AutoProcessor.from_pretrained(f'{CWD}/paligemma-mix-local', use_fast=False)
    assert_paligemma_tokens(processor.tokenizer)

    # Load base model
    quant_cfg = None
    if args.use_qlora:
        from transformers import BitsAndBytesConfig
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if args.bf16 else torch.float16,
            bnb_4bit_use_double_quant=True,
        )

    model = PaliGemmaForConditionalGeneration.from_pretrained(
        f'{CWD}/paligemma-mix-local',
        torch_dtype=dtype,
        quantization_config=quant_cfg,
        device_map="auto",
    )
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()       # make inputs require grad so checkpointing can backprop [HF]
    else:
        def make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)
        model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)  # fallback hook

    model.config.use_cache = False               # cache must be off during training [HF]


    # LoRA config (target Gemma decoder projection modules)
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Datasets
    train_ds = JsonlSegDataset(args.train_jsonl, args.images_root, processor)
    eval_ds = JsonlSegDataset(args.val_jsonl, args.images_root, processor) if args.val_jsonl else None

    data_collator = VlmDataCollator(processor=processor)

    # Training args
    train_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        max_steps=args.max_steps,
        logging_steps=args.logging_steps,
        eval_strategy="steps" if eval_ds is not None else "no",
        eval_steps=args.eval_steps if eval_ds is not None else None,
        save_steps=args.save_steps,
        save_total_limit=15,
        bf16=args.bf16,
        fp16=args.fp16 and not args.bf16,
        dataloader_num_workers=4,
        report_to="none",
        remove_unused_columns=False,
        gradient_checkpointing=True,
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=data_collator,
    )

    trainer.train()
    # Save LoRA adapter
    trainer.model.save_pretrained(args.output_dir)
    # Save processor for inference
    processor.save_pretrained(args.output_dir)

    print("Training complete. Adapter and processor saved to:", args.output_dir)

if __name__ == "__main__":
    main()