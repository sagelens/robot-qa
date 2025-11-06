import os, torch
from PIL import Image
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
from peft import PeftModel

CWD = os.getcwd()

base_id = f"{CWD}/paligemma-mix-local"
adapter_base_dir = f"{CWD}/pg_seg_lora/"
adapter_dir = f"{CWD}/pg_seg_lora/checkpoint-50"

def load_model(base_model_id, adapter_base_dir, ckpt_dir):
    base = PaliGemmaForConditionalGeneration.from_pretrained(
        base_model_id, torch_dtype=torch.bfloat16, device_map="auto"
    ).eval()
    processor = AutoProcessor.from_pretrained(adapter_base_dir, use_fast=False)
    model = PeftModel.from_pretrained(base, adapter_base_dir + '/' + ckpt_dir,
                                    adapter_name="active", is_trainable=False).eval()
    return processor, model

def get_paligemma_tokens(image_path, prompt, processor, model):
    prefix = f"<image> {prompt}"
    image = Image.open(image_path).convert("RGB")
    inputs = processor(text=prefix, images=image, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        gen = model.generate(
            **inputs,
            max_new_tokens=128,  # ~4 <loc> + 16 <seg> + label/eos budget
            do_sample=False,
            num_beams=1,
            eos_token_id=processor.tokenizer.eos_token_id,
        )

    continuation = gen[0][input_len:]
    decoded = processor.tokenizer.decode(continuation, skip_special_tokens=False)
    return decoded
