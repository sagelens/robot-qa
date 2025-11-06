import os, torch
from PIL import Image
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
from peft import PeftModel

CWD = os.getcwd()

base_id = f"{CWD}/paligemma-mix-local"
adapter_base_dir = f"{CWD}/pg_seg_lora"
image_path = f"{CWD}/all_data/images/Why-Do-Cracked-Walls-Happen_jpg.rf.f22a836854e86c33a0dc947743585931.jpg"
prefix = "<image> segment cracks"

checkpoint_dirs = sorted(
    [d for d in os.listdir(adapter_base_dir) if d.startswith("checkpoint-")],
    key=lambda x: int(x.split("-")[1])
)

base = PaliGemmaForConditionalGeneration.from_pretrained(
    base_id, torch_dtype=torch.bfloat16, device_map="auto"
).eval()
processor = AutoProcessor.from_pretrained(adapter_base_dir, use_fast=False)
image = Image.open(image_path).convert("RGB")

first = checkpoint_dirs[0]
model = PeftModel.from_pretrained(base, os.path.join(adapter_base_dir, first),
                                  adapter_name="active", is_trainable=False).eval()

results = {}
for idx, ckpt in enumerate(checkpoint_dirs):
    if idx > 0:
        # Remove previous adapter and load the next one with the same name
        model.delete_adapter("active")
        model.load_adapter(os.path.join(adapter_base_dir, ckpt),
                           adapter_name="active", is_trainable=False)
        model.set_adapter("active")

    inputs = processor(text=prefix, images=image, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        gen = model.generate(
            **inputs,
            max_new_tokens=32,  # ~4 <loc> + 16 <seg> + label/eos budget
            do_sample=False,
            num_beams=1,
            eos_token_id=processor.tokenizer.eos_token_id,
        )

    continuation = gen[0][input_len:]
    decoded = processor.tokenizer.decode(continuation, skip_special_tokens=False)
    step = ckpt.split("-")[1]
    results[step] = decoded
    print(f"Step {step}: {decoded}")