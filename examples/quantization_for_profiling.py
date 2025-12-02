"""Load llama 3.1 8B model and quantize it with Auto-AdpQ."""

import glob
import os

import torch
from transformers import AutoModelForCausalLM

from auto_adpq import Auto_AdpQ, AutoAdpQConfig

DUMMY_LLAMA = True
if DUMMY_LLAMA:
    model_name = "tiny-random/llama-3"  # tiny model based on llama-3 for testing
    group_size = 8
else:
    model_name = "meta-llama/Llama-3.1-8B"

    # Check if preloaded on disk
    save_path = (
        "../MasterThesis/experiments/weights/meta-llama/Meta-Llama-3.1-8B-weights"
    )
    files = glob.glob(os.path.join(save_path, "*.safetensors"))

    if len(files) == 0:
        model_name = "meta-llama/Llama-3.1-8B"
    else:
        model_name = save_path
    group_size = 128


# Setup Auto-AdpQ configuration
adpq_config = AutoAdpQConfig(
    group_size=group_size,
    n_iters=20,  # Seems quite slow otherwise
    alpha=0.08,
    device="cpu",
    q_bit=4,
    data_packing=False,
    symmetrical_quantization=True,
)

adpq = Auto_AdpQ(config=adpq_config)


model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)

print(model.dtype)

quantized_model = adpq.quantize_model_multithreaded(model, max_workers=16)

adpq.save_pretrained("quantized/")
adpq.fuse_model_from_pretrained(model, "quantized/")

with open(".env", "r") as f:
    for line in f:
        if line.startswith("HF_TOKEN="):
            hf_token = line.strip().split("=")[1]
            os.environ["HF_TOKEN"] = hf_token
            break

model.push_to_hub(f"Tfloow/{model_name.split('/')[-1]}-adpq-4bit-sim")
