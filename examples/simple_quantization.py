"""Load llama 3.1 8B model and quantize it with Auto-AdpQ."""

import glob
import os

import model_card
import torch
from transformers import AutoModelForCausalLM

from auto_adpq import Auto_AdpQ, AutoAdpQConfig

with open(".env", "r") as f:
    for line in f:
        if line.startswith("HF_TOKEN="):
            hf_token = line.strip().split("=")[1]
            os.environ["HF_TOKEN"] = hf_token
            break

DUMMY_LLAMA = False
SMALL_LLAMA = False
if DUMMY_LLAMA:
    model_name = "tiny-random/llama-3"  # tiny model based on llama-3 for testing
    group_size = 8
elif SMALL_LLAMA:
    model_name = "meta-llama/Llama-3.2-1B"  # small llama-3 for testing
    group_size = 128
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

# START
model_name = "meta-llama/Llama-3.1-8B-Instruct"
group_size = 128
 
# Setup Auto-AdpQ configuration
adpq_config = AutoAdpQConfig(
    group_size=group_size,
    n_iters=250,  # Throw UserWarning if too low
    alpha=0.05,   # The higher, the better the PPL loss but higher overhead
    device="cpu",
    q_bit=4,
    data_packing=False,
    symmetrical_quantization=True,
)

user = "Tfloow"
adpq_model_name = f"{user}/{model_name.split('/')[-1]}-adpq-4bit-sim-0.02"

model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)

print(model.dtype)

# virtual quantization
quantized = Auto_AdpQ.apply_quantization(model, adpq_config, multi_threaded=16)

model.push_to_hub(adpq_model_name)
# END

# Read this file and trim between START and END tag
with open(__file__, "r") as f:
    code_lines = f.readlines()
    how_to_quantize = []
    recording = False
    for line in code_lines:
        if line.strip() == "# START":
            recording = True
            continue
        if line.strip() == "# END":
            recording = False
            continue
        if recording:
            how_to_quantize.append(line.rstrip())

model_card.generate_model_card(
    model_name=model_name,
    how_to_quantize="\n".join(how_to_quantize),
    repo_id=adpq_model_name
)
