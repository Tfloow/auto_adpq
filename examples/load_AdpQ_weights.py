# This will load the patched weights (after running `quantization_for_profiling.py`)

from transformers import AutoModelForCausalLM, AutoTokenizer
from auto_adpq import Auto_AdpQ, AutoAdpQConfig
import os
import glob
import gc
import torch
import numpy as np


# Setup Auto-AdpQ configuration
adpq_config = AutoAdpQConfig(
    group_size = 128,
    n_iters = 20, # Seems quite slow otherwise
    alpha = 0.08,
    device = "cpu",
    q_bit = 4,
    data_packing = True,
    symmetrical_quantization = True
)

adpq = Auto_AdpQ(config=adpq_config)

save_path = "../MasterThesis/experiments/weights/meta-llama/Meta-Llama-3.1-8B-weights" 
# Check if the model is present
files = glob.glob(os.path.join(save_path, "*.safetensors"))

print(os.getcwd())
print(files)

if len(files) == 0:
    model_name = "meta-llama/Llama-3.1-8B"
else:
    model_name = save_path

model = AutoModelForCausalLM.from_pretrained(model_name, device_map="cpu", torch_dtype="auto")

# in quantized/ npz files should be present
# Load each npz file and update the model weights

path = "quantized/"
npz_files = glob.glob(os.path.join(path, "*.npz"))

for name, module in model.named_modules():
    if isinstance(module, torch.nn.Linear):
        file_name = name.replace(".", "_")
        npz_path = os.path.join(path, f"{file_name+file_name}_adpq_quantized.npz") # I f up the naming
        
        if any(file_name in f for f in npz_files):
            print(f"Loading weights from {npz_path} into module {name}")
            adpq_weight = adpq.load_weights(npz_path)
            new_weight = adpq.reconstruct_weights(adpq_weight)
            
            if new_weight.shape != module.weight.shape:
                    if new_weight.T.shape == module.weight.shape:
                        new_weight = new_weight.T
                    else:
                        print(f"Error: Shape mismatch for {name}. Model: {module.weight.shape}, File: {new_weight.shape}")
                        continue
                    
            # Convert to torch tensor first
            new_weight = torch.tensor(new_weight)
                    
            module.weight.data = new_weight.to(
                    device=module.weight.device, 
                    dtype=module.weight.dtype
            )

print("All weights loaded successfully.")
print("Saving locally...")
model.save_pretrained("F:\\adpq-model")

print("Pushing the model to the hub...")
# Save the model online on huggingface
model.push_to_hub("Tfloow/llama-3.1-8B-adpq-4bit-sim")