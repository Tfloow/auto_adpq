# Load llama 3.1 8B model and quantize it with Auto-AdpQ
from transformers import AutoModelForCausalLM, AutoTokenizer
from auto_adpq import Auto_AdpQ, AutoAdpQConfig
import os
import glob
import gc

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

quantized_model = adpq.quantize_model_multithreaded(model, max_workers=16)

adpq.save_pretrained("quantized/")
