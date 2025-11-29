# Load llama 3.1 8B model and quantize it with Auto-AdpQ
from auto_adpq import Auto_AdpQ, AutoAdpQConfig
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

weight_path = "tests/weights/llama-8B/model.layers.0.self_attn.q_proj.weight.npy"

adpq_weights = adpq.AdpQ_quantize(np.load(weight_path))

print("Quantized Weights:")
print(adpq_weights)