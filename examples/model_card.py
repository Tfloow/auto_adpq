from huggingface_hub import ModelCard, ModelCardData


def generate_model_card(model_name: str, how_to_quantize: str, repo_id: str = None):
    """Generates and pushes a model card to the Hugging Face Hub for a quantized model.

    Args:
        model_name (str): The name or path of the base model that was quantized.
        how_to_quantize (str): A code snippet demonstrating how the model was quantized.
    """
    # 1. Define your Repo ID (Must match where you pushed the model)

    # 2. Define the Metadata (YAML tags used for filtering on HF)
    card_data = ModelCardData(
        language="en",
        license="apache-2.0",  # Change if the base model is llama2/3 or other
        base_model=model_name,  # accurately links to the original model
        tags=[
            "quantization",
            "4-bit",
            "adpq",  # Your specific quantization method
            "causal-lm",
            "pytorch",
        ],
    )

    # 3. Define the Markdown Content
    # This template is optimized for quantized models
    content = f"""---
{card_data.to_yaml()}
---

# {model_name.split("/")[-1]} - ADPQ 4-bit Quantized

This work is part of a master thesis. The library used for quantization is available at [auto-adpq](https://github.com/Tfloow/auto_adpq).

```
pip install auto-adpq
```

## Model Description
This is a compressed version of **[{model_name}](https://huggingface.co/{model_name})** created using 4-bit quantization. 

This model was quantized to reduce VRAM usage and increase inference speed while maintaining majority of the original model's performance.

## Quantization Details
* **Original Model:** {model_name}
* **Quantization Method:** ADPQ (Adaptive Quantization with data-free calibration)
* **Precision:** 4-bit
* **Simulated:** Yes

## How to Use

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "{repo_id}"

model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_id)

inputs = tokenizer("Hello, world!", return_tensors="pt").to("cuda")
output = model.generate(**inputs, max_new_tokens=20)
print(tokenizer.decode(output[0]))
```

### Performance

| model                                          |        PPL |
| :--------------------------------------------- | ---------: |
| unsloth/Meta-Llama-3.1-8B                      |     4.8693 |
| unsloth/Meta-Llama-3.1-8B-bnb-4bit             |     5.0733 |
| Tfloow/Meta-Llama-3.1-8B-weights-adpq-4bit-sim |     5.3671 |
| ----                                           |       ---- |
| unsloth/Meta-Llama-3.2-1B                      |     6.5546 |
| unsloth/Meta-Llama-3.2-1B-bnb-4bit             |     6.9971 |
| unsloth/Meta-Llama-3.2-1B-adpq                 | **6.9491** |

### How was the model quantized?

```python

import torch
from transformers import AutoModelForCausalLM

from auto_adpq import Auto_AdpQ, AutoAdpQConfig

model_name = "{model_name}"

{how_to_quantize}
```
    """

    card = ModelCard(content)
    card.push_to_hub(repo_id)
