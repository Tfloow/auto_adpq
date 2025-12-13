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

<table>
    <thead>
        <tr>
            <th width="40%">Model Variant</th>
            <th width="30%">Quantization Method</th>
            <th width="30%">PPL (Perplexity)</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td rowspan="3"><strong>meta-llama/Llama-3.1-8B</strong></td>
            <td>Baseline</td>
            <td>4.8693</td>
        </tr>
        <tr>
            <td>BNB</td>
            <td>5.0733</td>
        </tr>
        <tr>
            <td><strong>AdpQ</strong></td>
            <td><strong>5.3671</strong></td>
        </tr>
        <tr>
            <td rowspan="5"><strong>meta-llama/Llama-3.1-8B-Instruct</strong></td>
            <td>Baseline</td>
            <td>4.9080</td>
        </tr>
        <tr>
            <td>BNB</td>
            <td>4.9993</td>
        </tr>
        <tr>
            <td><strong>AdpQ</strong></td>
            <td><strong>5.0069</strong></td>
        </tr>
        <tr>
            <td>AWQ</td>
            <td>5.0440</td>
        </tr>
         <tr>
            <td>GPTQ</td>
            <td>nan</td>
        </tr>
        <tr>
            <td rowspan="4"><strong>meta-llama/Llama-3.2-1B</strong></td>
            <td>Baseline</td>
            <td>6.5546</td>
        </tr>
        <tr>
            <td><strong>AdpQ 9%</strong></td>
            <td><strong>6.9491</strong></td>
        </tr>
        <tr>
            <td>BNB</td>
            <td>6.9971</td>
        </tr>
        <tr>
            <td><strong>AdpQ 2%</strong></td>
            <td><strong>7.0380</strong></td>
        </tr>
        <tr>
            <td rowspan="3"><strong>meta-llama/Llama-3.2-3B-Instruct</strong></td>
            <td>Baseline</td>
            <td>5.7864</td>
        </tr>
        <tr>
            <td>AWQ</td>
            <td>5.8339</td>
        </tr>
        <tr>
            <td><strong>AdpQ</strong></td>
            <td><strong>5.9040</strong></td>
        </tr>
    </tbody>
</table>

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

if __name__ == "__main__":
    from tqdm import tqdm
    model_in_collection = ["Tfloow/Llama-3.2-1B-adpq-4bit-sim",
                           "Tfloow/Meta-Llama-3.1-8B-adpq-4bit-sim",
                           "Tfloow/Llama-3.2-1B-adpq-4bit-sim-0.02",
                           "Tfloow/Llama-3.2-1B-Instruct-adpq-4bit-sim",
                           "Tfloow/Llama-3.2-3B-adpq-4bit-sim",
                           "Tfloow/Llama-3.2-3B-Instruct-adpq-4bit-sim",
                           "Tfloow/Llama-3.1-8B-Instruct-adpq-4bit-sim"]

    with open("examples/simple_quantization.py", "r") as f:
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

    for adpq_model_name in tqdm(model_in_collection):

        model_name = adpq_model_name.split("-adpq-")[0].replace("Tfloow/", "meta-llama/")

        generate_model_card(
            model_name=model_name,
            how_to_quantize="\n".join(how_to_quantize),
            repo_id=adpq_model_name
        )

