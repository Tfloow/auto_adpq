"""Testing the full flow of quantization."""

import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoModelForCausalLM

from auto_adpq import Auto_AdpQ, AutoAdpQConfig


def test_quantize_save_compare():
    """Quantize a model, save and reload, compare weights."""
    model_name = "tiny-random/llama-3"  # tiny model based on llama-3 for testing
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Instantiate Auto_AdpQ with default config
    adpq_config = AutoAdpQConfig(group_size=8)
    adpq = Auto_AdpQ(config=adpq_config)

    path = "tmp_dir/"

    # Quantize the model
    adpq.quantize_model(model)
    os.makedirs(path, exist_ok=True)

    for name in adpq.quantized_weights.keys():
        # name is like 'model.layers.0.self_attn.q_proj'
        # Compare with reference weights
        if name not in adpq.quantized_weights:
            raise AssertionError(f"Quantized weights for module {name} not found.")
        else:
            w_ref = model.get_submodule(name).weight
            w = torch.tensor(adpq.reconstruct_weights(adpq.quantized_weights[name])).to(
                w_ref.dtype
            )

            assert torch.allclose(w, w_ref, rtol=0.15, atol=0.15), (
                f"Weights for module {name} differ more than 15% after quantization."
            )
    # TODO: something is going wrong after this line
    # Save the quantized model
    adpq.save_pretrained(path)

    # Load the quantized model into a new model instance
    adpq.fuse_model_from_pretrained(model, path)

    model_ref = AutoModelForCausalLM.from_pretrained(model_name)

    tol = 0.15  # % - due to quantization error
    # Compare weights
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            fig, axs = plt.subplots(1, 3, figsize=(15, 5))

            if module.weight.dtype == torch.bfloat16:
                weight_array = module.weight.to(torch.float16).detach().cpu().numpy()
                weight_array_ref = (
                    model_ref.get_submodule(name)
                    .weight.to(torch.float16)
                    .detach()
                    .cpu()
                    .numpy()
                )
            else:
                weight_array_ref = (
                    model.get_submodule(name).weight.detach().cpu().numpy()
                )
                weight_array = (
                    module.weight.to(model.get_submodule(name).weight.dtype)
                    .detach()
                    .cpu()
                    .numpy()
                )

            fig.colorbar(
                axs[0].imshow(weight_array, cmap="viridis", aspect="auto"), ax=axs[0]
            )
            axs[0].set_title(f"Weights of {name} (Quantized)")

            fig.colorbar(
                axs[1].imshow(weight_array_ref, cmap="viridis", aspect="auto"),
                ax=axs[1],
            )
            np.save(
                f"tests/weights/random_array/{name.replace('.', '_')}_ref.npy",
                weight_array_ref,
            )
            axs[1].set_title(f"Weights of {name} (Reference)")

            diff = np.abs(weight_array / weight_array_ref)
            axs[2].set_title(f"Difference of {name}")
            fig.colorbar(axs[2].imshow(diff, cmap="viridis", aspect="auto"), ax=axs[2])

            # plt.show()
            plt.close()

            assert np.allclose(weight_array, weight_array_ref, rtol=tol, atol=tol), (
                f"Weights for module {name} differ more than {tol * 100:.2f}%"
            )
