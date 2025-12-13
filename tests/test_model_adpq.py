"""Testing the full flow of quantization."""

import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import pytest
from transformers import AutoModelForCausalLM

from auto_adpq import Auto_AdpQ, AutoAdpQConfig

def quantize_save_compare(multi_threaded=False):
    """Quantize a model, save and reload, compare weights."""
    model_name = "tiny-random/llama-3"  # tiny model based on llama-3 for testing
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)

    # Instantiate Auto_AdpQ with default config
    adpq_config = AutoAdpQConfig(group_size=8)
    adpq = Auto_AdpQ(config=adpq_config)

    path = "tmp_dir/"

    # Quantize the model
    if multi_threaded:
        adpq.quantize_model_multithreaded(model, max_workers=4)
    else:
        adpq.quantize_model(model)
    os.makedirs(path, exist_ok=True)
    os.makedirs("tests/weights/random_array/pickle", exist_ok=True)

    for name in adpq.quantized_weights.keys():
        # name is like 'model.layers.0.self_attn.q_proj'
        # Compare with reference weights
        if name not in adpq.quantized_weights:
            raise AssertionError(f"Quantized weights for module {name} not found.")
        else:
            w_ref = model.get_submodule(name).weight
            # save pickled weights for debugging
            import pickle

            with open(
                f"tests/weights/random_array/pickle/{name.replace('.', '_')}_adpq_quantized.pkl",
                "wb",
            ) as f:
                pickle.dump(adpq.quantized_weights[name], f)
                
            with open(
                f"tests/weights/random_array/pickle/{name.replace('.', '_')}_ref.pkl",
                "wb",
            ) as f:
                pickle.dump(w_ref, f)

            w = torch.tensor(adpq.reconstruct_weights(adpq.quantized_weights[name])).to(
                w_ref.dtype
            )

            assert torch.allclose(w, w_ref, rtol=0.15, atol=0.15), (
                f"Weights for module {name} differ more than 15% after quantization."
            )
            
            # Test the pack unpack
            tmp = adpq.quantized_weights[name].quantized_vector[0,0:4]
            packed = adpq.pack_bits(adpq.quantized_weights[name].quantized_vector)
            
            unpacked = adpq.unpack_bits(packed)
            assert np.array_equal(unpacked, adpq.quantized_weights[name].quantized_vector), (
                f"Pack/unpack failed for module {name}"
            )
    # TODO: something is going wrong after this line
    # Save the quantized model
    adpq.save_pretrained(path)

    # Load from path and compare weights
    for name in adpq.quantized_weights.keys():
        w_ref = model.get_submodule(name).weight

        w_loaded = adpq.load_weights(
            os.path.join(path, f"{name.replace('.', '_')}_adpq_quantized.npz")
        )
        w_loaded = adpq.reconstruct_weights(w_loaded)
        w_loaded = torch.tensor(w_loaded).to(w_ref.dtype)

        assert torch.allclose(w_loaded, w_ref, rtol=0.15, atol=0.15), (
            f"Weights for module {name} differ more than 15% after loading."
        )

    # Load the quantized model into a new model instance
    adpq.fuse_model_from_pretrained(model, path)

    model_ref = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16
    )

    tol = 0.15  # % - due to quantization error
    # Compare weights
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            weight_array = module.weight
            weight_array_ref = model_ref.get_submodule(name).weight
            
            weight_array_numpy = weight_array.to(torch.float32).detach().cpu().numpy()
            weight_array_ref_numpy = (
                weight_array_ref.to(torch.float32).detach().cpu().numpy()
            )
            
            """fig, axs = plt.subplots(1, 3, figsize=(15, 5))

            
            fig.colorbar(
                axs[0].imshow(weight_array_numpy, cmap="viridis", aspect="auto"),
                ax=axs[0],
            )
            axs[0].set_title(f"Weights of {name} (Quantized)")

            fig.colorbar(
                axs[1].imshow(weight_array_ref_numpy, cmap="viridis", aspect="auto"),
                ax=axs[1],
            )

            axs[1].set_title(f"Weights of {name} (Reference)")

            diff = np.abs(weight_array_numpy / weight_array_ref_numpy)
            axs[2].set_title(f"Difference of {name}")
            fig.colorbar(axs[2].imshow(diff, cmap="viridis", aspect="auto"), ax=axs[2])

            # plt.show()
            plt.close()"""
            
            np.save(
                f"tests/weights/random_array/{name.replace('.', '_')}_ref.npy",
                weight_array_ref_numpy,
            )

            assert torch.allclose(weight_array, weight_array_ref, rtol=tol, atol=tol), (
                f"Weights for module {name} differ more than {tol * 100:.2f}%"
            )
@pytest.mark.run_first
def test_quantize_save_compare_multithreaded():
    quantize_save_compare(multi_threaded=True)
    
def test_real_quantization():

    
    model_name = "tiny-random/llama-3"
    group_size = 8
    
    adpq_config = AutoAdpQConfig(
        group_size=group_size,
        n_iters=250,
        alpha=0.05,   # The higher, the better the PPL loss but higher overhead
        device="cpu",
        q_bit=4,
        data_packing=False,
        symmetrical_quantization=True,
    )

    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)

    # virtual quantization
    Auto_AdpQ.apply_quantization(model, adpq_config, multi_threaded=16)
    
    assert True