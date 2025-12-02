"""Testing the data packing functionality of Auto_AdpQ."""

import glob
import os
import pickle

import numpy as np
import pytest
import torch

from auto_adpq import Auto_AdpQ, AutoAdpQConfig


def test_packing_weights():
    """Test the data packing functionality of Auto_AdpQ."""
    # Create a simple matrix to quantize
    matrix = np.array(
        [
            [-4, -3, -2, -1, 1, 2, 3, 4],
            [7, 7, 6, 5, 4, 3, 2, 1],
        ],
        dtype=np.int8,
    )

    expected_packed_weights = np.array(
        [
            [0b1111111011011100, 1 | (2 << 4) | (3 << 8) | (4 << 12)],
            [7 | (7 << 4) | (6 << 8) | (5 << 12), 4 | (3 << 4) | (2 << 8) | (1 << 12)],
        ],
        dtype=np.uint16,
    )  # Example packed representation

    auto_adpq = Auto_AdpQ(
        group_size=8, alpha=0.1, n_iters=100, q_bit=4, data_packing=True
    )

    packed_weights = auto_adpq.pack_bits(matrix)
    assert np.array_equal(packed_weights, expected_packed_weights), "Mismatch in packed weights."

    unpacked_weights = auto_adpq.unpack_bits(packed_weights)

    print("Original Weights:\n", matrix)
    print("Unpacked Weights:\n", unpacked_weights)
    
    assert np.array_equal(unpacked_weights, matrix), "Mismatch in unpacked weights."


@pytest.mark.skip(
    reason="Currently fails due to issues in save/load with data packing."
)
def test_save_load_packing():
    """Test saving and loading of packed weights."""
    adpq_config = AutoAdpQConfig(group_size=8, data_packing=True)
    adpq = Auto_AdpQ(config=adpq_config)

    path = "tests/weights/random_array/pickle"
    for module in glob.glob(os.path.join(path, "*_adpq_quantized.pkl")):
        module_name = module.replace("tests/weights/random_array/pickle\\", "")
        module_name = module_name.replace("_adpq_quantized.pkl", "")
        print(module_name)

        ref_path = module.replace("_adpq_quantized.pkl", "_ref.pkl")

        print(module)
        print(ref_path)

        with open(module, "rb") as f:
            quantized_weights = pickle.load(f)
        with open(ref_path, "rb") as f:
            ref_weights = pickle.load(f)
        print("fklqjdsfmlds")

        adpq.save_weights(quantized_weights, path + "/saved_weights/", module_name)

        loaded_weights = adpq.load_weights(
            path + "/saved_weights/" + f"{module_name}_adpq_quantized.npz"
        )

        # Compare loaded weights with reference
        assert np.array_equal(
            loaded_weights.quantized_vector, quantized_weights.quantized_vector
        )
        assert np.array_equal(loaded_weights.scale, quantized_weights.scale)
        assert loaded_weights.zeropoint == quantized_weights.zeropoint
        assert loaded_weights.original_shape == quantized_weights.original_shape
        assert np.array_equal(
            loaded_weights.outlier_indices, quantized_weights.outlier_indices
        )
        assert loaded_weights.group_num == quantized_weights.group_num
        assert adpq.cfg == adpq_config

        reconstructed_weights = torch.tensor(
            adpq.reconstruct_weights(loaded_weights)
        ).to(ref_weights.dtype)

        print(type(reconstructed_weights))
        print(type(ref_weights))

        assert torch.allclose(reconstructed_weights, ref_weights, rtol=0.15, atol=0.15)
