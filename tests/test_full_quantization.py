"""Tests for full quantization process of Auto_AdpQ."""

import glob
import os

# import matplotlib.pyplot as plt
import numpy as np
import pytest

from auto_adpq import Auto_AdpQ


def test_lasso_outlier_detection():
    """Test the lasso_outlier_detection method of Auto_AdpQ."""
    # Create a matrix with known outliers
    matrix = np.array(
        [
            [1, 2, 3, 4, 5, 6, 7, 8],
            [1, 2, 1000, 4, 5, 6, 7, 8],  # Outlier at index 2
            [1, 2, 3, 4, 5, -999, 7, 8],  # Outlier at index 5
            [1, 2, 3, 4, 5, 6, 7, 8],
        ],
        dtype=np.float32,
    )

    alpha = 2 / 32  # Set alpha to detect outliers
    auto_adpq = Auto_AdpQ(
        group_size=8, alpha=alpha, n_iters=100, q_bit=4, data_packing=False
    )

    outlier_indices, detected_alpha = auto_adpq.lasso_outlier_detection(matrix)

    expected_outlier_indices = np.array(
        [
            [-1, -1, -1, -1, -1, -1, -1, -1],
            [2, -1, -1, -1, -1, -1, -1, -1],
            [5, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1],
        ],
        dtype=auto_adpq.outlier_index_format,
    )

    assert np.array_equal(outlier_indices, expected_outlier_indices)
    assert (
        pytest.approx(detected_alpha, 0.01) == alpha
    )  # Two outliers in total of 32 elements


def test_reconstruction():
    """Test the full quantization and reconstruction process of Auto_AdpQ."""
    matrix = np.array(
        [
            [1, 2, 3, 4, 5, 6, 7, 8],
            [1, 2, 1000, 4, 5, 6, 7, 8],  # Outlier at index 2
            [1, 2, 3, 4, 5, -999, 7, 8],  # Outlier at index 5
            [1, 2, 3, 4, 5, 6, 7, 8],
        ],
        dtype=np.float32,
    )

    alpha = 2 / 32  # Set alpha to detect outliers
    auto_adpq = Auto_AdpQ(
        group_size=4, alpha=alpha, n_iters=100, q_bit=4, data_packing=False
    )

    quantized_weights = auto_adpq.AdpQ_quantize(matrix)
    reconstructed = auto_adpq.reconstruct_weights(quantized_weights)

    # Tolerance of 15 % due to quantization error
    assert np.allclose(reconstructed, matrix, rtol=0.15, atol=0.15)


def test_random():
    """Testing random array based on tiny-random/llama-3."""
    path = "tests/weights/random_array/"
    arrs = glob.glob(os.path.join(path, "*.npy"))

    for arr_path in arrs:
        arr = np.load(arr_path)
        group_size = 8
        alpha = 0.01

        adpq = Auto_AdpQ(
            group_size=group_size,
            alpha=alpha,
            n_iters=70,
            q_bit=4,
            data_packing=False,
        )

        quantized_weights = adpq.AdpQ_quantize(arr)
        reconstructed = adpq.reconstruct_weights(quantized_weights)

        """fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        axs[0].set_title(f"Original Weights from {os.path.basename(arr_path)}")
        fig.colorbar(axs[0].imshow(arr, cmap="viridis", aspect="auto"), ax=axs[0])

        axs[1].set_title(f"Reconstructed Weights from {os.path.basename(arr_path)}")
        fig.colorbar(
            axs[1].imshow(reconstructed, cmap="viridis", aspect="auto"), ax=axs[1]
        )
        axs[2].set_title(f"Difference from {os.path.basename(arr_path)}")
        diff = arr - reconstructed
        fig.colorbar(axs[2].imshow(diff, cmap="viridis", aspect="auto"), ax=axs[2])
        plt.show()"""

        tol = 0.15  # 15 % tolerance due to quantization error
        assert np.allclose(reconstructed, arr, rtol=tol, atol=tol)


@pytest.mark.skip(reason="Need to update the library to support torch.bfloat16")
def test_random_bfloat16():
    """Testing random array with bfloat16 weights."""
    path = "tests/weights/random_array_bfloat16/"
    arrs = glob.glob(os.path.join(path, "*.npy"))

    for arr_path in arrs:
        arr = np.load(arr_path).astype(np.float32)
        group_size = 8
        alpha = 0.01

        adpq = Auto_AdpQ(
            group_size=group_size,
            alpha=alpha,
            n_iters=50,
            q_bit=4,
            data_packing=False,
        )

        quantized_weights = adpq.AdpQ_quantize(arr)
        reconstructed = adpq.reconstruct_weights(quantized_weights)

        tol = 0.15  # 15 % tolerance due to quantization error
        assert np.allclose(reconstructed, arr, rtol=tol, atol=tol)


# Skip this test
# @pytest.mark.skip(reason="Skipping synthetic data test for now")
def test_with_synthetic_data():
    """Test Auto_AdpQ initialization with a larger group size."""
    shape = (8, 8)
    np.random.seed(42)
    matrix_to_quantize = np.random.normal(loc=0, scale=1, size=shape).astype(np.float16)
    group_size = 4
    matrix_to_quantize.size // group_size

    # Create random outlier in matrix
    outlier_expected = []
    num_outliers = np.random.randint(3, 6)
    -np.ones(shape).reshape(-1, group_size)

    for _ in range(num_outliers):
        i = np.random.randint(0, 8)
        j = np.random.randint(0, 8)
        matrix_to_quantize[i, j] = np.random.uniform(1000, 2000)
        outlier_expected.append(
            ((i * shape[1] + j) // group_size, j % group_size, matrix_to_quantize[i, j])
        )

    alpha_synthetic = num_outliers / matrix_to_quantize.size

    adpq = Auto_AdpQ(
        group_size=group_size,
        alpha=alpha_synthetic,
        n_iters=50,
        q_bit=4,
        data_packing=False,
    )

    quantized_weights = adpq.AdpQ_quantize(matrix_to_quantize)

    # Expected outlier indices
    np.array(quantized_weights.outlier_indices).reshape(-1, group_size)

    assert True


def test_save_and_load_quantized_weights():
    """Test saving and loading of AdpQQuantizedWeights."""
    from auto_adpq import AdpQQuantizedWeights, Auto_AdpQ

    # Create a sample AdpQQuantizedWeights object
    original = AdpQQuantizedWeights(
        group_num=2,
        scale=np.array([0.1, 0.2]),
        zeropoint=np.array([0.0, 0.0]),
        quantized_vector=np.array([[1, 2, 3, 4], [5, 6, 7, 7]]),
        outlier_indices=np.array([[0], [1]]),
    )

    adpq = Auto_AdpQ()
    adpq.save_weights(original, "tests/out_test/")

    # Load the weights back
    loaded = adpq.load_weights("tests/out_test/weights_adpq_quantized.npz")

    # Verify that the loaded object matches the original
    assert np.array_equal(loaded.group_num, original.group_num)
    assert np.array_equal(loaded.scale, original.scale)
    assert np.array_equal(loaded.zeropoint, original.zeropoint)
    assert np.array_equal(loaded.quantized_vector, original.quantized_vector)
    assert np.array_equal(loaded.outlier_indices, original.outlier_indices)
