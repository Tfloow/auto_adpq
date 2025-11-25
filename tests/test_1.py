from auto_adpq import Auto_AdpQ, AutoAdpQConfig
import pytest

def test_auto_adpq_initialization():
    """Test the initialization of Auto_AdpQ."""
    group_size = 4
    alpha = 0.1
    n_iters = 10
    auto_adpq = Auto_AdpQ(group_size, alpha, n_iters)
    
    assert auto_adpq.group_size == group_size
    assert auto_adpq.alpha == alpha
    assert auto_adpq.n_iters == n_iters
    assert auto_adpq.device == "cpu"
    assert auto_adpq.q_bit == 4
    assert auto_adpq.data_packing == True
    
def test_auto_adpq_quantization():
    """Test the quantization method of Auto_AdpQ."""
    import numpy as np

    group_size = 4
    alpha = 0.1
    n_iters = 10
    auto_adpq = Auto_AdpQ(group_size, alpha, n_iters)

    sub_vector = np.array([1.0, -2.0, 3.0, -4.0])
    quantized, Delta = auto_adpq.quantize(sub_vector)

    expected_Delta = 4.0 / (2 ** (auto_adpq.q_bit - 1) - 1)
    expected_quantized = np.round(sub_vector / expected_Delta)

    assert np.array_equal(quantized, expected_quantized)
    assert Delta == expected_Delta
    
def test_reconstruct_vector():
    """Test the reconstruct_vector method of Auto_AdpQ."""
    import numpy as np
    
    # catch error for odd q_bit with data_packing
    try:
        auto_adpq = Auto_AdpQ(group_size=4, alpha=0.1, n_iters=10, q_bit=3, data_packing=True)
        auto_adpq.reconstruct_vector((np.array([]), np.array([])), [])
    except ValueError as e:
        assert str(e) == "Data packing is only supported for even q_bit values."

    group_size = 5
    alpha = 0.1
    n_iters = 10
    auto_adpq = Auto_AdpQ(group_size, alpha, n_iters, q_bit=4, data_packing=False)

    non_outlier_vector = np.array([10, -20, 30, -40], dtype=np.int8)
    outlier_vector = np.array([100], dtype=np.int8)
    quantized_vectors = (non_outlier_vector, outlier_vector)
    outlier_indices = [2]

    reconstructed = auto_adpq.reconstruct_vector(quantized_vectors, outlier_indices)

    expected_reconstructed = np.array([10, -20, 100, 30, -40], dtype=np.int8)
    
    assert np.array_equal(reconstructed, expected_reconstructed)
    
    # Test with data packing
    auto_adpq_dp = Auto_AdpQ(group_size=8, alpha=0.1, n_iters=10, q_bit=4, data_packing=True)
    non_outlier_vector_dp = np.array([1, 2, 3, 4, 5, 6, 7], dtype=np.int8)
    outlier_vector_dp = np.array([100], dtype=np.int8)
    quantized_vectors_dp = (non_outlier_vector_dp, outlier_vector_dp)
    outlier_indices_dp = [5]
    reconstructed_dp = auto_adpq_dp.reconstruct_vector(quantized_vectors_dp, outlier_indices_dp)
    expected_reconstructed_dp = np.zeros(2, dtype=np.int32)
    expected_reconstructed_dp[0] = (1 << 0) | (2 << 4) | (3 << 8) | (4 << 12) | (5 << 16) | (100 << 20) | (6 << 24) | (7 << 28)
    expected_reconstructed_dp[1] = 0  # since only 7 values are packed
    
    print(reconstructed_dp)
    print(expected_reconstructed_dp)

    # print bits
    print(bin(reconstructed_dp[0]))
    print(bin(expected_reconstructed_dp[0]))
    
    assert np.array_equal(reconstructed_dp, expected_reconstructed_dp)
    
@pytest.mark.slow
def test_lasso_outlier_detection():
    """Test the lasso_outlier_detection method of Auto_AdpQ."""
    import numpy as np
    
    config = AutoAdpQConfig()
    
    auto_adpq = Auto_AdpQ(config=config)

    matrix = np.load("tests/weights/llama-8B/model.layers.0.self_attn.q_proj.weight.npy")
    
    _, outlier_ratio = auto_adpq.lasso_outlier_detection(matrix)

    # I should run the 7B test and compare to known outliers from the paper
    """expected_outlier_indices = [2, 5]  # indices of 100.0 and -200.0
    expected_outlier_ratio = 2 / 6

    assert outlier_indices == expected_outlier_indices
    assert abs(outlier_ratio - expected_outlier_ratio) < 1e-6"""
    print(f"Detected outlier ratio: {outlier_ratio}")
    
    assert outlier_ratio < config.alpha
    
if __name__ == "__main__":
    test_lasso_outlier_detection()