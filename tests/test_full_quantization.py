from auto_adpq import Auto_AdpQ, AutoAdpQConfig
import pytest
import numpy as np

def test_lasso_outlier_detection():
    """Test the lasso_outlier_detection method of Auto_AdpQ."""
    # Create a matrix with known outliers
    shape = (4, 8)
    matrix = np.array([
        [1, 2, 3, 4, 5, 6, 7, 8],
        [1, 2, 1000, 4, 5, 6, 7, 8],  # Outlier at index 2
        [1, 2, 3, 4, 5, -999, 7, 8],   # Outlier at index 5
        [1, 2, 3, 4, 5, 6, 7, 8],
    ], dtype=np.float32)

    alpha = 2/32  # Set alpha to detect outliers
    auto_adpq = Auto_AdpQ(group_size=8, alpha=alpha, n_iters=20, q_bit=4, data_packing=False)

    outlier_indices, detected_alpha = auto_adpq.lasso_outlier_detection(matrix)

    expected_outlier_indices = np.array([
        [-1, -1, -1, -1, -1, -1, -1, -1],
        [2, -1, -1, -1, -1, -1, -1, -1],
        [5, -1, -1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1, -1, -1],
    ], dtype=auto_adpq.outlier_index_format)
    
    print(outlier_indices)
    print(detected_alpha)

    assert np.array_equal(outlier_indices, expected_outlier_indices)
    assert pytest.approx(detected_alpha, 0.01) == alpha  # Two outliers in total of 32 elements

# Skip this test
@pytest.mark.skip(reason="Skipping synthetic data test for now")
def test_with_synthetic_data():
    """Test Auto_AdpQ initialization with a larger group size."""
    shape = (8, 8)
    np.random.seed(42)
    matrix_to_quantize = np.random.normal(loc=0, scale=1, size=shape).astype(np.float16)
    group_size = 4
    num_group = matrix_to_quantize.size // group_size
    
    # Create random outlier in matrix
    outlier_expected = []
    num_outliers = np.random.randint(3, 6)
    outlier_indices_test = -np.ones(shape).reshape(-1, group_size)
    
    for _ in range(num_outliers):
        i = np.random.randint(0, 8)
        j = np.random.randint(0, 8)
        matrix_to_quantize[i, j] = np.random.uniform(1000, 2000)
        print(i,j)
        outlier_expected.append(( (i*shape[1] + j) // group_size ,j % group_size, matrix_to_quantize[i, j]))
    
    print(outlier_expected)
    
    alpha_synthetic = num_outliers / matrix_to_quantize.size
    
    print(f"Matrix to quantize:\n{matrix_to_quantize.reshape(-1, group_size)}")
    print("alpha used:", alpha_synthetic)
        
    adpq = Auto_AdpQ(group_size=group_size, alpha=alpha_synthetic, n_iters=10, q_bit=4, data_packing=False)
    
    quantized_weights = adpq.AdpQ_quantize(matrix_to_quantize)
    
    print(f"Quantized Weights:\n{quantized_weights}")
    
    assert False