"""Tests for the Auto_AdpQ lasso outlier detection method."""

import matplotlib.pyplot as plt
import numpy as np

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

    outliers = [0 for _ in range(10)]
    outliers_llm = [0 for _ in range(10)]

    for i in range(10):
        adpq = Auto_AdpQ(
            group_size=8, alpha=0.0, n_iters=100, q_bit=4, data_packing=False
        )
        outliers[i] = adpq._optimization_function_fast(
            matrix=matrix, lambda_prime=10**i
        )
        outliers_llm[i] = adpq._optimization_function_fast_llm(
            matrix=matrix, lambda_prime=10**i
        )

    assert np.array_equal(np.array(outliers), np.array(outliers_llm)), (
        "Fast and LLM methods yield different outlier counts"
    )

    outliers = np.array(outliers)
    outliers -= 2  # We know there are 2 outliers in the matrix
    plt.plot(range(10), outliers)
    plt.grid(True)
    plt.axhline(0, color="red", linestyle="--")
    plt.show()


if __name__ == "__main__":
    test_lasso_outlier_detection()
