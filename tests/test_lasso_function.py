"""Testing different implementations of lasso function in Auto_AdpQ."""

import numpy as np

from auto_adpq import Auto_AdpQ, AutoAdpQConfig


def test_slow_fast():
    """Comparing different implementations of outlier detection."""
    matrix = np.load(
        "tests/weights/random_array/model_layers_0_mlp_down_proj_ref.npy"
    )

    config = AutoAdpQConfig()
    auto_adpq = Auto_AdpQ(config=config)

    for i in range(1, 100):
        lambda_prime = 10 * i
        _, slow_outliers = auto_adpq._optimization_function(
            matrix, lambda_prime=lambda_prime
        )
        fast_outliers = auto_adpq._optimization_function_fast(
            matrix, lambda_prime=lambda_prime
        )

        assert slow_outliers == fast_outliers, (
            f"Mismatch between slow and fast at lambda_prime={lambda_prime}"
        )
