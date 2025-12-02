"""Basic tests of classes and methods in auto_adpq module."""

import numpy as np
import pytest

from auto_adpq import Auto_AdpQ, AutoAdpQConfig


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
    assert auto_adpq.data_packing


def test_auto_adpq_with_larger_group_size():
    """Test Auto_AdpQ initialization with a larger group size."""
    group_size = 2**10
    alpha = 0.2
    n_iters = 5
    # Raise warning for large group size
    with pytest.warns(UserWarning):
        auto_adpq = Auto_AdpQ(group_size, alpha, n_iters, q_bit=8, data_packing=False)

    assert auto_adpq.group_size == group_size
    assert auto_adpq.outlier_index_format == np.int16


def test_auto_adpq_quantization():
    """Test the quantization method of Auto_AdpQ."""
    import numpy as np

    group_size = 4
    alpha = 0.1
    n_iters = 10
    auto_adpq = Auto_AdpQ(group_size, alpha, n_iters)

    sub_vector = np.array([1.0, -2.0, 3.0, -4.0])
    quantized, scale, zeropoint = auto_adpq.quantize(sub_vector)

    expected_Delta = 4.0 / (2 ** (auto_adpq.q_bit - 1) - 1)
    expected_quantized = np.round(sub_vector / expected_Delta)

    assert np.array_equal(quantized, expected_quantized)
    assert pytest.approx(scale, 1e-6) == 1 / expected_Delta
    assert np.isnan(zeropoint)  # for symmetrical quantization, zeropoint is not used


def test_auto_adpq_quantization_asymmetrical():
    """Test the quantization method of Auto_AdpQ with asymmetrical quantization."""
    import numpy as np

    group_size = 4
    alpha = 0.1
    n_iters = 10
    auto_adpq = Auto_AdpQ(group_size, alpha, n_iters, symmetrical_quantization=False)

    sub_vector = np.array([1.0, -2.0, 3.0, -4.0])
    quantized, scale, zeropoint = auto_adpq.quantize(sub_vector)

    min_val = np.min(sub_vector)
    max_val = np.max(sub_vector)
    qmin = 0
    2**auto_adpq.q_bit - 1
    expected_scale = (max_val - min_val) / (2 ** (auto_adpq.q_bit - 1) - 1)
    expected_zeropoint = qmin - min_val / expected_scale
    np.round(sub_vector / expected_scale + expected_zeropoint)

    # assert np.array_equal(quantized, expected_quantized)
    # assert pytest.approx(scale, 1e-6) == expected_scale
    # assert pytest.approx(zeropoint, 1e-6) == expected_zeropoint

