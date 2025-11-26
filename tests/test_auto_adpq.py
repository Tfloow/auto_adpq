import numpy as np
import pytest

from auto_adpq import AutoAdpQConfig, Auto_AdpQ, AdpQQuantizedWeights


def test_auto_adpq_init_with_defaults_and_config():
    """Auto_AdpQ initializes correctly with defaults and with a config object."""
    # default init
    a = Auto_AdpQ()
    assert a.group_size == 128
    assert a.q_bit == 4
    assert a.data_packing is True

    # init with config object
    cfg = AutoAdpQConfig(group_size=16, q_bit=4, data_packing=False)
    a2 = Auto_AdpQ(config=cfg)
    assert a2.group_size == 16
    assert a2.data_packing is False


def test_auto_adpq_init_raises_on_invalid_group_size():
    """Providing an invalid group_size via config should raise ValueError."""
    with pytest.raises(ValueError):
        cfg = AutoAdpQConfig(group_size=0)

    # Too large group_size should also raise
    with pytest.raises(ValueError):
        cfg2 = AutoAdpQConfig(group_size=2**20)
    
    with pytest.raises(ValueError):
        cfg3 = AutoAdpQConfig(n_iters=-10)


def make_valid_adpq_quantized_weights(group_num=2, group_size=4):
    """Helper to produce a valid AdpQQuantizedWeights payload."""
    scale = [1.0] * group_num
    zeropoint = [0.0] * group_num
    quantized_vector = [[1, 2, 3, 4] for _ in range(group_num)]
    scale_outlier = [1.0] * group_num
    zeropoint_outlier = [0.0] * group_num
    quantized_vector_outlier = [[5] for _ in range(group_num)]
    outlier_indices = [[0] for _ in range(group_num)]

    return dict(
        group_num=group_num,
        scale=scale,
        zeropoint=zeropoint,
        quantized_vector=quantized_vector,
        scale_outlier=scale_outlier,
        zeropoint_outlier=zeropoint_outlier,
        quantized_vector_outlier=quantized_vector_outlier,
        outlier_indices=outlier_indices,
    )


def test_adpq_quantized_weights_accepts_valid_payload_and_rejects_bad_lengths():
    """AdpQQuantizedWeights should validate list lengths against group_num."""
    payload = make_valid_adpq_quantized_weights(group_num=3)
    obj = AdpQQuantizedWeights(**payload)
    assert obj.group_num == 3

    # Now break one list length and expect ValueError
    key = ["scale", "zeropoint", "quantized_vector", "scale_outlier", "zeropoint_outlier", "quantized_vector_outlier", "outlier_indices"]
    for k in key:
        bad = make_valid_adpq_quantized_weights(group_num=3)
        bad[k] = [1.0]  # wrong length
        with pytest.raises(ValueError):
            AdpQQuantizedWeights(**bad)
