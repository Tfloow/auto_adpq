"""Testing the data packing functionality of Auto_AdpQ."""

import numpy as np

from auto_adpq import Auto_AdpQ


def test_packing_weights():
    """Test the data packing functionality of Auto_AdpQ."""
    # Create a simple matrix to quantize
    matrix = np.array(
        [
            [1, 2, 3, 4, 5, 6, 7, 8],
            [8, 7, 6, 5, 4, 3, 2, 1],
        ],
        dtype=np.uint8,
    )

    expected_packed_weights = np.array(
        [
            [1 | (2 << 4) | (3 << 8) | (4 << 12), 5 | (6 << 4) | (7 << 8) | (8 << 12)],
            [8 | (7 << 4) | (6 << 8) | (5 << 12), 4 | (3 << 4) | (2 << 8) | (1 << 12)],
        ],
        dtype=np.uint16,
    )  # Example packed representation

    auto_adpq = Auto_AdpQ(
        group_size=8, alpha=0.1, n_iters=100, q_bit=4, data_packing=True
    )

    packed_weights = auto_adpq.pack_bits(matrix)

    for _i in range(packed_weights.shape[0]):
        for _j in range(packed_weights.shape[1]):
            pass

    assert np.array_equal(packed_weights, expected_packed_weights)

    unpacked_weights = auto_adpq.unpack_bits(packed_weights)

    assert np.array_equal(unpacked_weights, matrix)
