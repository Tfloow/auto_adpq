"""Dataclasses for Auto_AdpQ configuration and quantized weights."""

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np
from pydantic import BaseModel


class AutoAdpQConfig(BaseModel):
    """Configuration for Auto_AdpQ.

    Attributes:
        group_size (int): Number of elements in a group for group-wise
            quantization. Must be between 1 and 65535 (inclusive).
        n_iters (int): Maximum number of iterations for outlier detection.
        alpha (float): Target fraction (0..1) of entries considered outliers.
        device (str): Device string (e.g. "cpu" or "cuda"). Informational.
        q_bit (int): Quantization bitwidth (e.g. 4 for 4-bit quantization).
        data_packing (bool): If True, multiple quantized values are packed
            into 32-bit integers; otherwise plain int8 arrays are used.
        symmetrical_quantization (bool): If True, use symmetric quantization
            (no zeropoint). If False, use asymmetric quantization with
            zeropoints.
        target_layers (Optional[Tuple[str, ...]]): Tuple of layer names to
            quantize. If None, all linear layers are quantized.

    Raises:
        ValueError: If `group_size` or `n_iters` are out of valid ranges.
    """

    group_size: int = 128
    n_iters: int = 100
    alpha: float = 0.08
    device: str = "cpu"
    q_bit: int = 4
    data_packing: bool = True
    symmetrical_quantization: bool = True
    target_layers: Optional[Tuple[str, ...]] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "up_proj",
        "down_proj",
        "gate_proj",
    )

    def __init__(self, **kwargs):
        """Init ADPQ config.

        Raises:
            ValueError: if the group_size is not between 1 and 65536.
            ValueError: if n_iters is not positive.
        """
        super().__init__(**kwargs)
        if self.group_size <= 0:
            raise ValueError("group_size must be a positive integer.")
        if self.n_iters <= 0:
            raise ValueError("n_iters must be a positive integer.")
        if self.group_size > 2**16:
            raise ValueError("group_size too large, must be less than 65536.")

    def __eq__(self, value) -> bool:
        """Check equality for the object.

        Args:
            value (AutoAdpQConfig): Another AutoAdpQConfig instance to compare.

        Returns:
            bool: True if all configuration attributes are equal, False otherwise.
        """
        return (
            self.group_size == value.group_size
            and self.n_iters == value.n_iters
            and self.alpha == value.alpha
            and self.device == value.device
            and self.q_bit == value.q_bit
            and self.data_packing == value.data_packing
            and self.symmetrical_quantization == value.symmetrical_quantization
        )


@dataclass(frozen=True)  # frozen=True makes it immutable (optional but safer)
class AdpQQuantizedWeights:
    """Container for AdpQ quantization outputs.

    Attributes:
        original_shape (Optional[tuple[int, ...]]): Original shape of the
            matrix passed to `AdpQ_quantize`. Used to reshape reconstructed
            output back to original shape.
        group_num (int): Number of groups after reshaping to (-1, group_size).
        scale (Union[list[float], np.ndarray]): Per-group scale values. In
            practice an array of shape (group_num, 2) where second column is
            for outliers.
        zeropoint (Optional[Union[list[float], np.ndarray]]): Per-group
            zeropoints (None when symmetric quantization is used).
        quantized_vector (Union[list[list[int]], np.ndarray]): Quantized
            integer vectors for each group (group_num x group_size).
        outlier_indices (Union[list[list[int]], np.ndarray]): Per-group list
            of outlier indices or sentinel values.

    Raises:
        ValueError: If lengths of lists do not match `group_num`.

    TODO: Currently, there is a major overhead when creating a new object
    to validate the field. Since it is used internally only, we could ditch
    the Pydantic module but would need to ensure proper dump and load function.
    """

    group_num: int
    scale: Union[list[float], np.ndarray]
    quantized_vector: Union[list[list[int]], np.ndarray]
    outlier_indices: Union[list[list[int]], np.ndarray]
    original_shape: Optional[Tuple[int, ...]] = None
    zeropoint: Optional[Union[list[float], np.ndarray]] = None

    # Optional: If you really need that check, use __post_init__
    # This runs faster than Pydantic validators because there is no pydantic overhead
    def __post_init__(self):
        """Post init.

        Check for the right size of the values.

        Raises:
            ValueError: if mismatched dimensions are found. Groups must match
                group_num.

        TODO: sometimes when loading from npz, I get zeropoint as np.array(None)
        """
        # Only run this if you suspect bugs in your generation logic
        if (
            len(self.scale) != self.group_num
            or len(self.quantized_vector) != self.group_num
            or len(self.outlier_indices) != self.group_num
        ):
            raise ValueError("Dimensions mismatch")

        if self.zeropoint is not None:
            # Meaning it is an array which is not none, can have np.array(None)
            if type(self.zeropoint) is np.ndarray:
                if self.zeropoint.ndim != 0 and len(self.zeropoint) != self.group_num:
                    raise ValueError("Dimensions mismatch for zeropoint")
            elif len(self.zeropoint) != self.group_num:
                raise ValueError("Dimensions mismatch for zeropoint")
