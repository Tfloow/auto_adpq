"""Auto ADPQ module."""

from __future__ import annotations

# replace print with logging
import logging
import os
import warnings
from typing import Optional, Union

import numpy as np
import torch
from pydantic import BaseModel

debug_enabled = os.getenv("AUTO_ADPQ_DEBUG", "0") == "1"
if debug_enabled:
    logging.basicConfig(
        filename="auto_adpq_debug.log",
        filemode="a",
        format="%(asctime)s,%(msecs)03d %(name)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
        level=logging.DEBUG,
    )
    logging.debug("Debugging enabled for auto_adpq module.")
logger = logging.getLogger(__name__)


class AutoAdpQConfig(BaseModel):
    """AutoADPQ config.

    defines the basic configuration for the class auto ADPQ

    Args:
        group_size (int): the group size.
        n_iters (int): number of iterations.
        alpha (float): the percentage of outlier.
        device (str, optional): the device to use. Defaults to "cpu".
        q_bit (int, optional): the quantization bit. Defaults to 4.
        data_packing (bool, optional): whether to use data packing.
                                        Defaults to True.
        symmetrical_quantization (bool, optional): whether to use symmetrical
                                                   quantization.
                                                   Defaults to True.
    """

    group_size: int = 128
    n_iters: int = 100
    alpha: float = 0.08
    device: str = "cpu"
    q_bit: int = 4
    data_packing: bool = True
    symmetrical_quantization: bool = True

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


class AdpQQuantizedWeights(BaseModel):
    """AdpQ Quantized weights Config.

    defines the configuration for a sub-vector in AdpQ.

    Args:
        scale (float): the scale used for quantization.
        zeropoint (Optional[float]): the zero point used for quantization.
            only needed for asymmetric quantization. Defaults to None.
        quantized_vector (list[int]): the quantized sub-vector.
        scale_outlier (list[float]): scale for outlier.
        zeropoint_outlier (Optional[float], optional): zero point for outlier.
                                                Defaults to None.
        quantized_vector_outlier (list[int]): quantized sub-vector for outlier.
        outlier_indices (list[int]): indices of outliers.
    """

    group_num: int
    scale: Union[list[float], np.ndarray]
    zeropoint: Optional[Union[list[float], np.ndarray]] = None
    quantized_vector: Union[list[list[int]], np.ndarray]
    # For the outlier
    scale_outlier: Union[list[float], np.ndarray]
    zeropoint_outlier: Optional[Union[list[float], np.ndarray]] = None
    quantized_vector_outlier: Union[list[list[int]], np.ndarray]
    outlier_indices: Union[list[list[int]], np.ndarray]

    class Config:
        """Pydantic config."""

        arbitrary_types_allowed = True

    # Force each list to be same length as group_num
    def __init__(self, **data):
        """Init.

        Raises:
            ValueError: if the length of any list is not equal to group_num.
        """
        super().__init__(**data)
        if len(self.scale) != self.group_num:
            raise ValueError("Length of scale must be equal to group_num.")
        if len(self.zeropoint) != self.group_num:
            raise ValueError("Length of zeropoint must be equal to group_num.")
        if len(self.quantized_vector) != self.group_num:
            raise ValueError("Length of quantized_vector must be equal to group_num.")
        if len(self.scale_outlier) != self.group_num:
            raise ValueError("Length of scale_outlier must be equal to group_num.")
        if len(self.zeropoint_outlier) != self.group_num:
            raise ValueError("Length of zeropoint_outlier must be equal to group_num.")
        if len(self.outlier_indices) != self.group_num:
            raise ValueError("Length of outlier_indices must be equal to group_num.")


class Auto_AdpQ:
    """Auto_AdpQ.

    Runs the AdpQ algorithm.
    """

    def __init__(
        self,
        group_size: int = 128,
        alpha: float = 0.06,
        n_iters: int = 100,
        device: str = "cpu",
        q_bit: int = 4,
        data_packing: bool = True,
        symmetrical_quantization: bool = True,
        config: Optional[AutoAdpQConfig] = None,
    ):
        """Init AutoADPQ.

        Args:
            group_size (int): the group size.
            alpha (float): the percentage of outlier.
            n_iters (int): number of iterations.
            device (str, optional): the device to use. Defaults to "cpu".
            q_bit (int, optional): the quantization bit. Defaults to 4.
            data_packing (bool, optional): whether to use data packing.
                                            Defaults to True.
            symmetrical_quantization (bool, optional): whether to use symmetrical
                                                         quantization.
                                                         Defaults to True.
            config (Optional[AutoAdpQConfig], optional): Pydantic config object.
                                                        Defaults to None.
        """
        # If a Pydantic config is provided, prefer it (validated values).
        if config is not None:
            cfg = config
        else:
            # validate/create config from provided args
            cfg = AutoAdpQConfig(
                group_size=group_size,
                n_iters=n_iters,
                device=device,
                q_bit=q_bit,
                data_packing=data_packing,
                alpha=alpha,
                symmetrical_quantization=symmetrical_quantization,
            )

        # Validate group_size and set outlier index format
        self.outlier_index_format = np.int8
        if cfg.group_size > 2**8:
            warnings.warn(
                "group_size is large, will have larger memory overhead."
                " Consider using a 128 group_size for better performance.",
                UserWarning,
                stacklevel=2,
            )
            self.outlier_index_format = np.int16

        # assign validated attributes
        self.cfg = cfg
        self.group_size = cfg.group_size
        self.alpha = cfg.alpha
        self.n_iters = cfg.n_iters
        self.device = cfg.device
        self.q_bit = cfg.q_bit
        self.data_packing = cfg.data_packing
        self.symmetrical_quantization = cfg.symmetrical_quantization

    def quantize(
        self, sub_vector: Union[list[float], np.ndarray, torch.Tensor]
    ) -> tuple[np.ndarray, float, float]:
        """Quantize.

        quantize a sub-vector from a group quantization.

        Args:
            sub_vector (numpy.ndarray): the sub-vector to quantize.

        Returns:
            numpy.ndarray: the quantized sub-vector.
            float: the scale used for quantization.
            float: the zero point used for quantization.
        """
        if self.symmetrical_quantization:
            # Symmetrical quantization
            max_abs = np.max(np.abs(sub_vector))
            scale = (2 ** (self.q_bit - 1) - 1) / max_abs
            zeropoint = np.nan  # not used in symmetrical quantization
            quantized = np.round(scale * sub_vector).astype(np.int8)

            logger.debug(f"Symmetrical Quantization: max_abs={max_abs}, scale={scale}")
        else:
            scale = (2**self.q_bit - 1) / (np.max(sub_vector) - np.min(sub_vector))
            zeropoint = -np.round(np.min(sub_vector) * scale) - 2 ** (self.q_bit - 1)
            quantized = np.round(scale * sub_vector + zeropoint).astype(np.int8)

        # Store in FP16
        scale = np.float16(scale)
        zeropoint = np.float16(zeropoint)

        return quantized, scale, zeropoint

    def reconstruct_vector(
        self, quantized_vectors: list[np.ndarray], outlier_indices: list[int]
    ) -> np.ndarray:
        """Reconstruct quantized.

        Reconstruct the quantized sub-vector.

        Args:
            quantized_vectors (list[numpy.ndarray]): the quantized sub-vector.
                                        pass first non-outliers, then outliers.
            outlier_indices (list[int]): the indices of the outliers.

        TODO: since we have a sparsity of outlier around 95% depending on the
        alpha parameter, we can optimize this function further by using a CSR
        representation for the outlier vector.
        """
        if self.data_packing and self.q_bit % 2:
            raise ValueError("Data packing is only supported for even q_bit values.")

        if self.data_packing:
            amount_per_int32 = 32 // self.q_bit
            reconstructed = np.zeros(
                self.group_size // amount_per_int32 + 1, dtype=np.int32
            )
        else:
            reconstructed = np.zeros(self.group_size, dtype=np.int8)

        non_outlier_vector, outlier_vector = quantized_vectors

        if len(outlier_indices) == 0:
            # No outliers
            for i in range(self.group_size):
                if self.data_packing:
                    non_outlier_value = np.int32(non_outlier_vector[i])
                    reconstructed[i // amount_per_int32] |= non_outlier_value << (
                        (i % amount_per_int32) * self.q_bit
                    )
                else:
                    reconstructed[i] = non_outlier_vector[i]
            return reconstructed

        pos_outlier_indices = 0
        current_outlier_idx = outlier_indices[pos_outlier_indices]
        current_non_outlier_idx = 0

        for i in range(self.group_size):
            if i == current_outlier_idx:
                to_be_assigned = outlier_vector[pos_outlier_indices]
                pos_outlier_indices += 1
                if pos_outlier_indices < len(outlier_indices):
                    current_outlier_idx = outlier_indices[pos_outlier_indices]
            else:
                to_be_assigned = non_outlier_vector[current_non_outlier_idx]
                current_non_outlier_idx += 1

            if self.data_packing:
                to_be_assigned = np.int32(to_be_assigned)
                reconstructed[i // amount_per_int32] |= to_be_assigned << (
                    (i % amount_per_int32) * self.q_bit
                )
            else:
                reconstructed[i] = to_be_assigned

        return reconstructed

    def _optimization_function(
        self, matrix: np.ndarray, lambda_prime: float
    ) -> tuple[np.ndarray, float]:
        """Optimization function for Lasso outlier detection.

        Args:
            matrix (numpy.ndarray): the input matrix. shaped by (N, group_size).
            lambda_prime (float): the regularization parameter.

        Returns:
            np.ndarray: the outlier indices.
            float: the number of outliers detected.
        """
        num_groups = matrix.shape[0]
        outlier_indices = -np.ones_like(matrix, dtype=self.outlier_index_format)
        n_outlier = 0

        for i in range(num_groups):
            group_vector = matrix[i]
            adjusted_value = np.abs(group_vector) - (
                lambda_prime / np.abs(group_vector)
            )

            # Find the one that are above zero = Outliers
            outliers = adjusted_value > 0

            # Find indices where outliers == 1
            outlier_index = outliers.nonzero()[0]

            outlier_indices[i, : len(outlier_index)] = outlier_index.astype(
                self.outlier_index_format
            )
            n_outlier += len(outlier_index)

        return outlier_indices, n_outlier

    def _brent_function(
        bk: float, bk_1: float, ak: float, f_bk: float, f_bk_1: float
    ) -> float:
        """Brent's method helper function.

        Following the algorithm from Brent's method to find root of a function.
        https://en.wikipedia.org/wiki/Brent%27s_method

        Args:
            bk (float): current point.
            bk_1 (float): previous point.
            ak (float): contra point.
            f_bk (float): function value at current point.
            f_bk_1 (float): function value at previous point.

        Returns:
            float: next point.
        """
        if f_bk != f_bk_1:
            return bk - (bk - bk_1) / (f_bk - f_bk_1) * f_bk
        else:
            return (bk + ak) / 2

    def lasso_outlier_detection(
        self, matrix: Union[list[float], np.ndarray, torch.Tensor]
    ) -> tuple[np.ndarray, float]:
        r"""Lasso outlier detection.

        Detect outliers in the vector using Lasso regression.
        According to the paper, using the Adaptive LASSO method, it is possible to
        detect outliers effectively.

        The detection works based on this expression:

        .. math::

            \hat w_i = \text{sign}(w_i)\,\mathrm{ReLU}\left(|w_i| -
            \frac{\lambda'}{|w_i|}\right)

        If :math:`\hat w_i` is zero, then :math:`w_i` is considered an outlier.
        So we can tweak the formula to find the outlier indices such that:
        :math:`max(0, |w_i| - \frac{\lambda'}{|w_i|})` and we sum up all values from
        the matrix.

        Based on Brent's method, we can find the root of the function. Inspired by
        https://nickcdryan.com/2017/09/13/root-finding-algorithms-in-python-line-search-bisection-secant-newton-raphson-boydens-inverse-quadratic-interpolation-brents/

        Args:
            matrix (numpy.ndarray): the input matrix. already arrange per group
                                    (N, group_size).

        Returns:
            np.ndarray: the indices of the outliers given
            in relative group value.
                e.g., if group_size=4 and outlier indices are [[],[1,3],[]],
            float: the ratio of outliers in the matrix.
        """
        x0 = 0.0
        x1 = 1e9

        # Previous points
        _, prev_n_outlier = self._optimization_function(matrix, x0)

        # Initial point
        _, n_outlier = self._optimization_function(matrix, x1)

        ite = 0
        n_item = matrix.size
        target_outlier = self.alpha * n_item

        fx0, fx1 = prev_n_outlier - target_outlier, n_outlier - target_outlier

        logger.debug(f"Initial bracket values: fx0={fx0}, fx1={fx1}")
        assert (fx0 * fx1) < 0, (
            "Initial points do not bracket the target outlier number."
        )

        if abs(fx0) < abs(fx1):
            x0, x1 = x1, x0
            prev_n_outlier, n_outlier = n_outlier, prev_n_outlier
            fx0, fx1 = fx1, fx0

        x2, fx2 = x0, fx0

        mflag = True

        # 0.5% tolerance based on target outlier
        # tol = 0.005 * target_outlier
        tolerance = 1e-5

        d = None

        while ite < self.n_iters and abs(x1 - x0) > tolerance:
            _, fx0 = self._optimization_function(matrix, x0)
            _, fx1 = self._optimization_function(matrix, x1)
            _, fx2 = self._optimization_function(matrix, x2)

            fx0 = fx0 - target_outlier
            fx1 = fx1 - target_outlier
            fx2 = fx2 - target_outlier

            logger.debug(
                f"Iteration {ite}: x0={x0}, fx0={fx0}, x1={x1}, fx1={fx1},\
                    x2={x2}, fx2={fx2}"
            )

            if fx0 != fx2 and fx1 != fx2:
                L0 = (x0 * fx1 * fx2) / ((fx0 - fx1) * (fx0 - fx2))
                L1 = (x1 * fx0 * fx2) / ((fx1 - fx0) * (fx1 - fx2))
                L2 = (x2 * fx1 * fx0) / ((fx2 - fx0) * (fx2 - fx1))
                new = L0 + L1 + L2
            # Since the function is not continuous, we can have a case where
            # fx1 - fx0 == 0 all of sudden
            elif (fx1 - fx0) == 0 and fx1 == 0:
                new = x1
            else:
                new = x1 - ((fx1 * (x1 - x0)) / (fx1 - fx0))

            if (
                (new < ((3 * x0 + x1) / 4) or new > x1)
                or (mflag and (abs(new - x1)) >= (abs(x1 - x2) / 2))
                or (not mflag and (abs(new - x1)) >= (abs(x2 - d) / 2))
                or (mflag and (abs(x1 - x2)) < tolerance)
                or (not mflag and (abs(x2 - d)) < tolerance)
            ):
                new = (x0 + x1) / 2
                mflag = True

            else:
                mflag = False

            _, fnew = self._optimization_function(matrix, new)
            fnew = fnew - target_outlier
            d, x2 = x2, x1

            if (fx0 * fnew) < 0:
                x1 = new
            else:
                x0 = new

            if abs(fx0) < abs(fx1):
                x0, x1 = x1, x0

            ite += 1

        if ite == self.n_iters:
            warnings.warn(
                f"Lasso outlier detection did not converge within max iterations.\n\
                Check tolerance or increase n_iters. Latest step size: {abs(x1 - x0)}",
                UserWarning,
                stacklevel=2,
            )

        outlier_indices, n_outlier = self._optimization_function(matrix, new)

        return outlier_indices, n_outlier / n_item

    def AdpQ_quantize(
        self, matrix: Union[list[float], np.ndarray, torch.Tensor]
    ) -> AdpQQuantizedWeights:
        """Lasso based quantization.

        Quantize the matrix using Lasso regression.

        Args:
            matrix (numpy.ndarray): the input matrix.
        """
        original_shape = matrix.shape
        matrix = matrix.reshape((-1, self.group_size))

        outlier_indices, alpha = self.lasso_outlier_detection(matrix)
        logger.debug(f"Detected outlier ratio: {alpha}")

        # Create bitmask for non-outlier and outlier elements
        non_outlier_mask = np.ones(matrix.shape, dtype=bool)
        for group_idx in range(outlier_indices.shape[0]):
            for outlier_idx in outlier_indices[group_idx]:
                if outlier_idx != -1 and outlier_idx < self.group_size:
                    non_outlier_mask[group_idx, outlier_idx] = False
                else:
                    warnings.warn(
                        f"Outlier index exceeds group size; skipping this index for"
                        f" group {group_idx}, index {outlier_idx}",
                        UserWarning,
                        stacklevel=2,
                    )

        outlier_weight = matrix.copy()
        outlier_weight[non_outlier_mask] = 0

        non_outlier_weight = matrix.copy()
        non_outlier_weight[~non_outlier_mask] = 0

        logger.debug("Weights to separation")
        logger.debug(outlier_indices)
        logger.debug(outlier_weight)
        logger.debug(non_outlier_weight)

        # Quantize non-outlier and outlier weights separately
        num_groups = matrix.shape[0]
        scales = np.zeros((num_groups,), dtype=np.float16)
        zeropoints = (
            np.zeros((num_groups,), dtype=np.float16)
            if not self.symmetrical_quantization
            else None
        )
        # To be removed, just to pass linter
        del scales
        del zeropoints
        del original_shape

        quantized_non_outlier = []

        for group_idx in range(num_groups):
            quantized_non_outlier, scale, zeropoint = self.quantize(
                non_outlier_weight[group_idx]
            )
            quantized_outlier, scale_outlier, zeropoint_outlier = self.quantize(
                outlier_weight[group_idx]
            )
            logger.debug(f"Group {group_idx}:")
            logger.debug(quantized_non_outlier)
            logger.debug(scale, zeropoint)

        logger.debug(type(quantized_non_outlier))
        logger.debug(type(quantized_outlier))

        return AdpQQuantizedWeights(
            group_num=num_groups,
            scale=scale,
            zeropoint=zeropoint if not self.symmetrical_quantization else None,
            quantized_vector=quantized_non_outlier,
            scale_outlier=scale_outlier,
            zeropoint_outlier=zeropoint_outlier
            if not self.symmetrical_quantization
            else None,
            outlier_indices=outlier_indices,
            quantized_vector_outlier=quantized_outlier,
        )

    def quantize_weight_matrix(
        self, weight_matrix: Union[list[float], np.ndarray, torch.Tensor]
    ) -> np.ndarray:
        """Quantize weight matrix.

        Quantize the weight matrix using group quantization.

        Args:
            weight_matrix (numpy.ndarray): the weight matrix to quantize.
        """
        ite = weight_matrix.size // self.group_size
        if weight_matrix.size % self.group_size != 0:
            ite += 1  # account for remaining elements

        if weight_matrix.shape[1] % self.group_size != 0:
            raise ValueError(
                "Weight matrix columns must be divisible by group_size.\
                \n TODO: handle remaining elements."
            )

        if self.data_packing:
            amount_per_int32 = 32 // self.q_bit
            quantized_matrix = np.zeros(
                (
                    weight_matrix.shape[0],
                    weight_matrix.shape[1] // amount_per_int32 + 1,
                ),
                dtype=np.int32,
            )
        else:
            quantized_matrix = np.zeros(weight_matrix.shape, dtype=np.int8)

        for i in range(ite):
            # determine position in quantized_matrix
            m_idx = (i * self.group_size) // weight_matrix.shape[1]
            n_idx = (i * self.group_size) % weight_matrix.shape[0]

            sub_vector = weight_matrix[m_idx, n_idx : n_idx + self.group_size]
            quantized_sub_vector, _ = self.quantize(sub_vector)

            # Handle data packing index adjustment
            if self.data_packing:
                n_idx = (
                    (i * self.group_size) % weight_matrix.shape[1]
                ) // amount_per_int32

            quantized_matrix[m_idx, n_idx : n_idx + len(quantized_sub_vector)] = (
                quantized_sub_vector
            )
