"""Auto ADPQ module."""

from __future__ import annotations

import logging
import os
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

# replace print with logging
from glob import glob
from typing import Optional, Union

# Dependency imports
import numpy as np
import torch
from tqdm import tqdm

# Local imports
from .class_format import AdpQQuantizedWeights, AutoAdpQConfig

warnings.filterwarnings("always", category=UserWarning)


# Save info and warning logs to console
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)

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
    logging.info("Debugging enabled for auto_adpq module.")

logger = logging.getLogger(__name__)


class Auto_AdpQ:
    """Adaptive Post-Training Quantization driver.

    This class implements the end-to-end AdpQ flow: outlier detection,
    separate quantization of non-outlier and outlier values, and packaging of
    the quantized representation into :class:`AdpQQuantizedWeights`.
    """

    linear_target_layers = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "up_proj",
        "down_proj",
        "gate_proj",
    )

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
        """Initialize Auto_AdpQ.

        Args:
            group_size (int): Number of elements per group.
            alpha (float): Target fraction of outliers.
            n_iters (int): Maximum iterations for outlier detection.
            device (str): Device string (informational).
            q_bit (int): Quantization bitwidth.
            data_packing (bool): Whether to pack quantized values into ints.
            symmetrical_quantization (bool): Use symmetric quantization if
                True; asymmetric if False.
            config (Optional[AutoAdpQConfig]): A validated config object. If
                provided, individual kwargs are ignored.

        Raises:
            ValueError: If provided config contains invalid values for
                `group_size` or `n_iters` (validated in AutoAdpQConfig).
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

        self.quantized_weights = {}

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
        """Quantize a 1-D sub-vector.

        The function supports symmetric and asymmetric quantization. For
        symmetric quantization, `zeropoint` is not used and will be set to
        ``np.nan`` before conversion to ``np.float16``.

        Args:
            sub_vector (Union[list[float], np.ndarray, torch.Tensor]): 1-D
                numeric array-like containing values to quantize.

        Returns:
            Tuple[np.ndarray, float, float]: ``(quantized, scale, zeropoint)``
                where ``quantized`` is an ``np.int8`` array and ``scale``/
                ``zeropoint`` are returned as ``np.float16`` values.

        Raises:
            ValueError: If input vector leads to invalid arithmetic (e.g.
                division by zero for a zero vector in symmetric mode).
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

        if scale == 0 or np.isnan(scale) or np.isinf(scale):
            raise ValueError(
                f"Invalid scale computed during quantization.\n\
                Scale={scale}, sub_vector={sub_vector} max={np.max(sub_vector)}, \
                min={np.min(sub_vector)}"
            )

        # Store in FP16
        scale = np.float16(scale)
        zeropoint = np.float16(zeropoint)

        return quantized, scale, zeropoint

    def pack_bits(self, quantized_weights: np.ndarray) -> np.ndarray:
        """Pack quantized weights vector.

        Args:
            quantized_weights (np.ndarray): the quantized weights must be of
                size (M,N) typical matrix size.

        Returns:
            np.ndarray: the bit-packed quantized weights.
        """
        if self.q_bit % 2 != 0:
            raise ValueError("Data packing is only supported for even q_bit values.")

        weights_per_int16 = 16 // self.q_bit
        mask = (1 << self.q_bit) - 1

        bit_pack_array = np.zeros(
            (
                quantized_weights.shape[0],
                quantized_weights.shape[1] // weights_per_int16,
            ),
            dtype=np.uint16,
        )

        for row in range(quantized_weights.shape[0]):
            for i in range(bit_pack_array.shape[1]):
                packed_value = np.uint16(0)
                for j in range(weights_per_int16):
                    q_value = quantized_weights[row, i * weights_per_int16 + j] & mask
                    # To save space, the quantized weights are saved in int8,
                    # must be upcasted
                    q_value = np.uint16(q_value)
                    packed_value |= q_value << (j * self.q_bit)
                bit_pack_array[row, i] = packed_value

        return bit_pack_array

    def unpack_bits(self, packed_weights: np.ndarray) -> np.ndarray:
        """Unpack bit-packed quantized weights.

        Args:
            packed_weights (np.ndarray): the bit-packed quantized weights.

        Returns:
            np.ndarray: the unpacked quantized weights.
        """
        if self.q_bit % 2 != 0:
            raise ValueError("Data packing is only supported for even q_bit values.")

        weights_per_int16 = 16 // self.q_bit
        mask = (1 << self.q_bit) - 1

        unpacked_array = np.zeros(
            (packed_weights.shape[0], packed_weights.shape[1] * weights_per_int16),
            dtype=np.int8,
        )

        for row in range(packed_weights.shape[0]):
            for i in range(packed_weights.shape[1]):
                packed_value = packed_weights[row, i]
                for j in range(weights_per_int16):
                    tmp = packed_value >> (j * self.q_bit)
                    q_value = tmp & mask
                    # Handle sign
                    if tmp & (0b1 << (self.q_bit - 1)):
                        q_value = -np.int8((~q_value + 1) & mask)
                    unpacked_array[row, i * weights_per_int16 + j] = q_value

        return unpacked_array

    def _indices_to_bitmask_of_outliers(self, outlier_indices: list[int]) -> np.ndarray:
        """Convert per-group outlier index lists to a boolean mask.

        Args:
            outlier_indices (list[list[int]]): A sequence of per-group lists of
                outlier indices. Negative indices are treated as sentinels and
                stop the per-group scan.

        Returns:
            np.ndarray: Boolean array with shape ``(group_num, group_size)``
                where True indicates an outlier position.
        """
        bitmask = np.zeros_like(outlier_indices, dtype=bool)

        for m, group in enumerate(outlier_indices):
            for idx in group:
                if idx >= 0:
                    bitmask[m, idx] = True
                else:
                    break  # Stop at the first -1
        return bitmask

    def reconstruct_weights(
        self, adpq_quantized_weights: AdpQQuantizedWeights
    ) -> np.ndarray:
        """Reconstruct the full matrix from an AdpQQuantizedWeights object.

        Args:
            adpq_quantized_weights (AdpQQuantizedWeights): Container produced
                by :meth:`AdpQ_quantize` that includes scales, zeropoints,
                quantized vectors and outlier indices.

        Returns:
            np.ndarray: Reconstructed matrix with dtype ``np.float16`` and
                shape matching ``original_shape`` from the provided object.
        """
        if self.data_packing and self.q_bit % 2:
            raise ValueError("Data packing is only supported for even q_bit values.")

        # Unpack the quantized vectors and outlier indices
        original_shape = adpq_quantized_weights.original_shape
        quantized_vectors = adpq_quantized_weights.quantized_vector
        outlier_indices = adpq_quantized_weights.outlier_indices
        scale = adpq_quantized_weights.scale
        zeropoint = adpq_quantized_weights.zeropoint

        bitmask = self._indices_to_bitmask_of_outliers(outlier_indices)

        non_outlier = quantized_vectors.copy()
        non_outlier[bitmask] = 0

        outlier = quantized_vectors.copy()
        outlier[~bitmask] = 0

        # Replace 0 values in scale by 1 to avoid division by zero
        scale[scale == 0] = 1.0
        logger.debug(f"Reconstructing weights with scale: {scale}")
        if self.symmetrical_quantization:
            del zeropoint  # Not used in symmetrical quantization
            # Symmetrical quantization
            reconstructed = non_outlier.astype(np.float16) / scale[:, 0][:, np.newaxis]
            reconstructed += outlier.astype(np.float16) / scale[:, 1][:, np.newaxis]
        else:
            reconstructed = (non_outlier.astype(np.float16) - zeropoint) / scale[:, 0][
                :, np.newaxis
            ]
            reconstructed += (outlier.astype(np.float16) - zeropoint) / scale[:, 1][
                :, np.newaxis
            ]
        reconstructed = reconstructed.reshape(original_shape)
        return reconstructed

    def save_weights(
        self,
        adpq_quantized_weights: AdpQQuantizedWeights,
        filepath: str,
        weight_name: str = "weights",
    ):
        """Save the AdpQQuantizedWeights to a file.

        Args:
            adpq_quantized_weights (AdpQQuantizedWeights): The quantized weights
                to save.
            weight_name (str): The name of the weight matrix.
            filepath (str): The path to the file where the weights will be saved.

        TODO: Data packing fails at the moment!
        """
        quantized_vectors = adpq_quantized_weights.quantized_vector.reshape(
            adpq_quantized_weights.original_shape
        )
        quantized_vectors = (
            self.pack_bits(quantized_vectors)
            if self.data_packing
            else quantized_vectors
        )
        np.savez(
            filepath + f"{weight_name}_adpq_quantized.npz",
            quantized_vectors=quantized_vectors,
            scale=adpq_quantized_weights.scale,
            zeropoint=adpq_quantized_weights.zeropoint,
            outlier_indices=adpq_quantized_weights.outlier_indices,
            group_num=adpq_quantized_weights.group_num,
            ADPQ_config=self.cfg.model_dump(),
        )

    def load_weights(
        self,
        filepath: str,
    ) -> AdpQQuantizedWeights:
        """Load the AdpQQuantizedWeights from a file.

        Args:
            weight_name (str): The name of the weight matrix.
            filepath (str): The path to the file where the weights are saved.

        Returns:
            AdpQQuantizedWeights: The loaded quantized weights.
        """
        data = np.load(filepath, allow_pickle=True)
        quantized_vectors = data["quantized_vectors"]
        if self.data_packing:
            quantized_vectors = self.unpack_bits(quantized_vectors)
        group_num = data["group_num"].item()
        scale = data["scale"]
        zeropoint = data["zeropoint"]
        outlier_indices = data["outlier_indices"]

        # Load config
        ADPQ_config = data["ADPQ_config"].item()
        self.cfg = AutoAdpQConfig.model_validate(ADPQ_config)

        return AdpQQuantizedWeights(
            original_shape=quantized_vectors.shape,
            group_num=group_num,
            scale=scale,
            zeropoint=zeropoint,
            quantized_vector=quantized_vectors.reshape((group_num, -1)),
            outlier_indices=outlier_indices,
        )

    def _optimization_function(
        self, matrix: np.ndarray, lambda_prime: float
    ) -> tuple[np.ndarray, float]:
        """Evaluate outlier selection for a given regularization parameter.

        Args:
            matrix (np.ndarray): 2-D array shaped (num_groups, group_size).
            lambda_prime (float): Regularization parameter controlling the
                threshold for outlier selection.

        Returns:
            Tuple[np.ndarray, int]: (outlier_indices, n_outlier) where
                outlier_indices is an integer array of shape
                (num_groups, group_size) using -1 as sentinel for unused
                positions, and n_outlier is the total count of outliers.
        """
        num_groups = matrix.shape[0]
        outlier_indices = -np.ones_like(matrix, dtype=self.outlier_index_format)
        n_outlier = 0

        for i in range(num_groups):
            group_vector = matrix[i]
            # np.abs(group_vector) is sometimes = 0 TODO: check why
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

    def _optimization_function_fast(
        self, matrix: np.ndarray, lambda_prime: float
    ) -> int:
        """Like ``_optimization_function`` but only return the amount of outliers.

        Args:
            matrix (np.ndarray): 2-D array shaped (num_groups, group_size).
            lambda_prime (float): Regularization parameter controlling the
                threshold for outlier selection.

        Returns:
            float: n_outlier, the total count of outliers.
        """
        abs_matrix = np.abs(matrix)
        # Avoid division by zero
        abs_matrix = np.where(abs_matrix == 0, np.finfo(float).eps, abs_matrix)
        adjusted = abs_matrix - (lambda_prime / abs_matrix)
        return np.count_nonzero(adjusted > 0)

    def _brent_function(
        bk: float, bk_1: float, ak: float, f_bk: float, f_bk_1: float
    ) -> float:
        """Compute the next point using Brent-like interpolation.

        Args:
            bk (float): Current point.
            bk_1 (float): Previous point.
            ak (float): Contra point.
            f_bk (float): Function value at current point.
            f_bk_1 (float): Function value at previous point.

        Returns:
            float: Proposed next point computed by interpolation.
        """
        if f_bk != f_bk_1:
            return bk - (bk - bk_1) / (f_bk - f_bk_1) * f_bk
        else:
            return (bk + ak) / 2

    def lasso_outlier_detection(
        self, matrix: Union[list[float], np.ndarray, torch.Tensor]
    ) -> tuple[np.ndarray, float]:
        """Detect outliers using an adaptive LASSO-inspired method.

        The method searches for a regularization parameter that produces a
        target fraction of outliers (``alpha``) using a Brent-like root
        finding procedure. The selection criterion follows::

            hat_w_i = sign(w_i) * ReLU(|w_i| - lambda' / |w_i|)

        Args:
            matrix (Union[list, np.ndarray, torch.Tensor]): 2-D array shaped
                (num_groups, group_size) containing values to analyze.

        Returns:
            Tuple[np.ndarray, float]: ``(outlier_indices, outlier_ratio)``
                where ``outlier_indices`` is an integer array listing per-group
                outlier positions and ``outlier_ratio`` is the fraction of
                entries detected as outliers.
        """
        x0 = 0.0
        x1 = 1e7

        # Previous points
        prev_n_outlier = self._optimization_function_fast(matrix, x0)

        # Initial point
        n_outlier = self._optimization_function_fast(matrix, x1)

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
        tolerance_outliers = 1e-3 * target_outlier

        d = None

        while ite < self.n_iters and abs(x1 - x0) > tolerance:
            fx0 = self._optimization_function_fast(matrix, x0)
            fx1 = self._optimization_function_fast(matrix, x1)
            fx2 = self._optimization_function_fast(matrix, x2)

            fx0 = fx0 - target_outlier
            fx1 = fx1 - target_outlier
            fx2 = fx2 - target_outlier

            # Check if any function value is within tolerance
            if np.isclose(abs(fx0), 0, atol=tolerance_outliers):
                new = x0
                break
            if np.isclose(abs(fx1), 0, atol=tolerance_outliers):
                new = x1
                break
            if np.isclose(abs(fx2), 0, atol=tolerance_outliers):
                new = x2
                break

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
            elif (fx1 - fx0) == 0:
                new = x0
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

            fnew = self._optimization_function_fast(matrix, new)
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
        """Quantize a matrix using the AdpQ (LASSO-based) flow.

        Args:
            matrix (Union[list, np.ndarray, torch.Tensor]): Input weight
                matrix. The method reshapes the input to ``(-1, group_size)``
                and processes each group independently.

        Returns:
            AdpQQuantizedWeights: Container with quantized values, scales,
                optional zeropoints and outlier indices.
        """
        original_shape = matrix.shape
        matrix = matrix.reshape((-1, self.group_size))

        outlier_indices, alpha = self.lasso_outlier_detection(matrix)
        logger.debug(f"Detected outlier ratio: {alpha}")

        # Create bitmask for non-outlier and outlier elements
        outlier_mask = self._indices_to_bitmask_of_outliers(outlier_indices)

        outlier_weight = matrix.copy()
        outlier_weight[~outlier_mask] = 0

        non_outlier_weight = matrix.copy()
        non_outlier_weight[outlier_mask] = 0

        num_groups = matrix.shape[0]

        # Initialize storage for quantized values
        scales = np.empty((num_groups, 2), dtype=np.float16)
        zeropoints = (
            np.empty((num_groups, 2), dtype=np.float16)
            if not self.symmetrical_quantization
            else None
        )
        quantized_values = np.empty_like(non_outlier_weight, dtype=np.int8)

        for group_idx in range(num_groups):
            logger.debug(f"Group {group_idx}:")
            quantized_non_outlier, scale, zeropoint = self.quantize(
                non_outlier_weight[group_idx]
            )

            if outlier_indices[group_idx, 0] == -1:
                # No outliers in this group
                quantized_outlier = np.zeros(
                    (quantized_values.shape[1],), dtype=np.int8
                )
                scale_outlier = np.float16(0.0)
                zeropoint_outlier = (
                    np.float16(0.0) if not self.symmetrical_quantization else None
                )
            else:
                logger.debug(f"Quantizing outliers for group {group_idx}:")
                quantized_outlier, scale_outlier, zeropoint_outlier = self.quantize(
                    outlier_weight[group_idx]
                )
            # Save results
            scales[group_idx, 0] = scale
            scales[group_idx, 1] = scale_outlier
            if not self.symmetrical_quantization:
                zeropoints[group_idx, 0] = zeropoint
                zeropoints[group_idx, 1] = zeropoint_outlier
            quantized_values[group_idx, :] = quantized_non_outlier + quantized_outlier

        return AdpQQuantizedWeights(
            original_shape=original_shape,
            group_num=num_groups,
            scale=scales,
            zeropoint=zeropoints,
            quantized_vector=quantized_values,
            outlier_indices=outlier_indices,
        )

    def quantize_reconstruct(
        self, matrix: Union[list[float], np.ndarray, torch.Tensor]
    ) -> np.ndarray:
        """Quantize and reconstruct a matrix using AdpQ.

        Args:
            matrix (Union[list, np.ndarray, torch.Tensor]): Input weight
                matrix. The method reshapes the input to ``(-1, group_size)``
                and processes each group independently.

        Returns:
            np.ndarray: Reconstructed matrix after quantization.
        """
        adpq_quantized_weights = self.AdpQ_quantize(matrix)
        reconstructed_matrix = self.reconstruct_weights(adpq_quantized_weights)
        return reconstructed_matrix

    def quantize_model_multithreaded(
        self, model: torch.nn.Module, max_workers: int = 4
    ):
        """Quantize valid linear layers using a thread pool.

        Args:
            model: The PyTorch model.
            max_workers: Limit threads to avoid OOM (Out of Memory).
                         Set to 4-8 for desktop, higher for servers.
        """
        warnings.warn(
            "Deprecated: Use `apply_quantization` if you want to\
            quantize a full model easily.",
            DeprecationWarning,
            stacklevel=2,
        )

        target_suffixes = (
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "up_proj",
            "down_proj",
            "gate_proj",
        )
        future_to_layer = {}

        logger.info(f"Starting threaded quantization with {max_workers} workers...")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Iterate through model, find layers, extract data, submit to pool
            for name, module in model.named_modules():
                if isinstance(module, torch.nn.Linear) and name.endswith(
                    target_suffixes
                ):
                    logger.info(f"Extracting weights for: {name}")

                    if module.weight.dtype == torch.bfloat16:
                        # Throw UserWarning if max value exceeds float16 range
                        max_val = torch.max(torch.abs(module.weight)).item()
                        if max_val > 65504:
                            warnings.warn(
                                f"Max weight value {max_val} exceeds float16 range. "
                                "This may lead to overflow during conversion.",
                                UserWarning,
                                stacklevel=2,
                            )
                        weight_array = (
                            module.weight.to(torch.float16).detach().cpu().numpy()
                        )
                    else:
                        weight_array = module.weight.detach().cpu().numpy()

                    future = executor.submit(self.AdpQ_quantize, weight_array)
                    future_to_layer[future] = name

            # 2. COLLECTION PHASE
            for future in as_completed(future_to_layer):
                layer_name = future_to_layer[future]
                try:
                    result = future.result()
                    self.quantized_weights[layer_name] = result
                    logger.info(f"✅ Finished: {layer_name}")

                except Exception as exc:
                    logger.error(f"❌ Exception in layer {layer_name}: {exc}")

        logger.info("Quantization complete.")

    def save_pretrained(self, save_directory: str):
        """Save the quantized model in Hugging Face format.

        Args:
            save_directory (str): The directory where the model will be saved.
        """
        os.makedirs(save_directory, exist_ok=True)

        for name, quantized_weights in self.quantized_weights.items():
            logger.info(f"Saving quantized weights for layer: {name}")
            self.save_weights(
                quantized_weights,
                filepath=save_directory,
                weight_name=name.replace(".", "_"),
            )

    def fuse_model_from_pretrained(self, model: torch.nn.Module, load_directory: str):
        """Load the quantized model from Hugging Face format.

        Args:
            model (torch.nn.Module): The PyTorch model to load the weights into.
            load_directory (str): The directory where the model in ADPQ format is saved.
        """
        npz_files = glob(os.path.join(load_directory, "*.npz"))

        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                file_name = name.replace(".", "_")
                npz_path = os.path.join(
                    load_directory, f"{file_name}_adpq_quantized.npz"
                )  # I f up the naming

                if any(file_name in f for f in npz_files):
                    adpq_weight = self.load_weights(npz_path)
                    new_weight = self.reconstruct_weights(adpq_weight)

                    if new_weight.shape != module.weight.shape:
                        if new_weight.T.shape == module.weight.shape:
                            new_weight = new_weight.T
                        else:
                            continue

                    # Convert to torch tensor first
                    new_weight = torch.tensor(new_weight).to(torch.bfloat16)

                    module.weight.data = new_weight

    @classmethod
    def apply_quantization(
        cls, model: torch.nn.Module, config: AutoAdpQConfig, multi_threaded: int = 1
    ):
        """Apply quantization to a model given a configuration.

        Args:
            model (torch.nn.Module): The model to be quantized.
            config (AutoAdpQConfig): Configuration for quantization.
            multi_threaded (int): Whether to use multi-threaded quantization.
                Default to 1, single-threaded is used. else, specify the number
                of threads.
        """
        quantizer = cls(config=config)

        if quantizer.cfg.target_layers is not None:
            target_suffixes = tuple(quantizer.cfg.target_layers)
        else:
            target_suffixes = quantizer.linear_target_layers

        future_to_module = {}

        logger.info(f"Starting threaded quantization with {multi_threaded} workers...")

        with tqdm(desc="Preparing Layer Weights", unit="layer") as pbar:
            with ThreadPoolExecutor(max_workers=multi_threaded) as executor:
                for name, module in model.named_modules():
                    if isinstance(module, torch.nn.Linear) and name.endswith(
                        target_suffixes
                    ):
                        # logger.info(f"Checking datatype: {name}")
                        # extract weights as numpy array
                        # If Bfloat16, convert to float16 first
                        if module.weight.dtype == torch.bfloat16:
                            # Throw UserWarning if max value exceeds float16 range
                            max_val = torch.max(torch.abs(module.weight)).item()
                            if max_val > 65504:
                                warnings.warn(
                                    f"Max weight value {max_val} exceeds float16 range."
                                    "This may lead to overflow during conversion.",
                                    UserWarning,
                                    stacklevel=2,
                                )
                            weight_array = (
                                module.weight.to(torch.float16).detach().cpu().numpy()
                            )
                        else:
                            weight_array = module.weight.detach().cpu().numpy()

                        # logger.info(f"Quantizing & Reconstructing layer: {name}")
                        future = executor.submit(
                            quantizer.quantize_reconstruct, weight_array
                        )
                        future_to_module[future] = (name, module)
                        pbar.update(1)

            # 2. COLLECTION PHASE
            with tqdm(
                total=len(future_to_module),
                desc="Quantizing Layer Weights",
                unit="layer",
            ) as pbar:
                for future in as_completed(future_to_module):
                    layer_name, layer_module = future_to_module[
                        future
                    ]  # Retrieve correct module

                    try:
                        result = future.result()

                        # Convert result back to tensor
                        original_device = layer_module.weight.device
                        new_weight = torch.tensor(
                            result, dtype=torch.bfloat16, device=original_device
                        )

                        # Assign to the correct module instance
                        layer_module.weight.data = new_weight

                    except Exception as exc:
                        logger.error(f"❌ Exception in layer {layer_name}: {exc}")

                    pbar.update(1)
                    pbar.set_postfix(finished=layer_name, refresh=True)

                # pbar finalize
                pbar.close()
