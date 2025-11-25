"""Auto ADPQ module."""

from typing import Optional

import numpy as np
from pydantic import BaseModel


class AutoAdpQConfig(BaseModel):
    """AutoADPQ config.

    defines the basic configuration for the class auto ADPQ

    Args:
        BaseModel (): PYDANTIC BASE MODEL.
    """

    group_size: int
    lambda1: float
    n_iters: int
    device: str = "cpu"
    q_bit: int = 4
    data_packing: bool = True


class Auto_AdpQ:
    """Auto_AdpQ.

    Runs the AdpQ algorithm.
    """

    def __init__(
        self,
        group_size: int,
        lambda1: float,
        n_iters: int,
        device: str = "cpu",
        q_bit: int = 4,
        data_packing: bool = True,
        config: Optional[AutoAdpQConfig] = None,
    ):
        """Init AutoADPQ.

        Args:
            group_size (int): the group size.
            lambda1 (float): the lambda1 parameter.
            n_iters (int): number of iterations.
            device (str, optional): the device to use. Defaults to "cpu".
            q_bit (int, optional): the quantization bit. Defaults to 4.
            data_packing (bool, optional): whether to use data packing.
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
                lambda1=lambda1,
                n_iters=n_iters,
                device=device,
                q_bit=q_bit,
                data_packing=data_packing,
            )

        # assign validated attributes
        self.group_size = cfg.group_size
        self.lambda1 = cfg.lambda1
        self.n_iters = cfg.n_iters
        self.device = cfg.device
        self.q_bit = cfg.q_bit
        self.data_packing = cfg.data_packing

    def quantize(self, sub_vector):
        """Quantize.

        quantize a sub-vector from a group quantization.

        Args:
            sub_vector (numpy.ndarray): the sub-vector to quantize.
        """
        max_value = np.max(np.abs(sub_vector))
        Delta = max_value / (2 ** (self.q_bit - 1) - 1)
        quantized = np.round(sub_vector / Delta)

        return quantized, Delta

    def reconstruct_vector(self, quantized_vectors, outlier_indices):
        """Reconstruct quantized.

        Reconstruct the quantized sub-vector.

        Args:
            quantized_vectors (list[numpy.ndarray]): the quantized sub-vector.
                                        pass first non-outliers, then outliers.
            outlier_indices (list[int]): the indices of the outliers.
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

    def lasso_outlier_detection(self, matrix):
        """Lasso outlier detection.

        Detect outliers in the vector using Lasso regression.

        Args:
            matrix (numpy.ndarray): the input matrix.
        """
        lambda_prime = 1e5
        
        raise NotImplementedError("Lasso outlier detection is not implemented yet.")

    def lasso_quantization(self, vector):
        """Lasso quantization.

        Quantize the vector using Lasso regression.

        Args:
            vector (numpy.ndarray): the input vector.
        """
        outlier_index = self.lasso_outlier_detection(vector)

        outlier_vector = vector[outlier_index]
        non_outlier_vector = np.delete(vector, outlier_index)

        # Quantize non-outlier vector
        quantized_non_outlier, _ = self.quantize(non_outlier_vector)
        quantized_outlier, _ = self.quantize(outlier_vector)

        # Reconstruct quantized vectors
        quantized_vectors = (quantized_non_outlier, quantized_outlier)
        return self.reconstruct_vector(quantized_vectors, outlier_index)

    def quantize_weight_matrix(self, weight_matrix):
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
