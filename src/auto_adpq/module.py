"""Auto ADPQ module."""

class Auto_AdpQ:
    """Auto_AdpQ.

    Runs the AdpQ algorithm.
    """

    def __init__(self, group_size, lambda1, n_iters):
        """Init AutoADPQ.

        Args:
            group_size (int): _description_
            lambda1 (float): _description_
            n_iters (int): _description_
        """
        self.group_size = group_size
        self.lambda1 = lambda1
        self.n_iters = n_iters

    def quantize(self, sub_vector):
        """Quantize.

        quantize a sub-vector from a group quantization.

        Args:
            sub_vector (numpy.ndarray): the sub-vector to quantize.
        """
        pass
