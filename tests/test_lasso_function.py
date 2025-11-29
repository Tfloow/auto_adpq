from auto_adpq import Auto_AdpQ, AutoAdpQConfig
import pytest
import numpy as np

def test_slow_fast():
    matrix = np.load("tests/weights/llama-8B/model.layers.0.self_attn.q_proj.weight.npy")
    
    config = AutoAdpQConfig()
    auto_adpq = Auto_AdpQ(config=config)
    
    for i in range(1,100):
        lambda_prime = 10*i
        _, slow_outliers = auto_adpq._optimization_function(matrix, lambda_prime=lambda_prime)
        fast_outliers = auto_adpq._optimization_function_fast(matrix, lambda_prime=lambda_prime)
        
        assert slow_outliers == fast_outliers, f"Mismatch between slow and fast at lambda_prime={lambda_prime}"
