from auto_adpq import Auto_AdpQ

def test_auto_adpq_initialization():
    """Test the initialization of Auto_AdpQ."""
    group_size = 4
    lambda1 = 0.1
    n_iters = 10
    auto_adpq = Auto_AdpQ(group_size, lambda1, n_iters)
    
    assert auto_adpq.group_size == group_size
    assert auto_adpq.lambda1 == lambda1
    assert auto_adpq.n_iters == n_iters