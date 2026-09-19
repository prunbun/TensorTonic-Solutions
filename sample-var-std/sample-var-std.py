import numpy as np

def sample_var_std(x: list) -> dict:
    """
    Returns a dictionary with variance and standard_deviation.
    """
    
    x = np.array(x)
    x_bar = np.mean(x)

    variance = np.sum(np.pow(x - x_bar, 2)) / (len(x) - 1)

    std_dev = np.sqrt(variance)

    return {"variance": float(variance), "standard_deviation": float(std_dev)}