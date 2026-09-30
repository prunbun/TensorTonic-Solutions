import numpy as np

def covariance_matrix(X: list) -> np.ndarray:
    """
    Returns the covariance matrix as a NumPy array.
    """
    N, D = len(X), len(X[0])
    X = np.array(X)

    # center the features
    x_bar = np.mean(X, axis=0)
    X = X - x_bar

    covariance_matrix = (X.T @ X) / (N - 1)

    return covariance_matrix

    

    