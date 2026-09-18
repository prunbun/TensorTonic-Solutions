import numpy as np

def cosine_similarity(a: list, b: list) -> float:
    """
    Returns the cosine similarity as a Python float.
    """
    a = np.array(a, dtype = float)
    b = np.array(b, dtype = float)

    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)

    if a_norm == 0 or b_norm == 0:
        return 0.0

    cosine_result = np.dot(a, b) / (a_norm * b_norm)

    return float(cosine_result)