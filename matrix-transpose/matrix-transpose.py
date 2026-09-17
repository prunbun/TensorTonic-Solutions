import numpy as np

def matrix_transpose(A: list) -> np.ndarray:
    """
    Returns the transposed matrix as a NumPy array.
    """

    ROWS, COLS = len(A), len(A[0])
    
    A_transpose = np.zeros((COLS, ROWS), dtype=float)
    

    for r in range(ROWS):
        for c in range(COLS):
            A_transpose[c, r] = A[r][c]

    return A_transpose
