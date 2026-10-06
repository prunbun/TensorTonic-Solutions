import numpy as np

def matrix_normalization(matrix: list, axis=None, norm_type: str = "l2") -> np.ndarray:
    """
    Returns a NumPy array with the same shape as matrix.
    """
    mat = np.array(matrix)

    def l1norm_mat(mat, axis):
        if axis == None:
            norm = np.sum(np.abs(mat))
            return mat / norm

        l1norms = np.expand_dims(np.sum(np.abs(mat), axis=axis), axis=axis)
        return mat / l1norms

    def l2norm_mat(mat, axis):
        if axis == None:
            norm = np.sqrt(np.sum(np.pow(mat, 2)))
            return mat / norm
            

        l2norms = np.sqrt(np.sum(np.pow(mat, 2), axis=axis))
        l2norms = np.expand_dims(l2norms, axis=axis)

        print(mat, axis, l2norms, mat / l2norms)
        return mat / l2norms

    def infnorm_mat(mat, axis):
        if axis == None:
            norm = np.max(np.abs(mat))
            return mat / norm

        infnorms = np.expand_dims(np.max(np.abs(mat), axis=axis), axis=axis)
        return mat / infnorms


    res = None
    
    if norm_type == 'l1':
        res = l1norm_mat(mat, axis)
    elif norm_type == 'l2':
        res = l2norm_mat(mat, axis)
    else:
        res = infnorm_mat(mat, axis)

    return np.nan_to_num(res, nan=0.0)

        

    