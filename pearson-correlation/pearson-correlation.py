import numpy as np

def pearson_correlation(X: list) -> np.ndarray:
    """
    Returns the correlation matrix as a NumPy array.
    """


    '''
    ok so we have to find covariance(i, j) / (sqrt(variance_i) * sqrt(variance_j))

    it has to be in a matrix

    so we know X has shape N x D, the result has to be D x D
    if we do X.T @ X ==> D x N @ N x D ==> we get D x D but each element is actually:
        sum_{n = 0}^N x_ni * x_nj for each feature

    this is not exactly correct:
        the covariance is sum_{n = 0}^N (x_ni - xbar_i) * (x_nj - xbar_j)
        which expands to:
            sum_{n = 0}^N x_ni * x_nj - xbar_i * x_nj - xbar_j * x_ni + xbar_i * xbar_j
        so we need to 'correct' this:
            there is a nice trick where we can say that:
                sum_{n = 0}^N xbar_i * x_nj
                xbar_i sum_{n = 0}^N x_nj (take out xbar_i)
                xbar_i * N * xbar_j (exclam! look at the replacement)
            
        then, the main expression fully collapses
        sum_{n = 0}^N x_ni * x_nj - N * xbar_i xbar_j - N * xbar_i xbar_j + N * xbar_i xbar_j
        sum_{n = 0}^N x_ni * x_nj - N * xbar_i xbar_j

    now N * xbar_i xbar_j is not hard to compute, we can do the mean on axis = 0 on X 
    and form the D x D matrix of means and subtract N copies from X.T @ X

    but that is not stable due to precision issues, so we will actually center X first
    and then xbar_i and xbar_j is just 0, so we can just do X.T @ X and divide by (N - 1)

    finally, we have to deal with the denominator, but remember we centered the matrix!
    so that means that we are just looking for:
        sum_{n = 0}^N (xn_i)^2 (notice the mean is 0, exclam!)

        and guess what, the diagonal of X.T @ X is exactly this!!

        and guess what else the denominator is also N - 1 for the variance...

    this means, we can just take the np.diag of the centered X.T @ X and we get each variance
        NOTE: before dividing by N - 1
    then, notice we need std dev, but we have var, so we need to sqrt, but for each term,
    it is a product of std dev in the denominator and we can merge them
    so now, we can get another D x D matrix of variances (need to divide by ((N - 1)^2)
    but then we sqrt the whole term so now we only need to divide by N - 1 once
    and guess what, that cancels out with the N - 1 from the numerator
    '''

    # center matrix to enable simplifications when the mean is 0
    X = np.array(X)
    x_bar = np.mean(X, axis = 0)
    X_centered = X - x_bar
    
    # compute covariance (partial result, not divided by N - 1)
    X_cov = X_centered.T @ X_centered

    # compute variance products
    feature_variances = np.expand_dims(np.diag(X_cov), axis = -1) # .T does nothing on (D,)
    feature_variance_products = feature_variances @ feature_variances.T

    # compute std dev
    feature_std_dev_products = np.sqrt(feature_variance_products)

    # assemble final result

    return X_cov / feature_std_dev_products