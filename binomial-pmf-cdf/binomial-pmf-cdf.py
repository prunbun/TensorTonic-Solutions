import math

def binomial_pmf_cdf(n: int, p: float, k: int) -> dict:
    """
    Returns a dictionary with pmf and cdf.
    """

    # res = {"pmf": 0.0, "cdf": 0.0}

    # for i in range(0,k+1):
    #     n_choose_k_prob = math.comb(n, i)
    #     p_to_kth_power = np.pow(p, i)
    #     one_minus_p_to_kth_power = np.pow(1-p, n-i)
        
    #     res["cdf"] += float(n_choose_k_prob * p_to_kth_power * one_minus_p_to_kth_power)

    #     if (i == k):
    #         res["pmf"] = float(n_choose_k_prob * p_to_kth_power * one_minus_p_to_kth_power)

    # return res

    probabilties = [
        math.comb(n, i) * (p ** i) * ((1.0 - p) ** (n - i))
        for i in range(k + 1)
    ]

    return {"pmf": float(probabilties[k]), "cdf": float(sum(probabilties))}