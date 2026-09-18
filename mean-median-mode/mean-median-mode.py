from collections import Counter
import numpy as np

def mean_median_mode(x: list) -> dict:
    """
    Returns a dictionary with mean, median, and mode.
    """
    res = {}
    x = np.array(x)

    res['median'] = float(np.median(x))
    res['mean'] = float(np.mean(x))

    frequencies = Counter(x)
    MAX_COUNT = max(frequencies.values())

    for key in sorted(x):
        if frequencies[key] == MAX_COUNT:
            res['mode'] = float(key)
            break

    return res
    