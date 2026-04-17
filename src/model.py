import numpy as np

def dual_exp(k, a, b, c, d):
    return a * np.exp(b * k) + c * np.exp(d * k)