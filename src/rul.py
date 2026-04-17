from .model import dual_exp
import numpy as np

def predict_future(x_particles, w_particles, start_k, horizon, threshold):
    """
    Predict future capacity from start_k for `horizon` cycles.
    Returns predicted capacity array and estimated RUL.
    """
    future = np.zeros((len(w_particles), horizon))
    for j in range(horizon):
        kk = start_k + j
        future[:, j] = dual_exp(kk, x_particles[:, 0], x_particles[:, 1],
                                 x_particles[:, 2], x_particles[:, 3])

    pred_cap = np.dot(w_particles, future)

    # RUL = first j where predicted capacity falls below threshold
    rul = horizon  # not found within horizon
    for j in range(horizon):
        if pred_cap[j] <= threshold:
            rul = j
            break

    return pred_cap, rul