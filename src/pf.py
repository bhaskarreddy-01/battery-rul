from .model import dual_exp
import numpy as np

def pf_run(y_obs, init_params, n_particles=200, R=5e-5):
    """
    Standard Particle Filter for battery capacity estimation.
    Uses prior distribution as importance function.
    """
    N = len(y_obs)
    Q = np.diag([5e-7, 4e-9, 5e-7, 4e-11])   # process noise covariance

    # Initialize
    x = np.tile(init_params, (n_particles, 1))
    x += np.random.randn(n_particles, 4) * np.sqrt(np.diag(Q)) * 10
    w = np.ones(n_particles) / n_particles

    cap_est = np.zeros(N)
    cap_err = np.zeros(N)

    for k in range(N):
        # Propagate particles with process noise
        noise = np.random.multivariate_normal(np.zeros(4), Q, n_particles)
        x = x + noise

        # Predicted capacity
        y_pred = dual_exp(k, x[:, 0], x[:, 1], x[:, 2], x[:, 3])

        # Likelihood-based weight update
        w = np.exp(-0.5 * (y_obs[k] - y_pred) ** 2 / R) + 1e-300
        w /= w.sum()

        # Weighted state estimate
        cap_est[k] = np.dot(w, y_pred)
        cap_err[k] = cap_est[k] - y_obs[k]

        # Systematic resampling when effective sample size is low
        N_eff = 1.0 / np.sum(w ** 2)
        if N_eff < n_particles / 2:
            idx = np.random.choice(n_particles, n_particles, p=w)
            x = x[idx]
            w = np.ones(n_particles) / n_particles

    return cap_est, cap_err, x, w
