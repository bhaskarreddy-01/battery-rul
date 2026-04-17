import numpy as np
from .model import dual_exp

def ipso_pf_run(y_obs, init_params, n_particles=200, R=5e-5, inner_iter=15):
    """
    IPSO-PF: uses IPSO to guide particle updates (importance distribution).
    Reduces particle degeneracy compared to standard PF.
    """
    N = len(y_obs)
    Q = np.diag([5e-7, 4e-9, 5e-7, 4e-11])
    lb = np.array([0.30, -0.05,  0.30, -0.005])
    ub = np.array([2.00, -0.001, 2.00, -0.0001])

    # Initialize
    x = np.tile(init_params, (n_particles, 1))
    x += np.random.randn(n_particles, 4) * np.sqrt(np.diag(Q)) * 10
    vel = np.zeros_like(x)
    w = np.ones(n_particles) / n_particles

    # Personal bests start at initial positions
    pbest = x.copy()
    pbest_fit = np.full(n_particles, -np.inf)

    # IPSO parameters
    w_max, w_min = 0.9, 0.4
    sigma = 0.1
    c1_s, c1_e = 2.5, 0.5
    c2_s, c2_e = 1.0, 3.0

    cap_est = np.zeros(N)
    cap_err = np.zeros(N)

    for k in range(N):
        # Propagate with process noise
        x = x + np.random.multivariate_normal(np.zeros(4), Q, n_particles)

        # Fitness: how well each particle matches current observation
        y_pred = dual_exp(k, x[:, 0], x[:, 1], x[:, 2], x[:, 3])
        fitness = np.exp(-0.5 * (y_obs[k] - y_pred) ** 2 / R)

        # Fitness function from Eq. 13 in paper
        fit_fn = np.exp(-0.5 * (y_obs[k] - y_pred) ** 2 / R)

        # Update personal bests
        better = fit_fn > pbest_fit
        pbest[better] = x[better].copy()
        pbest_fit[better] = fit_fn[better]

        # Global best
        gbest = pbest[np.argmax(pbest_fit)].copy()

        # Inner IPSO loop to move particles toward better solutions
        for t in range(inner_iter):
            ratio = t / inner_iter
            wi = (w_min + (w_max - w_min) * np.exp(-t / inner_iter)
                  + sigma * np.random.beta(1, 3))
            c1 = float(np.clip(
                (c1_s - c1_e) * np.tan(0.875 * (1 - ratio ** 0.6)) + c1_e, 0.1, 4.5))
            c2 = float(np.clip(
                (c2_s - c2_e) * np.arctan(2.8 * (1 - ratio ** 0.4)) + c2_e, 0.1, 4.5))

            r1 = np.random.rand(n_particles, 4)
            r2 = np.random.rand(n_particles, 4)
            vel = wi * vel + c1 * r1 * (pbest - x) + c2 * r2 * (gbest - x)
            x = np.clip(x + vel, lb, ub)

        # Weight update after IPSO refinement
        y_pred = dual_exp(k, x[:, 0], x[:, 1], x[:, 2], x[:, 3])
        w = np.exp(-0.5 * (y_obs[k] - y_pred) ** 2 / R) + 1e-300
        w /= w.sum()

        cap_est[k] = np.dot(w, y_pred)
        cap_err[k] = cap_est[k] - y_obs[k]

        # Resample
        N_eff = 1.0 / np.sum(w ** 2)
        if N_eff < n_particles / 2:
            idx = np.random.choice(n_particles, n_particles, p=w)
            x = x[idx];  pbest = pbest[idx]
            pbest_fit = pbest_fit[idx];  vel = vel[idx]
            w = np.ones(n_particles) / n_particles

    return cap_est, cap_err, x, w