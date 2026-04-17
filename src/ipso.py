from .model import dual_exp
import numpy as np

def ipso_fit(y_obs, n_particles=50, n_iter=100):
    """
    Fit dual-exponential parameters to observed capacity using IPSO.
    Returns optimal parameter vector [a, b, c, d].
    """
    N = len(y_obs)
    k = np.arange(N)

    # Parameter search bounds
    lb = np.array([0.30, -0.05,  0.30, -0.005])
    ub = np.array([2.00, -0.001, 2.00, -0.0001])

    # Initialize swarm
    pos = lb + np.random.rand(n_particles, 4) * (ub - lb)
    vel = np.zeros_like(pos)
    pbest = pos.copy()
    pbest_cost = np.full(n_particles, np.inf)
    gbest = pos[0].copy()
    gbest_cost = np.inf

    # IPSO hyperparameters (Eq. 6 & 7 in paper)
    w_max, w_min = 0.9, 0.4
    sigma = 0.1
    c1_s, c1_e = 2.5, 0.5   # c1 decreases over iterations
    c2_s, c2_e = 1.0, 3.0   # c2 increases over iterations

    def cost(p):
        pred = dual_exp(k, *p)
        return np.mean((y_obs - pred) ** 2)

    for t in range(n_iter):
        # Adaptive inertia weight (Eq. 6)
        w = (w_min + (w_max - w_min) * np.exp(-t / n_iter)
             + sigma * np.random.beta(1, 3))

        # Dynamic learning factors using tan/arctan (Eq. 7)
        ratio = t / n_iter
        c1 = (c1_s - c1_e) * np.tan(0.875 * (1 - ratio ** 0.6)) + c1_e
        c2 = (c2_s - c2_e) * np.arctan(2.8 * (1 - ratio ** 0.4)) + c2_e
        c1 = float(np.clip(c1, 0.1, 4.5))
        c2 = float(np.clip(c2, 0.1, 4.5))

        # Evaluate & update personal / global best
        for i in range(n_particles):
            ci = cost(pos[i])
            if ci < pbest_cost[i]:
                pbest_cost[i] = ci
                pbest[i] = pos[i].copy()
            if ci < gbest_cost:
                gbest_cost = ci
                gbest = pos[i].copy()

        # Velocity & position update
        r1 = np.random.rand(n_particles, 4)
        r2 = np.random.rand(n_particles, 4)
        vel = w * vel + c1 * r1 * (pbest - pos) + c2 * r2 * (gbest - pos)
        pos = np.clip(pos + vel, lb, ub)

    return gbest
