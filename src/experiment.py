import numpy as np
import os

from .data_loading import load_nasa_mat
from .ipso import ipso_fit
from .pf import pf_run
from .ipso_pf import ipso_pf_run

BATTERY_CFG = {
    'B5':  ('data/B0005.mat', 168, 2.0, 0),
    'B6':  ('data/B0006.mat', 168, 2.0, 1),
    'B7':  ('data/B0007.mat', 168, 2.0, 2),
    'B18': ('data/B0018.mat', 132, 2.0, 3),
}

THRESHOLDS = {'B5': 1.4, 'B6': 1.6, 'B7': 1.4, 'B18': 1.4}

battery_data = {}
for name, (fname, n, rated, seed) in BATTERY_CFG.items():
    if os.path.exists(fname):
        battery_data[name] = load_nasa_mat(fname)
        print(f"  {name}: loaded {len(battery_data[name])} cycles from {fname}")
    else:
        print(f"  {name}: synthetic ({n} cycles)")
        
        
print("\nFitting IPSO parameters and running filters...")

results = {}
for name, y in battery_data.items():
    print(f"  Processing {name} ({len(y)} cycles)...")
    threshold = THRESHOLDS[name]

    # Fit initial model parameters with IPSO
    init_p = ipso_fit(y, n_particles=50, n_iter=80)

    # Run both filters
    cap_pf,      err_pf,      xp_pf,  wp_pf  = pf_run(y, init_p)
    cap_ipso_pf, err_ipso_pf, xp_ipo, wp_ipo = ipso_pf_run(y, init_p)

    # Compute MAE / RMSE
    mae_pf   = np.mean(np.abs(err_pf))
    rmse_pf  = np.sqrt(np.mean(err_pf ** 2))
    mae_ipo  = np.mean(np.abs(err_ipso_pf))
    rmse_ipo = np.sqrt(np.mean(err_ipso_pf ** 2))

    print(f"    PF      -> MAE={mae_pf:.4f}  RMSE={rmse_pf:.4f}")
    print(f"    IPSO-PF -> MAE={mae_ipo:.4f}  RMSE={rmse_ipo:.4f}")

    results[name] = dict(
        y=y, threshold=threshold,
        cap_pf=cap_pf, err_pf=err_pf, xp_pf=xp_pf, wp_pf=wp_pf,
        cap_ipso=cap_ipso_pf, err_ipso=err_ipso_pf, xp_ipo=xp_ipo, wp_ipo=wp_ipo,
        init_p=init_p,
        mae_pf=mae_pf, rmse_pf=rmse_pf,
        mae_ipo=mae_ipo, rmse_ipo=rmse_ipo,
    )

print("Done.\n")