import numpy as np
import os

def load_nasa_mat(filepath):
    from scipy.io import loadmat
    mat = loadmat(filepath, simplify_cells=True)
    key = [k for k in mat if not k.startswith('_')][0]
    caps = []
    for cyc in mat[key]['cycle']:
        if cyc['type'] == 'discharge':
            c = cyc['data']['Capacity']
            if np.isscalar(c) and c > 0.5:
                caps.append(float(c))
    return np.array(caps)