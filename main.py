import numpy as np
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

from src.experiment import results
from src.plotting import plot_capacity, plot_rul_k58, plot_rul_k78

print("Generating plots...")

plot_capacity(results)
plot_rul_k58(results)
plot_rul_k78(results)

print("Done.")