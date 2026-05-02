import numpy as np
from wasserstein_profile import profile_p

R = np.array([0.0, 1.0, 2.0], dtype=np.float64)
B = np.array([-2.0, -1.0, 0.0], dtype=np.float64)

costs = profile_p(R, B, 2)
print(costs)