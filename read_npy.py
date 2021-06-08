import numpy as np





l = np.load("/Users/maziargomrokchi/Documents/checkTraj/buffers/Robust_Hopper-v2_20_5_initial_state.npy")
m = np.load("/Users/maziargomrokchi/Documents/checkTraj/buffers/Robust_Hopper-v2_20_100_initial_state.npy")
print(np.shape(l))
print(np.shape(m))
if not np.array_equal(l[0: 100083, :], m[0: 100083, :]):
    raise ValueError("Error")