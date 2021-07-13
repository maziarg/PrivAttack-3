import numpy as np





# l = np.load("/Users/maziargomrokchi/Documents/checkTraj/buffers/Robust_Hopper-v2_20_5_initial_state.npy")
# m = np.load("/Users/maziargomrokchi/Documents/checkTraj/buffers/Robust_Hopper-v2_20_100_initial_state.npy")
# print(np.shape(l))
# print(np.shape(m))
# if not np.array_equal(l[0: 100083, :], m[0: 100083, :]):
#     raise ValueError("Error")

traj = np.array([[1, 2, 3],[4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15]])
print(np.shape(traj))
padding_element = np.tile(np.zeros((1, traj.shape[1])), (int(9) - traj.shape[0], 1))
test_seq = np.vstack([traj, padding_element])
print(test_seq)

last = np.array([traj[-1, :]])
padding_element1 = np.tile(last, (int(9) - traj.shape[0], 1))
test_seq1 = np.vstack([traj, padding_element1])
print(test_seq1)