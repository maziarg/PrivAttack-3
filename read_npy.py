import numpy as np





# l = np.load("/Users/maziargomrokchi/Documents/checkTraj/buffers/Robust_Hopper-v2_20_5_initial_state.npy")
# m = np.load("/Users/maziargomrokchi/Documents/checkTraj/buffers/Robust_Hopper-v2_20_100_initial_state.npy")
# print(np.shape(l))
# print(np.shape(m))
# if not np.array_equal(l[0: 100083, :], m[0: 100083, :]):
#     raise ValueError("Error")

# traj = np.array([[1, 2, 3],[4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15]])
# print(np.shape(traj))
# padding_element = np.tile(np.zeros((1, traj.shape[1])), (int(9) - traj.shape[0], 1))
# test_seq = np.vstack([traj, padding_element])
# print(test_seq)
#
# last = np.array([traj[-1, :]])
# padding_element1 = np.tile(last, (int(9) - traj.shape[0], 1))
# test_seq1 = np.vstack([traj, padding_element1])
# print(test_seq1)

robust_5 = np.load("/Users/maziargomrokchi/test_data/seed_5/Robust_Hopper-v2_20_5_trajectory_end_index.npy")
robust_100 = np.load("/Users/maziargomrokchi/test_data/seed_100/Robust_Hopper-v2_20_100_trajectory_end_index.npy")
robust_700 = np.load("/Users/maziargomrokchi/test_data/seed_700/Robust_Hopper-v2_200_700_trajectory_end_index.npy")
robust_75 = np.load("/Users/maziargomrokchi/test_data/seed_75/Robust_Hopper-v2_200_75_trajectory_end_index.npy")

target_5 = np.load("/Users/maziargomrokchi/test_data/seed_5/target_Robust_Hopper-v2_"
                   "20_5_1000000_compatible_trajectory_end_index.npy")
target_100 = np.load("/Users/maziargomrokchi/test_data/seed_100/target_Robust_Hopper-v2_"
                     "20_100_1000000_compatible_trajectory_end_index.npy")
target_700 = np.load("/Users/maziargomrokchi/test_data/seed_700/target_Robust_Hopper-v2_"
                     "200_700_1000000_compatible_trajectory_end_index.npy")
target_75 = np.load("/Users/maziargomrokchi/test_data/seed_75/target_Robust_Hopper-v2_"
                    "200_75_1000000_compatible_trajectory_end_index.npy")

print(f"robust_5 shape = {np.shape(robust_5)}")
print(f"robust_100_shape = {np.shape(robust_100)}")
print(f"robust_700 shape = {np.shape(robust_700)}")
print(f"robust_75_shape = {np.shape(robust_75)}")

print(f"target_5 shape = {np.shape(target_5)}")
print(f"target_100_shape = {np.shape(target_100)}")
print(f"target_700 shape = {np.shape(target_700)}")
print(f"target_75_shape = {np.shape(target_75)}")

robust_traj_len_5 = np.zeros(len(robust_5))
robust_traj_len_100 = np.zeros(len(robust_100))
robust_traj_len_700 = np.zeros(len(robust_700))
robust_traj_len_75 = np.zeros(len(robust_75))

target_traj_len_5 = np.zeros(len(target_5))
target_traj_len_100 = np.zeros(len(target_100))
target_traj_len_700 = np.zeros(len(target_700))
target_traj_len_75 = np.zeros(len(target_75))

print(vars()[f"robust_traj_len_{5}"])
for i in [5, 100, 700, 75]:
    vars()[f"robust_traj_len_{i}"][0] = vars()[f"robust_{i}"][0] + 1
    vars()[f"target_traj_len_{i}"][0] = vars()[f"target_{i}"][0] + 1
    for j in range(1, len(vars()[f"robust_{i}"]) - 1):
        vars()[f"robust_traj_len_{i}"][j] = vars()[f"robust_{i}"][j] - vars()[f"robust_{i}"][j - 1]
    for j in range(1, len(vars()[f"target_{i}"]) - 1):
        vars()[f"target_traj_len_{i}"][j] = vars()[f"target_{i}"][j] - vars()[f"target_{i}"][j - 1]

    print(f"robust_traj_len_{i} = ", vars()[f"robust_traj_len_{i}"], np.average(vars()[f"robust_traj_len_{i}"]))
    print(f"target_traj_len_{i} = ", vars()[f"target_traj_len_{i}"], np.average(vars()[f"target_traj_len_{i}"]))
