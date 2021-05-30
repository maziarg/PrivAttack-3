import numpy as np

a = np.load("/Users/maziargomrokchi/learning_output/Hopper-v3/2000/100/20/buffers/Robust_Hopper-v3_100_initial_state.npy")
b = np.load("/Users/maziargomrokchi/learning_output/Hopper-v3/2000/100/20/buffers/Robust_Hopper-v3_100_action.npy")
c = np.load("/Users/maziargomrokchi/learning_output/Hopper-v3/2000/100/20/buffers/Robust_Hopper-v3_100_reward.npy")
d = np.load("/Users/maziargomrokchi/learning_output/Hopper-v3/2000/100/20/buffers/Robust_Hopper-v3_100_state.npy")
e = np.load("/Users/maziargomrokchi/learning_output/Hopper-v3/2000/100/20/buffers/Robust_Hopper-v3_100_next_state.npy")
print('a' , np.shape(a))
print('b' , np.shape(b))
print('c', np.shape(c))
print('d', np.shape(d))
print('e', np.shape(e))
# print(a)
# print(d)


print(np.load("/Users/maziargomrokchi/learning_output/Hopper-v3/500/1/15/results/behavioral_Hopper-v3_1.npy"))


f = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
g = np.array([0, 2])
print(np.shape(g))
h = ([f[g[i], :] for i in range(len(g))])
print(len(h))

print(np.load('/Users/maziargomrokchi/learning_output/Hopper-v3/500/1/15/buffers/Robust_Hopper-v3_1_number_of_trajectories.npy'))
print(np.load('/Users/maziargomrokchi/learning_output/Hopper-v3/500/1/15/buffers/target_Robust_Hopper-v3_1_200_compatible_number_of_trajectories.npy'))
print(np.shape(np.load('/Users/maziargomrokchi/learning_output/Hopper-v3/500/1/15/buffers/Robust_Hopper-v3_1_initial_state.npy')))

x = np.array([[1, 2], [3, 4], [5, 6]])
last = np.array(x[-1])
last = np.tile(last, (4, 1))
final = np.concatenate([x, last])
final = np.ravel(final)
print(final)


y = np.load("/Users/maziargomrokchi/learning_output/Hopper-v3/500/1/15/buffers/Robust_Hopper-v3_1_action.npy")
y = np.load("/Users/maziargomrokchi/learning_output/Hopper-v3/500/1/15/buffers/Robust_Hopper-v3_1_initial_state.npy")
print(y)