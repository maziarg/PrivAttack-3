import numpy as np
import torch


class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim, device, max_size=int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        self.num_trajectories = 0

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))
        self.initial_state = []
        self.trajectory_end_index = []

        self.device = device

    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done
        if done:
            self.trajectory_end_index.append(self.ptr)
            self.num_trajectories += 1
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device)
        )

    def save(self, save_folder):
        np.save(f"{save_folder}_state.npy", self.state[:self.size])
        np.save(f"{save_folder}_action.npy", self.action[:self.size])
        np.save(f"{save_folder}_next_state.npy", self.next_state[:self.size])
        np.save(f"{save_folder}_reward.npy", self.reward[:self.size])
        np.save(f"{save_folder}_not_done.npy", self.not_done[:self.size])
        np.save(f"{save_folder}_ptr.npy", self.ptr)
        np.save(f"{save_folder}_initial_state.npy", self.initial_state)
        np.save(f"{save_folder}_trajectory_end_index.npy", self.trajectory_end_index)
        np.save(f"{save_folder}_number_of_trajectories.npy", self.num_trajectories)

    def load(self, save_folder, size=-1):
        reward_buffer = np.load(f"{save_folder}_reward.npy")

        # Adjust crt_size if we're using a custom size
        size = min(int(size), self.max_size) if size > 0 else self.max_size
        self.size = min(reward_buffer.shape[0], size)

        self.state[:self.size] = np.load(f"{save_folder}_state.npy")[:self.size]
        self.action[:self.size] = np.load(f"{save_folder}_action.npy")[:self.size]
        self.next_state[:self.size] = np.load(f"{save_folder}_next_state.npy")[:self.size]
        self.reward[:self.size] = reward_buffer[:self.size]
        self.not_done[:self.size] = np.load(f"{save_folder}_not_done.npy")[:self.size]
        self.num_trajectories = int(np.load(f"{save_folder}_number_of_trajectories.npy"))
        self.trajectory_end_index[:self.num_trajectories] = np.load(f"{save_folder}_trajectory_end_index.npy")
        self.initial_state[:self.num_trajectories] = np.load(f"{save_folder}_initial_state.npy")


