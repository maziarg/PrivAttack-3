import argparse
import math
import os
import shutil
from itertools import permutations
from random import shuffle, sample

import numpy as np

def format_trajectory(trajectory_length, traj_plk):
    counter = 0
    curr_traj = []
    num_trajectories_to_return = int(len(traj_plk) / trajectory_length)
    complete_trajectory_list = []
    append_list = complete_trajectory_list.append

    for (obs, action, reward, done) in traj_plk:
        # pad trajectory and reset counter
        if done and counter < trajectory_length:
            append_list(
                flatten_trajectory(pad_trajectory(curr_traj, obs, counter, trajectory_length, action.shape[0])))
            counter = 0
            curr_traj = []
        elif counter == (trajectory_length - 1):
            curr_traj.append(flatten_tuple(obs, action, reward))
            append_list(flatten_trajectory(curr_traj))
            counter = 0
            curr_traj = []
        else:
            curr_traj.append(flatten_tuple(obs, action, reward))
            counter += 1
    return np.asarray(sample(complete_trajectory_list, num_trajectories_to_return))


def flatten_trajectory(curr_traj):
    return np.concatenate(curr_traj)


def pad_trajectory(curr_traj, obs, counter, trajectory_length, act_dim):
    append = curr_traj.append
    while counter < trajectory_length:
        append(flatten_tuple(obs, np.zeros(act_dim), 0))
        counter += 1
    return curr_traj


def flatten_tuple(obs, action, reward):
    return np.concatenate([obs, action, [reward]])


def is_same_set(num_traj_per_model, x_i, y_i):
    # check if the two indexes are from the same set
    if x_i == 0:
        if y_i < num_traj_per_model:
            return True
    if y_i == 0:
        if x_i < num_traj_per_model:
            return True

    elif math.floor(x_i / num_traj_per_model) == math.floor(y_i / num_traj_per_model):
        return True

    else:
        return False


def pad_pairs(pairs, attack_training_size):
    remaining = attack_training_size - len(pairs)
    cont = True
    counter = 0
    append = pairs.append
    while cont:
        for (xi, yi) in pairs:
            if counter < remaining:
                append((xi, yi))
            else:
                cont = False
                break
            counter += 1
    return pairs


def generate_pairs(total_pairs_needed, available_trajectories, num_predictions, attack_train_size):
    print("generating pairs")
    test_pairs = []
    perms = math.factorial(available_trajectories) / math.factorial(available_trajectories - 2)

    if perms < total_pairs_needed:
        pairs = list(permutations(range(0, available_trajectories), 2))
        shuffle(pairs)

        for i in range(num_predictions):
            x_i, y_i = pairs.pop()
            test_pairs.append((x_i, y_i))

        train_pairs = pad_pairs(pairs, attack_train_size)
    else:
        pairs = sample(list(permutations(range(0, available_trajectories), 2)), total_pairs_needed)
        shuffle(pairs)

        for i in range(num_predictions):
            x_i, y_i = pairs.pop()
            test_pairs.append((x_i, y_i))

        train_pairs = pairs

    return train_pairs, test_pairs


def get_models(x_i, y_i, num_trajectories_per_model):
    if x_i == 0:
        x_model = 0
    else:
        x_model = math.floor((x_i - 1) / num_trajectories_per_model)

    if y_i == 0:
        y_model = 0
    else:
        y_model = math.floor((y_i - 1) / num_trajectories_per_model)

    if x_model == y_model:
        same_set = True
    else:
        same_set = False

    index_x = x_i % num_trajectories_per_model
    index_y = y_i % num_trajectories_per_model
    return x_model, y_model, same_set, index_x, index_y


def print_experiment(env, num_models, threshold, num_predictions,
                     max_traj_len):
    print("Running Environment: ", env)
    print("Seeds: ", num_models)
    print("Threshold: ", threshold)
    print("Number of predictions: ", num_predictions)
    print("Maximum Trajectory Length: ", max_traj_len)
    print("---------------------------")


def cleanup(files, buffers):
    path = 'tmp_plks/'
    for f in files:
        if os.path.exists(path + f + ".plk"):
            os.remove(path + f + ".plk")
        if os.path.exists(path + f + ".npy"):
            os.remove(path + f + ".npy")
    for b in buffers:
        if os.path.exists(path + b + ".buffer"):
            os.remove(path + b + ".buffer")
    if os.path.exists('tmp/'):
        shutil.rmtree('tmp/')


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')