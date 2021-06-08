import gc
import math
import os
import uuid
from random import randint, SystemRandom
import BCQutils
import BCQ
import pathlib
import logging
import pandas as pd
import copy
logger = logging.getLogger(__name__)

import numpy as np
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
# from sklearn import cross_validation, metrics   #Additional scklearn functions
from sklearn.model_selection import cross_validate
from sklearn import metrics
from sklearn.model_selection import GridSearchCV   #Performing grid search
from scipy.stats.mstats import gmean

from pandas import DataFrame
from utils.configs import CORRELATED, DECORRELATED, SEMI_CORRELATED, CORRELATION_MAP
from utils.helpers import cleanup, print_experiment, generate_pairs, get_models

import matplotlib.pylab as plt
#matplotlib inline
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 12, 4

# anonymous functions to randomly select a number of items in an np.array
RAND_SELEC_FUNC_REPLACE_FALSE = lambda data, num: np.random.choice(data, num, replace=False)
RAND_SELEC_FUNC_REPLACE_TRUE = lambda data, num: np.random.choice(data, num, replace=True)


def get_random_seqs(seq_source, seq_size, eval_size):
    # To randomly select train, test, and eval items, we need to cache train and test first,
    # then remove the selected ones from the entire list, and then select eval ones
    source = list(range(seq_source))
    seq_selected = RAND_SELEC_FUNC_REPLACE_FALSE(source, seq_size)
    # we need to remove the selected items before we select eval seq
    source = [val for val in source if val not in seq_selected]
    eval_seq_selected = RAND_SELEC_FUNC_REPLACE_FALSE(source, eval_size)
    return seq_selected, eval_seq_selected


def get_trajectory(seed, index, trajectory_length):
    path = "tmp/"
    npy_train = path + str(seed) + '_' + str(trajectory_length) + '.npy'

    return np.load(npy_train, 'r', allow_pickle=True)[index]


def get_trajectory_test(seed, index, trajectory_length):
    path = "tmp/"
    npy_test = path + str(seed) + '_' + str(trajectory_length) + '_test.npy'

    return np.load(npy_test, 'r', allow_pickle=True)[index]


def compute_max_trajectory_length(trajectories_end_indices):
    """
    from the list of trajectory end indexes, finds the maximum trajectory length
    """
    max_length = 0
    # The following is because the first item in the for loop would be 1 less than the expected value
    previous_index = -1
    for index in trajectories_end_indices:
        max_length = max((index - previous_index), max_length)
        previous_index = index
    return max_length


def pad_traj(traj, padd_len):
    """adds padding to a trajectory"""
    if not isinstance(traj, np.ndarray):
        raise Exception("Failed to padd the trajectory: Wrong trajectory type")
    # last = np.array([traj[-1]])
    # padding_element = np.tile(last, (int(padd_len) - traj.shape[0], 1))
    padding_element = np.tile(np.zeros((1, traj.shape[1])), (int(padd_len) - traj.shape[0], 1))
    test_seq = np.vstack([traj, padding_element])
    return test_seq


def get_seeds_pairs(label, seeds, index=0, test=False):
    """
    To create trajectory pairs
    For label 1, need to pair train and test from seed 0
    For label 0, need to pair test from seed 0 and train from seed 1
    Note: evidence == test == seed 0
    """
    if test:
        if label:
            train_seed = int(seeds[0])
            test_seed = int(seeds[0])
            # train_seed = int(seeds[0])
            # test_seed = int(seeds[0])
        else:
            train_seed = int(seeds[1])
            test_seed = int(seeds[0])
    else:
        if label:
            train_seed = int(seeds[index])
            test_seed = int(seeds[index])
        else:
            if index != 0:
                train_seed = int(seeds[index - 1])
            else:
                train_seed = int(seeds[-1])

            test_seed = int(seeds[index])

    return train_seed, test_seed

# def get_seeds_train_pairs(label, seeds):
#     """
#     To create trajectory pairs
#     For label 1, need to pair train and test from seed 0
#     For label 0, need to pair test from seed 0 and train from seed 1
#     Note: evidence == test == seed 0
#     """
#     if label:
#         train_seed = int(seeds[0])
#         test_seed = int(seeds[0])
#     else:
#         train_seed = int(seeds[1])
#         test_seed = int(seeds[0])
#
#     return train_seed, test_seed

def get_seeds_test_pairs(label, seeds):
    """Following the logic from get_seeds_train_pairs"""
    if label:
        train_seed = int(seeds[2])
        test_seed = int(seeds[2])
    else:
        train_seed = int(seeds[3])
        test_seed = int(seeds[2])

    return train_seed, test_seed


def get_buffer_properties(buffer_name, attack_path, state_dim, action_dim, device, args, seed, env_seed):
    """Loads buffers and returns some buffer properties"""
    logger.info("Retreiving buffer properties...")
    replay_buffer_train = BCQutils.ReplayBuffer(state_dim, action_dim, device)
    replay_buffer_train.load(f"{attack_path}/{env_seed}/{seed}/{args.max_traj_len}/buffers/{buffer_name}")

    num_trajectories = replay_buffer_train.num_trajectories
    start_states = replay_buffer_train.initial_state
    trajectories_end_index = replay_buffer_train.trajectory_end_index

    return num_trajectories, start_states, trajectories_end_index


def create_pairs(
    attack_path, state_dim, action_dim, device, args, label, train_seed, test_seed,
    do_train=True, train_padding_len=0, test_padding_len=0, padding_len=0):

    if not do_train:
        env_seed = args.env_seeds[-1]
    else:
        env_seed = args.env_seeds[args.shadow_seeds.index(test_seed)]

    # Getting training buffer properties
    buffer_name_train = f"{args.buffer_name}_{args.env}_{env_seed}_{train_seed}"
    train_num_trajectories, train_start_states, train_trajectories_end_index = get_buffer_properties(
        buffer_name_train, attack_path, state_dim, action_dim, device, args, train_seed, env_seed)

    # BCQ output
    buffer_name_test = f"target_{args.buffer_name}_{args.env}_{env_seed}_{test_seed}_{args.bcq_max_timesteps}_compatible"
    test_num_trajectories, test_start_states, test_trajectories_end_index = get_buffer_properties(
        buffer_name_test, attack_path, state_dim, action_dim, device, args, test_seed, env_seed)

    if train_num_trajectories != test_num_trajectories and train_seed == test_seed:
        raise ValueError('"in" and "out" buffers do not have same number of trajectories ... ')
    if not np.array_equal(train_start_states, test_start_states) and train_seed == test_seed:
        raise ValueError('the initial states are not the same in the "in" and "out" buffers with same seeds')
    if not np.array_equal(train_start_states[0:10], test_start_states[0:10]):
        raise ValueError('the initial states are not the same in the "in" and "out" buffers')

    if do_train:
        num_trajectories = int(round(min(train_num_trajectories, test_num_trajectories) / args.num_models))
        # Choosing 80% of input trajectories for training and the rest for evaluation
        train_size = int(round(num_trajectories * 0.80))
        # eval_train_size = int(round(num_trajectories - train_size))

    else:

        num_trajectories = train_size = int(min(args.attack_sizes[0], train_num_trajectories, test_num_trajectories))
        # eval_train_size = 0

    test_seq_buffer = np.load(
        f"{attack_path}/{env_seed}/{test_seed}/{args.max_traj_len}/buffers/{buffer_name_test}_action.npy")
    train_seq_buffer = np.load(
        f"{attack_path}/{env_seed}/{train_seed}/{args.max_traj_len}/buffers/{buffer_name_train}_action.npy")

    final_train_dataset, final_eval_dataset = generate_correlated_decorrelated_pairs(
        test_seq_buffer, train_seq_buffer, test_trajectories_end_index, train_trajectories_end_index, train_size,
        num_trajectories, train_start_states, label, do_train, correlation=args.correlation,
        test_padding_len=test_padding_len, train_padding_len=train_padding_len,
        padding_len=padding_len, pairing_mode=args.pairing_mode)

    return final_train_dataset, final_eval_dataset


def generate_correlated_pairs(
    test_seq_buffer, train_seq_buffer, test_trajectories_end_index, train_trajectories_end_index, train_size,
        num_trajectories, train_start_states, label, do_train, correlation=CORRELATED, test_padding_len=None,
        train_padding_len=None, padding_len=None, pairing_mode='horizontal'):
    """Pairing test and train pairs"""
    final_train_dataset = None
    final_train_dataset_label = None
    final_eval_dataset = None
    final_eval_dataset_label = None

    if do_train:
        logger.info(f"generating {CORRELATION_MAP.get(correlation)} pairs...")
    else:
        logger.info(f"generating correlated pairs for prediction...")
    # Pairing the entire training with test in the broadcast fashion
    for j in range(int(num_trajectories)):

        if do_train and j >= train_size:
            if pairing_mode == 'horizontal':
                eval_in_seq = pad_traj(
                    train_seq_buffer[train_trajectories_end_index[j-1] + 1: train_trajectories_end_index[j] + 1, :],
                    train_padding_len)
                eval_out_seq = pad_traj(
                    test_seq_buffer[test_trajectories_end_index[j-1] + 1: test_trajectories_end_index[j] + 1, :],
                    test_padding_len)
            elif pairing_mode == 'vertical':
                eval_in_seq = pad_traj(
                    train_seq_buffer[train_trajectories_end_index[j-1] + 1: train_trajectories_end_index[j] + 1, :],
                    padding_len)
                eval_out_seq = pad_traj(
                    test_seq_buffer[test_trajectories_end_index[j-1] + 1: test_trajectories_end_index[j] + 1, :],
                    padding_len)
            else:
                raise NotImplementedError
            if CORRELATION_MAP.get(correlation) == SEMI_CORRELATED:
                np.random.shuffle(eval_in_seq)
            if pairing_mode == 'horizontal':
                complete_traj_eval_seq = np.concatenate((eval_in_seq, eval_out_seq))
                complete_traj_eval_seq = np.ravel(complete_traj_eval_seq)
            else:
                complete_traj_eval_seq = np.concatenate((eval_in_seq, eval_out_seq), axis=1)
                complete_traj_eval_seq = np.ravel(complete_traj_eval_seq)

            final_eval_dataset_label = np.array([label]) if not isinstance(
                final_eval_dataset_label, np.ndarray) else np.vstack((final_eval_dataset_label, np.array([label])))
            final_eval_dataset = complete_traj_eval_seq if not isinstance(
                final_eval_dataset, np.ndarray) else np.vstack((final_eval_dataset, complete_traj_eval_seq))

        else:
            if pairing_mode == 'horizontal':
                if j == 0:
                    in_seq = pad_traj(train_seq_buffer[0: train_trajectories_end_index[0] + 1, :], train_padding_len)
                    out_seq = pad_traj(test_seq_buffer[0: test_trajectories_end_index[0] + 1, :], test_padding_len)
                else:
                    in_seq = pad_traj(
                        train_seq_buffer[train_trajectories_end_index[j-1] + 1: train_trajectories_end_index[j] + 1, :],
                        train_padding_len)
                    out_seq = pad_traj(
                        test_seq_buffer[test_trajectories_end_index[j-1] + 1: test_trajectories_end_index[j] + 1, :],
                        test_padding_len)
            elif pairing_mode == 'vertical':
                if j == 0:
                    in_seq = pad_traj(train_seq_buffer[0: train_trajectories_end_index[0] + 1, :], padding_len)
                    out_seq = pad_traj(test_seq_buffer[0: test_trajectories_end_index[0] + 1, :], padding_len)
                else:
                    in_seq = pad_traj(
                        train_seq_buffer[train_trajectories_end_index[j-1] + 1: train_trajectories_end_index[j] + 1, :],
                        padding_len)
                    out_seq = pad_traj(
                        test_seq_buffer[test_trajectories_end_index[j-1] + 1: test_trajectories_end_index[j] + 1, :],
                        padding_len)
            else:
                raise ValueError('No padding length defined!')
            if CORRELATION_MAP.get(correlation) == SEMI_CORRELATED:
                np.random.shuffle(in_seq)
            if pairing_mode == 'horizontal':
                complete_traj_seq = np.concatenate((in_seq, out_seq))
                complete_traj_seq = np.ravel(complete_traj_seq)
            else:
                complete_traj_seq = np.concatenate((in_seq, out_seq), axis=1)
                complete_traj_seq = np.ravel(complete_traj_seq)

            final_train_dataset_label = np.array([label]) if not isinstance(
                final_train_dataset_label, np.ndarray) else np.vstack((final_train_dataset_label, np.array([label])))
            final_train_dataset = complete_traj_seq if not isinstance(
                final_train_dataset, np.ndarray) else np.vstack((final_train_dataset, complete_traj_seq))

    if do_train:
        logger.info(f"generating {CORRELATION_MAP.get(correlation)} pairs... Done!")
    else:
        logger.info("generating correlated pairs for prediction... Done!")

    return (final_train_dataset, final_train_dataset_label), (final_eval_dataset, final_eval_dataset_label)


def generate_correlated_decorrelated_pairs(
        test_seq_buffer, train_seq_buffer, test_trajectories_end_index, train_trajectories_end_index, train_size,
        num_trajectories, train_start_states, label, do_train, correlation=CORRELATED, test_padding_len=None,
        train_padding_len=None, padding_len=None, pairing_mode='horizontal'):
    """
    Randomly selects start states, action train/test_seq_buffer, and label
    A trajectory length is set using args.max_traj_len. This value should be the length of the entire
    trajectory.
    """
    final_train_dataset = None
    final_train_dataset_label = None
    final_eval_dataset = None
    final_eval_dataset_label = None

    if do_train:
        logger.info(f"generating {CORRELATION_MAP.get(correlation)} pairs... ")
    else:
        logger.info("generating CORRELATED pairs for prediction... ")

    for j in range(num_trajectories):

        if do_train and j >= train_size:
            if pairing_mode == 'horizontal':
                if correlation == 'DECORRELATED':
                    eval_in_seq = train_seq_buffer[np.random.choice(
                        train_seq_buffer.shape[0], train_padding_len, replace=True), :]
                else:
                    eval_in_seq = pad_traj(
                        train_seq_buffer[train_trajectories_end_index[j - 1] + 1:
                                         train_trajectories_end_index[j] + 1, :], train_padding_len)
                eval_out_seq = pad_traj(
                    test_seq_buffer[test_trajectories_end_index[j-1] + 1: test_trajectories_end_index[j] + 1, :],
                    test_padding_len)
            elif pairing_mode == 'vertical':
                if correlation == 'DECORRELATED':
                    eval_in_seq = train_seq_buffer[np.random.choice(
                        train_seq_buffer.shape[0], padding_len, replace=True), :]
                else:
                    eval_in_seq = pad_traj(
                        train_seq_buffer[train_trajectories_end_index[j - 1] + 1:
                                         train_trajectories_end_index[j] + 1, :], padding_len)
                eval_out_seq = pad_traj(
                    test_seq_buffer[test_trajectories_end_index[j-1] + 1: test_trajectories_end_index[j] + 1, :],
                    padding_len)
            else:
                raise NotImplementedError
            if CORRELATION_MAP.get(correlation) == 'SEMI_CORRELATED':
                np.random.shuffle(eval_in_seq)
            if pairing_mode == 'horizontal':
                complete_traj_eval_seq = np.concatenate((eval_in_seq, eval_out_seq))
                complete_traj_eval_seq = np.ravel(complete_traj_eval_seq)
            else:
                complete_traj_eval_seq = np.concatenate((eval_in_seq, eval_out_seq), axis=1)
                complete_traj_eval_seq = np.ravel(complete_traj_eval_seq)

            final_eval_dataset_label = np.array([label]) if not isinstance(
                final_eval_dataset_label, np.ndarray) else np.vstack((final_eval_dataset_label, np.array([label])))
            final_eval_dataset = complete_traj_eval_seq if not isinstance(
                final_eval_dataset, np.ndarray) else np.vstack((final_eval_dataset, complete_traj_eval_seq))

        else:
            if pairing_mode == 'horizontal':
                if j == 0 and CORRELATION_MAP.get(correlation) == 'DECORRELATED' and do_train:
                    in_seq = train_seq_buffer[np.random.choice(
                        train_seq_buffer.shape[0], train_padding_len, replace=True), :]
                    out_seq = pad_traj(test_seq_buffer[0: test_trajectories_end_index[0] + 1, :], test_padding_len)
                elif j == 0:
                    in_seq = pad_traj(train_seq_buffer[0: train_trajectories_end_index[0] + 1, :], train_padding_len)
                    out_seq = pad_traj(test_seq_buffer[0: test_trajectories_end_index[0] + 1, :], test_padding_len)
                else:
                    if CORRELATION_MAP.get(correlation) == 'DECORRELATED' and do_train:
                        in_seq = train_seq_buffer[np.random.choice(
                            train_seq_buffer.shape[0], train_padding_len, replace=True), :]
                    else:
                        in_seq = pad_traj(
                            train_seq_buffer[train_trajectories_end_index[j - 1] + 1:
                                             train_trajectories_end_index[j] + 1, :], train_padding_len)
                    out_seq = pad_traj(
                        test_seq_buffer[test_trajectories_end_index[j-1] + 1: test_trajectories_end_index[j] + 1, :],
                        test_padding_len)
            elif pairing_mode == 'vertical':
                if j == 0 and CORRELATION_MAP.get(correlation) == 'DECORRELATED' and do_train:
                    in_seq = train_seq_buffer[np.random.choice(
                        train_seq_buffer.shape[0], padding_len, replace=True), :]
                    out_seq = pad_traj(test_seq_buffer[0: test_trajectories_end_index[0] + 1, :], padding_len)
                elif j == 0:
                    in_seq = pad_traj(train_seq_buffer[0: train_trajectories_end_index[0] + 1, :], padding_len)
                    out_seq = pad_traj(test_seq_buffer[0: test_trajectories_end_index[0] + 1, :], padding_len)
                else:
                    if CORRELATION_MAP.get(correlation) == 'DECORRELATED' and do_train:
                        in_seq = train_seq_buffer[np.random.choice(
                            train_seq_buffer.shape[0], padding_len, replace=True), :]
                    else:
                        in_seq = pad_traj(
                            train_seq_buffer[train_trajectories_end_index[j-1] + 1: train_trajectories_end_index[j] + 1, :],
                            padding_len)
                    out_seq = pad_traj(
                        test_seq_buffer[test_trajectories_end_index[j-1] + 1: test_trajectories_end_index[j] + 1, :],
                        padding_len)
            else:
                raise ValueError('No padding length defined!')
            if CORRELATION_MAP.get(correlation) == 'SEMI_CORRELATED' and do_train:
                np.random.shuffle(in_seq)
            if pairing_mode == 'horizontal':
                complete_traj_seq = np.concatenate((in_seq, out_seq))
                complete_traj_seq = np.ravel(complete_traj_seq)
            else:
                complete_traj_seq = np.concatenate((in_seq, out_seq), axis=1)
                complete_traj_seq = np.ravel(complete_traj_seq)

            final_train_dataset_label = np.array([label]) if not isinstance(
                final_train_dataset_label, np.ndarray) else np.vstack((final_train_dataset_label, np.array([label])))
            final_train_dataset = complete_traj_seq if not isinstance(
                final_train_dataset, np.ndarray) else np.vstack((final_train_dataset, complete_traj_seq))

    if do_train:
        logger.info(f"generating {CORRELATION_MAP.get(correlation)} pairs... Done!")
    else:
        logger.info("generating CORRELATED pairs for prediction... Done!")
    return (final_train_dataset, final_train_dataset_label), (final_eval_dataset, final_eval_dataset_label)


def logger_exp(baseline, precision_bl, recall_bl, rmse, accuracy, precision, recall, threshold):
    logger.info("****************************")
    logger.info("Baseline Results:")
    logger.info(f"Accuracy BL: {baseline}")
    logger.info(f"Precision BL: {precision_bl}")
    logger.info(f"Recall BL: {recall_bl}")
    logger.info("****************************")
    logger.info("Attack Classifier Results:")
    for i in range(len(threshold)):
        logger.info(f"Threshold {threshold[i]}:")
        logger.info(f"Accuracy= {accuracy[i]}")
        logger.info(f"Precision= {precision[i]}")
        logger.info(f"Recall= {recall[i]}")
        logger.info(f"Error (gmean)= {rmse[i]}")
        logger.info(30*"-")


def rsme(errors):
    return np.sqrt(gmean(np.square(errors)))


def calc_errors(classifier_predictions, labels_test, threshold, num_predictions):
    errors = []
    for i in range(num_predictions):
        e_i = (labels_test[i] - classifier_predictions[i]) / (labels_test[i] - threshold)
        errors.append(e_i)

    return errors


def baseline_accuracy(labels_test, num_predictions):
    false_positives = 0
    false_negatives = 0
    true_positives = 0
    true_negatives = 0
    for i in range(num_predictions):
        guess = randint(0, 1)

        # if they're the same
        if guess == labels_test[i]:
            if labels_test[i] == 1:
                true_positives += 1
            else:
                true_negatives += 1
        # said out was actually in
        elif guess == 0 and labels_test[i] == 1:
            false_negatives += 1
        elif guess == 1 and labels_test[i] == 0:
            false_positives += 1

    return output_prec_recall(true_positives, true_negatives, false_negatives, false_positives, num_predictions)


def accuracy_report_2(classifier_predictions, labels_test, threshold, num_predictions, results):
    # false_positives = 0
    # false_negatives = 0
    # true_positives = 0
    # true_negatives = 0
    # for i in range(num_predictions):
    #     if classifier_predictions[i] >= threshold[i] and labels_test[i] == 1:
    #         true_positives += 1
    #     elif classifier_predictions[i] < threshold[i] and labels_test[i] == 0:
    #         true_negatives += 1
    #
    #     # false negative (classifier is saying out but labels say in)
    #     elif classifier_predictions[i] < threshold[i] and labels_test[i] == 1:
    #         false_negatives += 1
    #
    #     # false positive (classifier is saying in but labels say out)
    #     elif classifier_predictions[i] >= threshold[i] and labels_test[i] == 0:
    #         false_positives += 1
    # logger.info(f"Threshold = {threshold[i]}: true_positive = {true_positives}, true_negative = {true_negatives}, "
    #             f"false_positive = {false_positives}, false_negative={false_negatives}")
    accuracy = np.zeros(len(threshold))
    precision = np.zeros(len(threshold))
    recall = np.zeros(len(threshold))
    RMSE_e_i = np.zeros(len(threshold))
    mcc = np.zeros(len(threshold))
    f1 = np.zeros(len(threshold))
    for j in range(len(threshold)):
        false_positives = 0
        false_negatives = 0
        true_positives = 0
        true_negatives = 0
        for i in range(num_predictions):
            if classifier_predictions[i] >= threshold[j] and labels_test[i] == 1:
                true_positives += 1
            elif classifier_predictions[i] < threshold[j] and labels_test[i] == 0:
                true_negatives += 1

            # false negative (classifier is saying out but labels say in)
            elif classifier_predictions[i] < threshold[j] and labels_test[i] == 1:
                false_negatives += 1

            # false positive (classifier is saying in but labels say out)
            elif classifier_predictions[i] >= threshold[j] and labels_test[i] == 0:
                false_positives += 1
        logger.info(
            f"Threshold = {threshold[j]}: true_positive = {true_positives}, true_negative = {true_negatives}, "
            f"false_positive = {false_positives}, false_negative={false_negatives}")

        accuracy[j], precision[j], recall[j], mcc[j], f1[j] = output_prec_recall(true_positives, true_negatives,
                                                                                 false_negatives, false_positives,
                                                                                 num_predictions)
        RMSE_e_i[j] = rsme(calc_errors(classifier_predictions, labels_test, threshold[j], num_predictions))

        results = f"{results}\nThreshold = {threshold[j]}: true_positive = {true_positives}, " \
                  f"true_negative = {true_negatives}, false_positive = {false_positives}, " \
                  f"false_negative={false_negatives}, MCC = {mcc[j]}, F1 = {f1[j]}"
    return accuracy, precision, recall, RMSE_e_i, results

def accuracy_report(classifier_predictions, labels_test, threshold, num_predictions):
    # false_positives = 0
    # false_negatives = 0
    # true_positives = 0
    # true_negatives = 0
    accuracy = np.zeros(len(threshold))
    precision = np.zeros(len(threshold))
    recall = np.zeros(len(threshold))
    RMSE_e_i = np.zeros(len(threshold))
    for j in range(len(threshold)):
        false_positives = 0
        false_negatives = 0
        true_positives = 0
        true_negatives = 0
        for i in range(num_predictions):
            if classifier_predictions[i] >= threshold[j] and labels_test[i] == 1:
                true_positives += 1
            elif classifier_predictions[i] < threshold[j] and labels_test[i] == 0:
                true_negatives += 1

            # false negative (classifier is saying out but labels say in)
            elif classifier_predictions[i] < threshold[j] and labels_test[i] == 1:
                false_negatives += 1

            # false positive (classifier is saying in but labels say out)
            elif classifier_predictions[i] >= threshold[j] and labels_test[i] == 0:
                false_positives += 1
        logger.info(
            f"Threshold = {threshold[j]}: true_positive = {true_positives}, true_negative = {true_negatives}, "
            f"false_positive = {false_positives}, false_negative={false_negatives}")
        accuracy[j], precision[j], recall[j], mcc[j] = output_prec_recall(
            true_positives, true_negatives, false_negatives, false_positives, num_predictions)
        RMSE_e_i[j] = rsme(calc_errors(classifier_predictions, labels_test, threshold[j], num_predictions))
    # for i in range(num_predictions):
    #     if classifier_predictions[i] >= threshold and labels_test[i] == 1:
    #         true_positives += 1
    #     elif classifier_predictions[i] < threshold and labels_test[i] == 0:
    #         true_negatives += 1
    #
    #     # false negative (classifier is saying out but labels say in)
    #     elif classifier_predictions[i] < threshold and labels_test[i] == 1:
    #         false_negatives += 1
    #
    #     # false positive (classifier is saying in but labels say out)
    #     elif classifier_predictions[i] >= threshold and labels_test[i] == 0:
    #         false_positives += 1
    # logger.info(
    #     f"true_positive={true_positives}, true_negative={true_negatives}, false_positive={false_positives}"
    #     f", false_negative={false_negatives}")
    # return output_prec_recall(true_positives, true_negatives, false_negatives, false_positives, num_predictions)
    return accuracy, precision, recall, RMSE_e_i


def output_prec_recall(tp, tn, fn, fp, total):
    num_correct = tp + tn

    acc = num_correct / total
    if (tp + fp) == 0:
        prec = -1
    else:
        prec = tp / (tp + fp)
    if (tp + fn) == 0:
        recall = -1
    else:
        recall = tp / (tp + fn)

    if (tn + fp) != 0 and (tn + fn) != 0 and (tp + fn) != 0 and (tp +fp) != 0:
        informed = recall + tn / (tn + fp) - 1
        marked = prec + tn / (tn + fn) - 1
        # mcc = np.sqrt(informed * marked)
        mcc = (tp * tn - fp * fn) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    else:
        mcc = tp * tn - fp * fn

    if prec + recall == 0:
        f1 = -1
    else:
        f1 = 2 * prec * recall / (prec + recall)

    return round(acc, 3), round(prec, 3), round(recall, 3), round(mcc, 3), round(f1, 3)


def generate_metrics(classifier_predictions, labels_test, threshold, num_predictions):
    accuracy, precision, recall, RMSE_e_i = accuracy_report(
        classifier_predictions, labels_test, threshold, num_predictions)
    
    # accuracy_bl, precision_bl, recall_bl = baseline_accuracy(labels_test, num_predictions)
    # RMSE_e_i = rsme(calc_errors(classifier_predictions, labels_test, threshold, num_predictions))

    # logger_exp(accuracy_bl, precision_bl, recall_bl, RMSE_e_i, accuracy, precision, recall, threshold)
    return true_positive, true_negative, false_positive, false_negative


def modelfit(alg, attack_train_eval_x, attack_train_eval_y, useTrainCV=True, cv_folds=5, early_stopping_rounds=10):
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(attack_train_eval_x, label=attack_train_eval_y)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
                          metrics={'mae'}, early_stopping_rounds=early_stopping_rounds, verbose_eval=False)
        alg.set_params(n_estimators=cvresult.shape[0])

    # Fit the algorithm on the data
    alg.fit(attack_train_eval_x, attack_train_eval_y, eval_metric='mae')

    # Predict training set:
    dtrain_predictions = alg.predict(attack_train_eval_x)
    # dtrain_predictions = [0 if val < t else 1 for val in dtrain_predictions]
    dtrain_predprob = alg.predict_proba(attack_train_eval_x)[:, 1]
    # dtrain_predprob = [0 if val < t else 1 for val in dtrain_predprob]



    # Print model report:
    logger.info("Model Report:")
    logger.info("Accuracy : %.4g" % metrics.accuracy_score(attack_train_eval_y, dtrain_predictions))
    logger.info("AUC Score (Train): %f" % metrics.roc_auc_score(attack_train_eval_y, dtrain_predprob))
    logger.info(
        f"Tuned n_estimators for learning_rate {alg.get_params()['learning_rate']} = {alg.get_params()['n_estimators']}")

    return alg.get_params()['n_estimators']

    # feat_imp = pd.Series(alg.get_booster().get_fscore()).sort_values(ascending=False)
    # feat_imp.plot(kind='bar', title='Feature Importances')
    # plt.ylabel('Feature Importance Score')
    # # plt.show()


def train_classifier(xgb1, xgb_train, xgb_eval, early_stopping_rounds=10, num_round=1000, eta=0.2):

    param = {'learning_rate': xgb1.get_params()['learning_rate'],
             'n_estimators': num_round,
             'max_depth': xgb1.get_params()['max_depth'],
             'min_child_weight': xgb1.get_params()['min_child_weight'],
             'gamma': xgb1.get_params()['gamma'],
             'subsample': xgb1.get_params()['subsample'],
             'colsample_bytree': xgb1.get_params()['colsample_bytree'],
             'objective': 'reg:logistic',
             'nthread': 4,
             'scale_pos_weight': 1,
             'seed': 27,
             'eval_metric': 'mae',
             'n_jobs': -1}

    watch_list = [(xgb_train, 'train'), (xgb_eval, 'eval')]
    evals_result = {}
    logger.info("training classifier")
    callbacks = [log_eval(20, True)]
    return xgb.train(param, xgb_train, num_round, watch_list, early_stopping_rounds=early_stopping_rounds,
                     evals_result=evals_result, callbacks=callbacks)

def log_eval(period=1, show_stdv=True):
    """Create a callback that logs evaluation result with logger.

    Parameters
    ----------
    period : int
        The period to log the evaluation results

    show_stdv : bool, optional
         Whether show stdv if provided

    Returns
    -------
    callback : function
        A callback that logs evaluation every period iterations into logger.
    """

    def _fmt_metric(value, show_stdv=True):
        """format metric string"""
        if len(value) == 2:
            return '%s:%g' % (value[0], value[1])
        elif len(value) == 3:
            if show_stdv:
                return '%s:%g+%g' % (value[0], value[1], value[2])
            else:
                return '%s:%g' % (value[0], value[1])
        else:
            raise ValueError("wrong metric value")

    def callback(env):
        if env.rank != 0 or len(env.evaluation_result_list) == 0 or period is False:
            return
        i = env.iteration
        if i % period == 0 or i + 1 == env.begin_iteration or i + 1 == env.end_iteration:
            msg = '\t'.join([_fmt_metric(x, show_stdv) for x in env.evaluation_result_list])
            logger.info('[%d]\t%s\n' % (i, msg))

    return callback


def train_attack_model_v4(file_path_results, pair_path_results, args):

    logger.info("loading the train/eval pairs ...")
    attack_train_data_x = np.load(pair_path_results + '/train_x.npy')
    attack_train_data_y = np.load(pair_path_results + '/train_y.npy')
    attack_eval_data_x = np.load(pair_path_results + '/eval_x.npy')
    attack_eval_data_y = np.load(pair_path_results + '/eval_y.npy')

    attack_train_eval_x = np.vstack((attack_train_data_x, attack_eval_data_x))
    attack_train_eval_y = np.vstack((attack_train_data_y, attack_eval_data_y))
    attack_train_eval_y = np.ravel(attack_train_eval_y)
    logger.info("loading the train/eval pairs ... Done")
    logger.info("Setting up the xgb properties ...")
    xgb1 = XGBClassifier(
        learning_rate=args.xg_eta,
        n_estimators=args.xgb_n_rounds,
        max_depth=args.max_depth,
        min_child_weight=args.min_child_weight,
        gamma=args.gamma,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        reg_alpha=args.reg_alpha,
        objective='reg:logistic',
        nthread=4,
        scale_pos_weight=1,
        seed=27,
        use_label_encoder=False
    )

    results = ""

    if args.cv_tune_xgb:

        modelfit(xgb1, attack_train_eval_x, attack_train_eval_y, early_stopping_rounds=args.early_stopping_rounds)

        if args. max_depth_vector or args.min_child_weight_vector:

            param_test1 = {
                'max_depth': args.max_depth_vector,
                'min_child_weight': args.min_child_weight_vector
            }
            gsearch1 = GridSearchCV(
                estimator=xgb1, param_grid=param_test1, scoring='neg_mean_absolute_error', cv=5)

            gsearch1.fit(attack_train_eval_x, attack_train_eval_y)
            # logger.info(gsearch1.cv_results_)
            logger.info(f"best parameter: {gsearch1.best_params_}")
            logger.info(f"best score: {gsearch1.best_score_}")

        # param_test2 = {
        #     'max_depth': range(3, 20, 2),
        #     'min_child_weight': range(2, 10, 2)
        # }
        # gsearch2 = GridSearchCV(
        #     estimator=xgb1, param_grid=param_test2, scoring='neg_mean_absolute_error', n_jobs=4, cv=5)
        # gsearch2.fit(attack_train_eval_x, attack_train_eval_y)
        # logger.info(f"best parameter: {gsearch2.best_params_}")
        # logger.info(f"best score: {gsearch2.best_score_}")

        # modelfit(gsearch1.best_estimator_, attack_train_eval_x, attack_train_eval_y, t)
        # xgb1.set_params(max_depth=gsearch1.best_params_['max_depth'] if gsearch1.best_score_ >= gsearch2.best_score_
        # else gsearch2.best_params_['max_depth'], min_child_weight=gsearch1.best_params_['min_child_weight']
        # if gsearch1.best_score_ >= gsearch2.best_score_ else gsearch2.best_params_['min_child_weight'])

            xgb1.set_params(max_depth=gsearch1.best_params_['max_depth'],
                            min_child_weight=gsearch1.best_params_['min_child_weight'])
        if args.gamma_vector:

            param_test3 = {
                'gamma': args.gamma_vector
            }

            gsearch3 = GridSearchCV(
                estimator=xgb1, param_grid = param_test3, scoring='neg_mean_absolute_error', cv=5)
            gsearch3.fit(attack_train_eval_x, attack_train_eval_y)
            logger.info(f"best parameter: {gsearch3.best_params_}")
            logger.info(f"best score: {gsearch3.best_score_}")
            xgb1.set_params(gamma=gsearch3.best_params_['gamma'])

        if (args.max_depth_vector or args.min_child_weight_vector or args.gamma_vector) and \
                (args.subsample_vector or args.colsample_bytree_vector or args.reg_alpha_vector):

            xgb1.set_params(n_estimators=args.xgb_n_rounds)
            modelfit(xgb1, attack_train_eval_x, attack_train_eval_y, early_stopping_rounds=args.early_stopping_rounds)

        if args.subsample_vector or args.colsample_bytree_vector:
            param_test4 = {
                'subsample': args.subsample_vector,
                'colsample_bytree': args.colsample_bytree_vector
            }

            gsearch4 = GridSearchCV(estimator=xgb1, param_grid=param_test4,
                                    scoring='neg_mean_absolute_error', cv=5)
            gsearch4.fit(attack_train_eval_x, attack_train_eval_y)
            logger.info(f"best parameter: {gsearch4.best_params_}")
            logger.info(f"best score: {gsearch4.best_score_}")

            xgb1.set_params(subsample=gsearch4.best_params_['subsample'],
                            colsample_bytree=gsearch4.best_params_['colsample_bytree'])
        if args.reg_alpha_vector:
            param_test5 = {
                'reg_alpha': args.reg_alpha_vector
            }
            gsearch5 = GridSearchCV(estimator=xgb1, param_grid = param_test5,
                                    scoring='neg_mean_absolute_error', cv=5)
            gsearch5.fit(attack_train_eval_x, attack_train_eval_y)
            logger.info(f"best parameter: {gsearch5.best_params_}")
            logger.info(f"best score: {gsearch5.best_score_}")

            xgb1.set_params(reg_alpha=gsearch5.best_params_['reg_alpha'])

        xgb1.set_params(n_estimators=args.xgb_n_rounds)
        # modelfit(xgb1, attack_train_eval_x, attack_train_eval_y)

    classifier_train_data = xgb.DMatrix(attack_train_data_x, attack_train_data_y)
    classifier_eval_data = xgb.DMatrix(attack_eval_data_x, attack_eval_data_y)

    logger.info("classifier training ...")
    attack_classifier = train_classifier(xgb1, classifier_train_data, classifier_eval_data,
                                         early_stopping_rounds=args.early_stopping_rounds,
                                         num_round=args.xgb_n_rounds, eta=args.xg_eta)

    logger.info("training finished ...")
    logger.info("loading the test pairs ...")

    attack_test_data_x = np.load(pair_path_results + '/test_x.npy')
    attack_test_data_y = np.load(pair_path_results + '/test_y.npy')

    classifier_test_data = xgb.DMatrix(attack_test_data_x, attack_test_data_y)

    logger.info("predicting ...")

    # prediction phase using the trained attack classifier
    classifier_predictions = attack_classifier.predict(classifier_test_data)
    logger.info("predicting ... Done")
    # NOTE: the number of predictions cannot be more than then number of rows in attack_test_data_x
    # Adjusting num_predictions accordingly
    num_rows, num_columns = attack_test_data_x.shape
    num_predictions = args.attack_sizes[0] if args.attack_sizes[0] <= num_rows else num_rows
    _, _, _, _, results = accuracy_report_2(
        classifier_predictions, attack_test_data_y, args.attack_thresholds, num_predictions, results)

    logger.info(f"Final tuned parameters:\n {xgb1}")

    num_training_samples = attack_train_data_x.shape[0]
    num_eval_samples = attack_eval_data_x.shape[0]

    print_experiment(args.env, args.shadow_seeds, args.target_seeds, args.attack_thresholds,
                     num_predictions, args.max_traj_len, args.num_models, num_training_samples, num_eval_samples)

    logger.info(results)


def get_pairs_max_traj_len(attack_path, file_path_results, state_dim, action_dim, device, args):
    """
    Let's get the maximum length for both positive/negative test/train trajectories.
    This is done for padding purposes.
    """
    logger.info("getting maximum trajectories length...")
    train_traj_lens = []
    test_traj_lens = []
    train_test_seeds = []
    # max_train_traj_len = []
    for label in [0, 1]:
        for i in range(args.num_models):
            train_test_seeds.append(get_seeds_pairs(label, args.shadow_seeds, index=i, test=False))
        train_test_seeds.append(get_seeds_pairs(label, args.target_seeds, test=True))

    for train_seed, test_seed in train_test_seeds:
        # if CORRELATION_MAP.get(args.correlation) != DECORRELATED:
        # loading buffers to get trajectories lengths
        if (train_seed in args.target_seeds) or (test_seed in args.target_seeds):
            env_seed = args.env_seeds[-1]
        else:
            env_seed = args.env_seeds[args.shadow_seeds.index(test_seed)]

        buffer_name_train = f"{args.buffer_name}_{args.env}_{env_seed}_{train_seed}"
        _, _, train_trajectories_end_index = get_buffer_properties(
        buffer_name_train, attack_path, state_dim, action_dim, device, args, train_seed, env_seed)

        # Maximum trajectory length is calculated for padding purposes
        train_traj_lens.append(compute_max_trajectory_length(train_trajectories_end_index))

        # BCQ output
        buffer_name_test = f"target_{args.buffer_name}_{args.env}_{env_seed}_{test_seed}_{args.bcq_max_timesteps}_compatible"
        _, _, test_trajectories_end_index = get_buffer_properties(
            buffer_name_test , attack_path, state_dim, action_dim, device, args, test_seed, env_seed)
        
        # Maximum trajectory length is calculated for padding purposes
        test_traj_lens.append(compute_max_trajectory_length(test_trajectories_end_index))

    # if CORRELATION_MAP.get(args.correlation) != DECORRELATED:
    #     max_train_traj_len = max(train_traj_lens)

    return max(test_traj_lens), max(train_traj_lens)


def shuffle_xgboost_params(attack_train_data_x, attack_train_data_y):
    """
    Merges X and Y data horizentally, then shuffles the results,
    and finally splits the lable column
    """
    merged_data = np.hstack((attack_train_data_x, attack_train_data_y))
    np.random.shuffle(merged_data)
    return np.hsplit(merged_data, np.array([-1]))


def train_attack_model_v3(attack_path, file_path_results, pair_path_results, state_dim, action_dim, device, args):

    # if CORRELATION_MAP.get(args.correlation) != DECORRELATED:
    # In correlated mode, we need to load existing trajectories, and find their maximum length
    # test_padding_len = train_padding_len = args.max_traj_len
    test_padding_len, train_padding_len = get_pairs_max_traj_len(
        attack_path, file_path_results, state_dim, action_dim, device, args)
    padding_len = max(test_padding_len, train_padding_len)
    # Pairing train and test trajectories
    # Feeding max length trajectory to be uesd for padding purposes
    attack_train_pos_data = None
    attack_train_pos_label = None
    attack_eval_pos_data = None
    attack_eval_pos_label = None
    attack_train_neg_data = None
    attack_train_neg_label = None
    attack_eval_neg_data = None
    attack_eval_neg_label = None
    for i in range(args.num_models):
        # Positive pairs
        train_seed, test_seed = get_seeds_pairs(1, args.shadow_seeds, index=i, test=False)
        attack_train_positive_data, attack_eval_positive_data = create_pairs(
            attack_path, state_dim, action_dim, device, args, 1,
            train_seed, test_seed,
            do_train=True,
            test_padding_len=test_padding_len,
            train_padding_len=train_padding_len,
            padding_len=padding_len
        )
        train_pos_data, train_pos_label = attack_train_positive_data
        eval_pos_data, eval_pos_label = attack_eval_positive_data
        attack_train_pos_data = train_pos_data if not isinstance(
                attack_train_pos_data, np.ndarray) else np.vstack((attack_train_pos_data, train_pos_data))
        attack_train_pos_label = train_pos_label if not isinstance(
            attack_train_pos_label, np.ndarray) else np.vstack((attack_train_pos_label, train_pos_label))
        attack_eval_pos_data = eval_pos_data if not isinstance(
                attack_eval_pos_data, np.ndarray) else np.vstack((attack_eval_pos_data, eval_pos_data))
        attack_eval_pos_label = eval_pos_label if not isinstance(
            attack_eval_pos_label, np.ndarray) else np.vstack((attack_eval_pos_label, eval_pos_label))
        # Negative pairs
        train_seed, test_seed = get_seeds_pairs(0, args.shadow_seeds, index=i, test=False)
        attack_train_negative_data, attack_eval_negative_data = create_pairs(
            attack_path, state_dim, action_dim, device, args, 0,
            train_seed, test_seed,
            do_train=True,
            test_padding_len=test_padding_len,
            train_padding_len=train_padding_len,
            padding_len=padding_len
        )
        train_neg_data, train_neg_label = attack_train_negative_data
        eval_neg_data, eval_neg_label = attack_eval_negative_data
        attack_train_neg_data = train_neg_data if not isinstance(
                attack_train_neg_data, np.ndarray) else np.vstack((attack_train_neg_data, train_neg_data))
        attack_train_neg_label = train_neg_label if not isinstance(
            attack_train_neg_label, np.ndarray) else np.vstack((attack_train_neg_label, train_neg_label))
        attack_eval_neg_data = eval_neg_data if not isinstance(
                attack_eval_neg_data, np.ndarray) else np.vstack((attack_eval_neg_data, eval_neg_data))
        attack_eval_neg_label = eval_neg_label if not isinstance(
            attack_eval_neg_label, np.ndarray) else np.vstack((attack_eval_neg_label, eval_neg_label))

    # Instanciating xgboost DMatrix with positive/negative train data
    # attack_train_pos_data, attack_train_pos_label = attack_train_positive_data
    # attack_train_neg_data, attack_train_neg_label = attack_train_negative_data
    attack_train_data_x = np.vstack((attack_train_pos_data, attack_train_neg_data))
    attack_train_data_y = np.vstack((attack_train_pos_label, attack_train_neg_label))
    attack_train_data_x, attack_train_data_y = shuffle_xgboost_params(attack_train_data_x, attack_train_data_y)
    attack_train_data_x, attack_train_data_y = shuffle_xgboost_params(attack_train_data_x, attack_train_data_y)

    np.save(pair_path_results + '/train_x', attack_train_data_x)
    np.save(pair_path_results + '/train_y', attack_train_data_y)
    logger.info("saving train data for classifier training ... Done")


    # Instanciating xgboost DMatrix with positive/negative train data
    # attack_eval_pos_data, attack_eval_pos_label = attack_eval_positive_data
    # attack_eval_neg_data, attack_eval_neg_label = attack_eval_negative_data
    attack_eval_data_x = np.vstack((attack_eval_pos_data, attack_eval_neg_data))
    attack_eval_data_y = np.vstack((attack_eval_pos_label, attack_eval_neg_label))
    attack_eval_data_x, attack_eval_data_y = shuffle_xgboost_params(attack_eval_data_x, attack_eval_data_y)
    attack_eval_data_x, attack_eval_data_y = shuffle_xgboost_params(attack_eval_data_x, attack_eval_data_y)

    np.save(pair_path_results + '/eval_x', attack_eval_data_x)
    np.save(pair_path_results + '/eval_y', attack_eval_data_y)
    logger.info("saving eval data for classifier training ... Done")

    # Positive pairs
    train_seed, test_seed = get_seeds_pairs(1, args.target_seeds, test=True)
    attack_train_positive_data, _ = create_pairs(
        attack_path, state_dim, action_dim, device, args, 1,
        train_seed, test_seed,
        do_train=False,
        test_padding_len=test_padding_len,
        train_padding_len=train_padding_len,
        padding_len=padding_len
    )
    # Negative pairs
    train_seed, test_seed = get_seeds_pairs(0, args.target_seeds, test=True)
    attack_train_negative_data, _ = create_pairs(
        attack_path, state_dim, action_dim, device, args, 0,
        train_seed, test_seed,
        do_train=False,
        test_padding_len=test_padding_len,
        train_padding_len=train_padding_len,
        padding_len=padding_len
    )

    final_train_dataset_pos, final_train_dataset_pos_label = attack_train_positive_data
    final_train_dataset_neg, final_train_dataset_neg_label = attack_train_negative_data
    attack_test_data_x = np.vstack((final_train_dataset_pos, final_train_dataset_neg))
    attack_test_data_y = np.vstack((final_train_dataset_pos_label, final_train_dataset_neg_label))
    attack_test_data_x, attack_test_data_y = shuffle_xgboost_params(attack_test_data_x, attack_test_data_y)
    attack_test_data_x, attack_test_data_y = shuffle_xgboost_params(attack_test_data_x, attack_test_data_y)

    np.save(pair_path_results + '/test_x', attack_test_data_x)
    np.save(pair_path_results + '/test_y', attack_test_data_y)
    logger.info("saving test data for prediction ... Done")


