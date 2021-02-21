import gc
import math
import os
import uuid
from random import randint
import BCQutils
import BCQ
import pathlib

import numpy as np
import xgboost as xgb
from scipy.stats.mstats import gmean

from pandas import DataFrame


from utils.helpers import cleanup, print_experiment, generate_pairs, get_models


def get_trajectory(seed, index, trajectory_length):
    path = "tmp/"
    npy_train = path + str(seed) + '_' + str(trajectory_length) + '.npy'

    return np.load(npy_train, 'r', allow_pickle=True)[index]


def get_trajectory_test(seed, index, trajectory_length):
    path = "tmp/"
    npy_test = path + str(seed) + '_' + str(trajectory_length) + '_test.npy'

    return np.load(npy_test, 'r', allow_pickle=True)[index]


def create_train_pairs(attack_path, state_dim, action_dim, max_action, device, args, label):
    #To create trajectory pairs
    if label:
        seed = int(args.seed[0])
    else:
        seed = int(args.seed[1])

    if not os.path.exists(f"./{attack_path}/{seed}/attack_outputs/traj_based_buffers/"):
        os.makedirs(f"./{attack_path}/{seed}/attack_outputs/traj_based_buffers/")

    print("creating input-output pairs...")
    # Load buffer
    setting = f"{args.env}_{seed}"
    buffer_name_train = f"{args.buffer_name}_{setting}"
    buffer_name_evidence = f"target_{args.buffer_name}_{setting}"
    buffer_name_attack_train_pairs = f"attack_train_{args.buffer_name}_{setting}"

    final_train_dataset = []

    final_eval_dataset = []

    print("loading train trajectories...")
    replay_buffer_train = BCQutils.ReplayBuffer(state_dim, action_dim, device)
    replay_buffer_train.load(f"./{attack_path}/{seed}/buffers/{buffer_name_train}")
    print("creating index set from not-done array in training set...")

    print("loading test trajectories...")
    replay_buffer_test = BCQutils.ReplayBuffer(state_dim, action_dim, device)
    replay_buffer_test.load(f"./{attack_path}/{seed}/buffers/{buffer_name_evidence}")
    print("creating index set from not-done array in test set...")

    train_num_trajectories = replay_buffer_train.num_trajectories
    train_start_states = replay_buffer_train.initial_state
    train_trajectories_end_index = replay_buffer_train.trajectory_end_index

    test_num_trajectories = replay_buffer_test.num_trajectories
    test_start_states = replay_buffer_test.initial_state
    test_trajectories_end_index = replay_buffer_test.trajectory_end_index

    if args.out_traj_size < test_num_trajectories:
        test_num_trajectories = args.out_traj_size

    #Choosing 80% of input trajectories for training and the rest of evaluation
    train_size = math.floor(train_num_trajectories / (10 / 8))
    eval_train_size = train_num_trajectories - train_size

    # Choosing 80% of output trajectories for training and the rest of evaluation
    test_size = math.floor(test_num_trajectories / (10 / 8))
    eval_test_size = test_num_trajectories - test_size

    print(f"creating training_pairs...")
    # Pairing the entire training with test in the broadcast fashion
    for j in range(test_size):
        #Pairing the entire train set with the j-th test trajectory
        temp_sequence = []
        for i in range(train_size):
            temp_sequence.extend([train_start_states[i]])
            temp_sequence.extend([test_start_states[j]])
            m = 2*len(temp_sequence[0])
            temp_sequence = np.concatenate(temp_sequence, axis=0).reshape(m, 1)
            if i == 0:
                temp_arr = np.ravel(np.load(f"./{attack_path}/{seed}/buffers/{buffer_name_train}_action.npy")[
                                                                        0:train_trajectories_end_index[i]:1])
                temp_arr = temp_arr.reshape(len(temp_arr), 1)
                temp_sequence = np.concatenate((temp_arr, temp_sequence), axis=0)
            else:
                temp_arr = np.ravel(np.load(f"./{attack_path}/{seed}/buffers/{buffer_name_train}_action.npy")
                                    [train_trajectories_end_index[i-1]:train_trajectories_end_index[i]:1])
                temp_arr = temp_arr.reshape(len(temp_arr), 1)
                temp_sequence = np.concatenate((temp_arr, temp_sequence), axis=0)
            if j == 0:
                temp_arr = np.ravel(np.load(f"./{attack_path}/{seed}/buffers/{buffer_name_evidence}_action.npy")[0:
                                                                                test_trajectories_end_index[j]:1])
                temp_arr = temp_arr.reshape(len(temp_arr), 1)
                temp_sequence = np.concatenate((temp_arr, temp_sequence), axis=0)
            else:
                temp_arr = np.ravel(np.load(f"./{attack_path}/{seed}/buffers/{buffer_name_evidence}_action.npy")
                                    [test_trajectories_end_index[j-1]:test_trajectories_end_index[j]:1])
                temp_arr = temp_arr.reshape(len(temp_arr), 1)
                temp_sequence = np.concatenate((temp_arr, temp_sequence), axis=0)

            temp_sequence = np.concatenate((temp_sequence, np.reshape(label, (1, 1))), axis=0)
            with open(f"./{attack_path}/{seed}/attack_outputs/traj_based_buffers/train_{args.out_traj_size}.npy", 'ab')\
                    as f:
                np.save(f, temp_sequence)
            final_train_dataset.append(temp_sequence)
            temp_sequence = []
    print(f"Done creating training_pairs!")

    # Pairing the eval training with test in the broadcast fashion
    print(f"creating eval_pairs...")
    # Pairing the entire training with test in the broadcast fashion
    for j in range(eval_test_size):
        # Pairing the entire train set with the j-th test trajectory
        temp_sequence = []
        for i in range(eval_train_size):
            temp_sequence.extend([train_start_states[i+train_size-1]])
            temp_sequence.extend([test_start_states[j+test_size-1]])
            m = 2 * len(temp_sequence[0])
            temp_sequence = np.concatenate(temp_sequence, axis=0).reshape(m, 1)
            if i == 0:
                temp_arr = np.ravel(np.load(f"./{attack_path}/{seed}/buffers/{buffer_name_train}_action.npy")[0:
                                                                             train_trajectories_end_index[i]:1])
                temp_arr = temp_arr.reshape(len(temp_arr), 1)
                temp_sequence = np.concatenate((temp_arr, temp_sequence), axis=0)
            else:
                temp_arr = np.ravel(np.load(f"./{attack_path}/{seed}/buffers/{buffer_name_train}_action.npy")[
                                     train_trajectories_end_index[i+train_size - 1]:train_trajectories_end_index
                                     [i+train_size]:1])
                temp_arr = temp_arr.reshape(len(temp_arr), 1)
                temp_sequence = np.concatenate((temp_arr, temp_sequence), axis=0)
            if j == 0:
                temp_arr = np.ravel(np.load(f"./{attack_path}/{seed}/buffers/{buffer_name_evidence}_action.npy")[0:
                                                                                 test_trajectories_end_index[j]:1])
                temp_arr = temp_arr.reshape(len(temp_arr), 1)
                temp_sequence = np.concatenate((temp_arr, temp_sequence), axis=0)
            else:
                temp_arr = np.ravel(np.load(f"./{attack_path}/{seed}/buffers/{buffer_name_evidence}_action.npy")[
                                     test_trajectories_end_index[j+test_size - 1]:test_trajectories_end_index[j+
                                                                                                    test_size]:1])
                temp_arr = temp_arr.reshape(len(temp_arr), 1)
                temp_sequence = np.concatenate((temp_arr, temp_sequence), axis=0)

            temp_sequence = np.concatenate((temp_sequence, np.reshape(label, (1, 1))), axis=0)
            with open(f"./{attack_path}/{seed}/attack_outputs/traj_based_buffers/eval_{args.out_traj_size}.npy", 'ab') as f:
                np.save(f, temp_sequence)
            final_eval_dataset.append(temp_sequence)
            temp_sequence = []
    print(f"Done creating eval_pairs!")

    return final_train_dataset, final_eval_dataset

#starting the attack test sequence genration
def create_test_pairs(attack_path, state_dim, action_dim, max_action, device, args, label):

    target_positive_seed = int(args.seed[2])
    target_negative_seed = int(args.seed[3])

    if label:
        seed = int(args.seed[0])
    else:
        seed = int(args.seed[1])

    if not os.path.exists(f"./{attack_path}/{seed}/attack_outputs/traj_based_buffers/"):
        os.makedirs(f"./{attack_path}/{seed}/attack_outputs/traj_based_buffers/")

    #To create trajectory pairs
    print("creating input-output pairs...")
    # Load buffer
    setting = f"{args.env}_{seed}"
    buffer_name_input = f"private_input_{args.buffer_name}_{setting}"
    buffer_name_output = f"private_output_{args.buffer_name}_{setting}"

    final_prediction_test_dataset = []

    print("loading input trajectories...")
    replay_buffer_input = BCQutils.ReplayBuffer(state_dim, action_dim, device)
    replay_buffer_input.load(f"./{attack_path}/{args.seed}/buffers/{buffer_name_input}")
    print("creating index set from not-done array in training set")

    print("loading output trajectories...")
    replay_buffer_output = BCQutils.ReplayBuffer(state_dim, action_dim, device)
    replay_buffer_output.load(f"./{attack_path}/{args.seed}/buffers/{buffer_name_output}")
    print("creating index set from not-done array in test set")

    input_num_trajectories = replay_buffer_input.num_trajectories
    input_start_states = replay_buffer_input.initial_state
    input_trajectories_end_index = replay_buffer_input.trajectory_end_index

    output_num_trajectories = replay_buffer_output.num_trajectories
    output_start_states = replay_buffer_output.initial_state
    output_trajectories_end_index = replay_buffer_output.trajectory_end_index

    print(f"creating test pairs...")
    # Pairing the entire training with test in the broadcast fashion
    for j in range(output_num_trajectories):
        #Pairing the entire train set with the j-th test trajectory
        temp_sequence = []
        for i in range(input_num_trajectories):
            temp_sequence.extend([input_start_states[i]])
            temp_sequence.extend([output_start_states[j]])
            m = 2 * len(temp_sequence[0])
            temp_sequence = np.concatenate(temp_sequence, axis=0).reshape(m, 1)
            if i == 0:
                temp_arr = np.ravel(np.load(f"./{attack_path}/{seed}/buffers/{buffer_name_input}_action.npy")
                                    [0:input_trajectories_end_index[i]:1])

                temp_arr = temp_arr.reshape(len(temp_arr), 1)
                temp_sequence = np.concatenate((temp_arr, temp_sequence), axis=0)
            else:
                temp_arr = np.ravel(np.load(f"./{attack_path}/{seed}/buffers/{buffer_name_input}_action.npy")
                                    [input_trajectories_end_index[i-1]:input_trajectories_end_index[i]:1])
                temp_arr = temp_arr.reshape(len(temp_arr), 1)
                temp_sequence = np.concatenate((temp_arr, temp_sequence), axis=0)
            if j == 0:
                temp_arr = np.ravel(np.load(f"./{attack_path}/{seed}/buffers/{buffer_name_output}_action.npy")
                                    [0:output_trajectories_end_index[j]:1])
                temp_arr = temp_arr.reshape(len(temp_arr), 1)
                temp_sequence = np.concatenate((temp_arr, temp_sequence), axis=0)
            else:
                temp_arr = np.ravel(np.load(f"./{attack_path}/{seed}/buffers/{buffer_name_output}_action.npy")
                                    [output_trajectories_end_index[j-1]:output_trajectories_end_index[j]:1])
                temp_arr = temp_arr.reshape(len(temp_arr), 1)
                temp_sequence = np.concatenate((temp_arr, temp_sequence), axis=0)

            temp_sequence = np.concatenate((temp_sequence, np.reshape(label, (1, 1))), axis=0)
            with open(f"./{attack_path}/{seed}/attack_outputs/traj_based_buffers/test_{args.out_traj_size}.npy", 'ab') \
                    as f:
                np.save(f, temp_sequence)
            final_prediction_test_dataset.append(temp_sequence)
            temp_sequence = []
    print(f"Done creating test pairs!")

    return final_prediction_test_dataset


def create_sets(seeds, attack_training_size, timesteps, trajectory_length, num_predictions, dimension):
    path = "tmp_plks/"
    if not os.path.exists(path):
        os.mkdir(path)

    train_size = math.floor(attack_training_size / (10 / 8))
    eval_size = math.floor(attack_training_size / (10 / 2))
    num_traj_per_model = int(timesteps / trajectory_length)
    total_pairs_needed = attack_training_size + num_predictions
    data_length = 2 * (trajectory_length * dimension)

    data_train = np.empty([0, data_length])
    data_eval = np.empty([0, data_length])
    labels_train = np.empty(train_size, dtype=int)
    labels_eval = np.empty(eval_size, dtype=int)
    data_test = np.empty([0, data_length])
    labels_test = []
    #train and test pair inedcies
    train_pairs, test_pairs = generate_pairs(total_pairs_needed, len(seeds) * num_traj_per_model, num_predictions,
                                             attack_training_size)

    d_test = str(uuid.uuid4())
    d_t = str(uuid.uuid4())
    l_t = str(uuid.uuid4())
    d_e = str(uuid.uuid4())
    l_e = str(uuid.uuid4())

    # save test pairs
    indx = 0
    for x_i, y_i in test_pairs:
        x_model, y_model, same_set, index_x, index_y = get_models(x_i, y_i, num_traj_per_model)
        seed_x = seeds[x_model]
        seed_y = seeds[y_model]
        data_test = np.insert(data_test, indx,
                              np.concatenate((get_trajectory(seed_x, index_x, trajectory_length),
                                              get_trajectory_test(seed_y, index_y, trajectory_length))), axis=0)

        if same_set:
            labels_test.append(1)
        else:
            labels_test.append(0)

        indx += 1

    np.save(path + d_test + '.npy', data_test)

    del test_pairs, data_test
    gc.collect()
    print("saved test pairs")

    # save train pairs
    indx = 0
    for x_i, y_i in train_pairs:
        if indx < train_size:
            x_model, y_model, same_set, index_x, index_y = get_models(x_i, y_i, num_traj_per_model)
            seed_x = seeds[x_model - 1]
            seed_y = seeds[y_model - 1]
            data_train = np.insert(data_train, indx, np.concatenate(
                (get_trajectory(seed_x, index_x, trajectory_length),
                 get_trajectory_test(seed_y, index_y, trajectory_length))),
                                   axis=0)

            if same_set:
                labels_train.put(indx, 1)
            else:
                labels_train.put(indx, 0)

        else:
            break

        indx += 1

    np.save(path + d_t + '.npy', data_train)
    np.save(path + l_t + '.npy', labels_train)

    del data_train, labels_train
    gc.collect()
    print("saved train pairs")

    # save eval pairs
    insrt = 0
    while indx < attack_training_size:
        x_i, y_i = train_pairs[indx]
        x_model, y_model, same_set, index_x, index_y = get_models(x_i, y_i, num_traj_per_model)
        seed_x = seeds[x_model - 1]
        seed_y = seeds[y_model - 1]
        data_eval = np.insert(data_eval, insrt, np.concatenate(
            (get_trajectory(seed_x, index_x, trajectory_length),
             get_trajectory_test(seed_y, index_y, trajectory_length))),
                              axis=0)

        if same_set:
            labels_eval.put(insrt, 1)
        else:
            labels_eval.put(insrt, 0)

        indx += 1
        insrt += 1

    np.save(path + d_e + '.npy', data_eval)
    np.save(path + l_e + '.npy', labels_eval)

    del data_eval, labels_eval
    gc.collect()
    print("saved eval pairs")

    return d_t, l_t, d_e, l_e, d_test, labels_test


def logger(baseline, false_negatives_bl, false_positives_bl, rmse, accuracy, false_negatives, false_positives):
    print("Baseline Accuracy: ", baseline)
    print("Precision BL: ", false_negatives_bl)
    print("Recall BL: ", false_positives_bl)
    print("Attack Classifier Accuracy: ", accuracy)
    print("Precision: ", false_negatives)
    print("Recall: ", false_positives)
    print("Root MSE: ", rmse)
    print("****************************")


def rsme(errors):
    return np.sqrt(gmean(np.square(errors)))


def calc_errors(classifier_predictions, labels_test, threshold, num_predictions):
    errors = []
    for i in range(num_predictions):
        e_i = (labels_test[i] - classifier_predictions[i]) / (labels_test[i] - threshold)
        errors.append(e_i)

    return errors


def baseline_accuracy(labels_test, num_predictions):
    num_correct = 0
    false_positives = 0
    false_negatives = 0
    for i in range(num_predictions):
        guess = randint(0, 1)

        # if theyre the same
        if guess == labels_test[i]:
            num_correct += 1
        # said out was acutually in
        elif guess == 0 and labels_test[i] == 1:
            false_negatives += 1
        elif guess == 1 and labels_test[i] == 0:
            false_positives += 1

    return output_prec_recall((num_correct / num_predictions), (false_negatives / num_predictions),
                              (false_positives / num_predictions))


def accuracy_report(classifier_predictions, labels_test, threshold, num_predictions):
    num_correct = 0
    false_positives = 0
    false_negatives = 0
    for i in range(num_predictions):
        if classifier_predictions[i] >= threshold and labels_test[i] is 1:
            num_correct += 1
        elif classifier_predictions[i] < threshold and labels_test[i] is 0:
            num_correct += 1

        # false negative (classifier is saying out but labels say in)
        elif classifier_predictions[i] < threshold and labels_test[i] is 1:
            false_negatives += 1

        # false positive (classifier is saying in but labels say out)
        elif classifier_predictions[i] >= threshold and labels_test[i] is 0:
            false_positives += 1

    return output_prec_recall((num_correct / num_predictions), (false_negatives / num_predictions),
                              (false_positives / num_predictions))


def output_prec_recall(acc, fn, fp):
    tp = int(500 * acc)
    fn = int(500 * fn)
    fp = int(500 * fp)

    prec = tp / (tp + fp)
    recall = tp / (tp + fn)

    return acc, round(prec, 3), round(recall, 3)


def generate_metrics(classifier_predictions, labels_test, threshold, num_predictions):
    accuracy, false_negatives, false_positives = accuracy_report(classifier_predictions, labels_test, threshold,
                                                                 num_predictions)
    baseline, false_negatives_bl, false_positives_bl = baseline_accuracy(labels_test, num_predictions)
    RMSE_e_i = rsme(calc_errors(classifier_predictions, labels_test, threshold, num_predictions))

    logger(baseline, false_negatives_bl, false_positives_bl, RMSE_e_i, accuracy, false_negatives, false_positives)
    return baseline, false_negatives_bl, false_positives_bl, RMSE_e_i, accuracy, false_negatives, false_positives


def train_classifier(xgb_train, xgb_eval):
    num_round = 150
    param = {'eta': '0.2',
             'n_estimators': '5000',
             'max_depth': 20,
             'objective': 'reg:logistic',
             'eval_metric': ['logloss', 'error', 'rmse']}

    watch_list = [(xgb_eval, 'eval'), (xgb_train, 'train')]
    evals_result = {}
    print("training classifier")
    return xgb.train(param, xgb_train, num_round, watch_list, evals_result=evals_result)


def train_attack_model_v2(environment, threshold, trajectory_length, seeds, attack_model_size, test_size, timesteps,
                          dimension):
    path = "tmp_plks/"
    d_t, l_t, d_e, l_e, d_test, labels_test = create_sets(seeds, attack_model_size, timesteps, trajectory_length,
                                                          test_size, dimension)

    # xgb_t, xgb_e, xgb_test = create_xgb_train_test_eval(d_t, l_t, d_e, l_e, d_test)
    attack_classifier = train_classifier(xgb.DMatrix(np.load(path + d_t + '.npy'), label=np.load(path + l_t + '.npy')),
                                         xgb.DMatrix(np.load(path + d_e + '.npy'), label=np.load(path + l_e + '.npy')))

    print("training finished --> generating predictions")
    xgb_testing = xgb.DMatrix(np.load(path + d_test + '.npy'))
    classifier_predictions = attack_classifier.predict(xgb_testing)

    cleanup([d_t, l_t, d_e, l_e, d_test], ["1"])

    print_experiment(environment, len(seeds), threshold, trajectory_length,
                     attack_model_size)

    return generate_metrics(classifier_predictions, labels_test, threshold, test_size)


def train_attack_model_v3(attack_path, state_dim, action_dim, max_action, device, args):

    attack_train_positive_data, attack_eval_positive_data = create_train_pairs(attack_path, state_dim, action_dim,
                                                                               max_action, device, args, 1)
    attack_train_negative_data, attack_eval_negative_data = create_train_pairs(attack_path, state_dim, action_dim,
                                                                               max_action, device, args, 0)

    attack_train_data = np.concatenate([attack_train_positive_data, attack_train_negative_data])
    attack_eval_data = np.concatenate([attack_eval_positive_data, attack_eval_negative_data])

    print("preparing data for classifier training ...")

    classifier_train_data_x, classifier_train_data_y = [item[:-1] for item in attack_train_data], \
                                                       [np.asarray(item[-1]) for item in attack_train_data]
    classifier_eval_data_x, classifier_eval_data_y = [item[:-1] for item in attack_eval_data], \
                                                     [np.asarray(item[-1]) for item in attack_eval_data]
    #train_x = np.load(f"./{attack_path}/{0}/attack_outputs/traj_based_buffers/train_{args.out_traj_size}.npy")[1]
    train_y = DataFrame(classifier_train_data_y, columns=['trajectory_label'])

    #eval_x = np.load(f"./{attack_path}/{0}/attack_outputs/traj_based_buffers/eval_{args.out_traj_size}.npy")[1]
    eval_y = DataFrame(classifier_eval_data_y, columns=['trajectory_label'])

    classifier_train_data = xgb.DMatrix(pathlib.PurePath(f"./{attack_path}/{0}/attack_outputs/traj_based_buffers/eval_{args.out_traj_size}.npy"), train_y)
    classifier_eval_data = xgb.DMatrix(classifier_eval_data_x, eval_y)

    print("classifier training ...")
    attack_classifier = train_classifier(classifier_train_data, classifier_eval_data)
    print("training finished --> generating predictions")

    attack_test_positive_pairs = np.load(
        create_test_pairs(attack_path, state_dim, action_dim, max_action, device, args, 1))
    attack_test_negative_pairs = np.load(
        create_test_pairs(attack_path, state_dim, action_dim, max_action, device, args, 0))

    attack_test_pairs = np.concatenate([attack_test_positive_pairs, attack_test_negative_pairs])
    xgb_testing = xgb.DMatrix(attack_test_pairs)
    classifier_predictions = attack_classifier.predict(xgb_testing)

    print_experiment(args.env, args.seed, args.attack_threshold, None,
                     args.attack_training_size)
    #At the moment we only test the classifier against positive pairs
    return generate_metrics(classifier_predictions, [item[-1] for item in attack_test_pairs], args.attack_threshold,
                            args.attack_training_size)
