import os
import numpy as np
import pandas as pd
from random import sample
from utils.helpers import print_experiment, format_trajectory
from itertools import product
from workers.attack import train_attack_model_v3


def run_experiment_v2(environment, seeds, threshold, attack_training_size, num_predictions,
                      dimension, num_models, model, timesteps, max_ep_length):
    seeds = sample(seeds, num_models)
    print("seeds chosen: ", seeds)

    save_models(seeds, environment, model, timesteps, max_ep_length)
    baseline, false_negatives_b1, false_positives_bl, rmse, accuracy, false_negatives, false_positives = train_attack_model_v2(
        environment, threshold,
        max_ep_length, seeds,
        attack_training_size,
        num_predictions,
        timesteps, dimension)
    return baseline, false_negatives_b1, false_positives_bl, rmse, accuracy, false_negatives, false_positives


def run_experiments_v2(attack_path, state_dim, action_dim, device, args):
    experiment = [args.attack_sizes, args.attack_thresholds]
    product_res = product(*experiment)
    results = []
    for (attack_size, attack_threshold) in product_res:
        accuracy_bl, precision_bl, recall_bl, rmse, accuracy, precision, recall = \
            train_attack_model_v3(attack_path, state_dim, action_dim, device, args)

        results.append(
            [args.max_timesteps, args.env, attack_size, attack_threshold, accuracy_bl, precision_bl,
             recall_bl, rmse, accuracy,
             precision, recall])

        logger_inplace(args.max_timesteps, args.env, attack_size, attack_threshold, accuracy_bl, precision_bl,
             recall_bl, rmse, accuracy,
             precision, recall)

    logger_overwrite(np.asarray(results), args.env, args.max_timesteps)


def logger_inplace(timesteps, env, attack_size, threshold, baseline, false_negatives_b1,
                   false_positives_bl, rmse, accuracy,
                   false_negatives, false_positives):
    # log to file just in-case
    if not os.path.exists('output/results'):
        os.makedirs('output/results')

    with open('./output/results/' + env + '_' + str(timesteps), "a") as results:
        print((timesteps, env, attack_size, threshold, baseline, false_negatives_b1,
               false_positives_bl, rmse, accuracy,
               false_negatives, false_positives),
              file=results)


def logger_overwrite(np_results, environment, timesteps):
    # sort and log when finished full experiment suite
    sorted_results = np_results[np_results[:, 8].argsort()[::-1]]
    with open('./output/results/' + environment + '_' + str(timesteps), "w") as results:
        print(
            "timesteps, env, attack_threshold, baseline_accuracy, baseline precision, baseline recall, rmse, attack_accuracy, attack precision, attack recall",
            file=results)
        for (timesteps, env, attack_size, threshold, baseline, false_negatives_b1,
             false_positives_bl, rmse,
             accuracy, false_negatives, false_positives) in sorted_results:
            print((timesteps, env, attack_size, threshold, baseline, false_negatives_b1,
                   false_positives_bl, rmse, accuracy,
                   false_negatives, false_positives),
                  file=results)

    return sorted_results


def save_models(seeds, environment, model, timesteps, max_ep_length):
    path = 'tmp/'
    extension = '.npy'

    if not os.path.exists(path):
        os.mkdir(path)

    for seed in seeds:
        np.save(path + environment + '_' + model + '_seed' + str(seed) + '_maxEpLen' + str(max_ep_length) + '_timeSteps' + str(timesteps) + '.npy',
                format_trajectory(max_ep_length,
                                  np.load('output/' + environment + '/' + model + '/TimeSteps_' + str(timesteps)
                                          + '/seed_' + str(seed) + '/maxEpLen_' + str(max_ep_length) + '/trajectories' + extension,
                                          allow_pickle=True)))

    for seed in seeds:
        np.save(path + environment + '_' + model + '_seed' + str(seed) + '_maxEpLen' + str(max_ep_length) + '_timeSteps' + str(timesteps) + '_test' + '.npy',
                format_trajectory(max_ep_length, np.load(
                    'output/' + environment + '/' + model + '/TimeSteps_' + str(timesteps) + '/seed_' + str(seed)
                    + '/maxEpLen_' + str(max_ep_length) + '/trajectories_test' + extension,
                    allow_pickle=True)))
