import os
import numpy as np
import pandas as pd
from random import sample
from utils.helpers import print_experiment, format_trajectory
from itertools import product
from workers.attack_classifier import train_attack_model_v2


def run_experiment_v2(environment, seeds, threshold, trajectory_length, attack_training_size, num_predictions,
                      timesteps,
                      dimension, num_models):
    seeds = sample(seeds, num_models)
    print("seeds chosen: ", seeds)

    save_models(seeds, environment, trajectory_length)
    baseline, false_negatives_b1, false_positives_bl, rmse, accuracy, false_negatives, false_positives = train_attack_model_v2(
        environment, threshold,
        trajectory_length, seeds,
        attack_training_size,
        num_predictions,
        timesteps, dimension)
    return baseline, false_negatives_b1, false_positives_bl, rmse, accuracy, false_negatives, false_positives


def run_experiments_v2(env, seeds, thresholds, trajectory_lengths, attack_sizes, num_predictions,
                       timesteps,
                       dimension, number_shadow_models):
    experiment = [trajectory_lengths, attack_sizes, number_shadow_models, thresholds]
    product_res = product(*experiment)
    results = []
    for (trajectory_length, attack_size, num_models, threshold) in product_res:
        baseline, false_negatives_b1, false_positives_bl, rmse, accuracy, false_negatives, false_positives = run_experiment_v2(
            env, seeds, threshold,
            trajectory_length,
            attack_size, num_predictions,
            timesteps, dimension,
            num_models)

        results.append(
            [timesteps, env, trajectory_length, attack_size, num_models, threshold, baseline, false_negatives_b1,
             false_positives_bl, rmse, accuracy,
             false_negatives, false_positives])

        logger_inplace(timesteps, env, trajectory_length, attack_size, num_models, threshold, baseline,
                       false_negatives_b1, false_positives_bl, rmse, accuracy,
                       false_negatives, false_positives)

    logger_overwrite(np.asarray(results), env, timesteps)


def logger_inplace(timesteps, env, trajectory_length, attack_size, num_models, threshold, baseline, false_negatives_b1,
                   false_positives_bl, rmse, accuracy,
                   false_negatives, false_positives):
    # log to file just in-case
    if not os.path.exists('output/results'):
        os.mkdir('output/results')

    with open('./output/results/' + env + '_' + str(timesteps), "a") as results:
        print((timesteps, env, trajectory_length, attack_size, num_models, threshold, baseline, false_negatives_b1,
               false_positives_bl, rmse, accuracy,
               false_negatives, false_positives),
              file=results)


def logger_overwrite(np_results, environment, timesteps):
    # sort and log when finished full experiment suite
    sorted_results = np_results[np_results[:, 8].argsort()[::-1]]
    with open('./output/results/' + environment + '_' + str(timesteps), "w") as results:
        print(
            "trajectory_length, env, attack_mdl_size, number_of_models, threshold, baseline_accuracy, baseline precision, baseline recall, rmse, attack_accuracy, attack precision, attack recall",
            file=results)
        for (timesteps, env, trajectory_length, attack_size, num_models, threshold, baseline, false_negatives_b1,
             false_positives_bl, rmse,
             accuracy, false_negatives, false_positives) in sorted_results:
            print((timesteps, env, trajectory_length, attack_size, num_models, threshold, baseline, false_negatives_b1,
                   false_positives_bl, rmse, accuracy,
                   false_negatives, false_positives),
                  file=results)

    return sorted_results


def save_models(seeds, environment, trajectory_length):
    path = 'tmp/'
    extension = '.npy'

    if not os.path.exists(path):
        os.mkdir(path)

    for seed in seeds:
        np.save(path + str(seed) + str(trajectory_length) + '.npy',
                format_trajectory(trajectory_length,
                                  np.load('output/' + environment + '/seed_' + str(seed) + '/maxEpLen_'
                                          + str(trajectory_length) + '/trajectories' + extension,
                                          allow_pickle=True)))

    for seed in seeds:
        np.save(path + str(seed) + str(trajectory_length) + '_test' + '.npy',
                format_trajectory(trajectory_length, np.load(
                    'output/' + environment + '/seed_' + str(seed) + '/maxEpLen_' + str(trajectory_length)
                    + '/trajectories_test' + extension,
                    allow_pickle=True)))
