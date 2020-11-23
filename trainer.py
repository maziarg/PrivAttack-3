import os
import torch
import gym
import argparse
import numpy as np
from sac.sac import sac
from ddpg.ddpg import ddpg


def output_model(model, environment, seed, timesteps, max_ep_length):
    path = './output/' + environment + '/' + model + '/TimeSteps_' + str(timesteps) + '/seed_' + str(seed) + '/maxEpLen_' + str(max_ep_length) + '/trajectories'
    epoch_length = 2000
    epochs = int(timesteps / epoch_length)
    env_fn = lambda: gym.make(environment)
    logger_kwargs = dict(output_dir='output/' + environment + '/' + model + '/TimeSteps_' + str(timesteps) + '/seed_' + str(seed) + '/maxEpLen_' + str(max_ep_length),
                         exp_name = environment + '_shadow_' + str(seed))
    if model == 'sac':
        sac(trajectory_output_path=path, env_fn=env_fn, logger_kwargs=logger_kwargs, seed=seed, epochs=epochs,
            steps_per_epoch=epoch_length, max_ep_len = max_ep_length)

    elif model == 'ddpg':
        ddpg(trajectory_output_path=path, env_fn=env_fn, logger_kwargs=logger_kwargs, seed=seed, epochs=epochs,
             steps_per_epoch=epoch_length, max_ep_len = max_ep_length)

    else:
        print('could not find model')
        exit(-1)


def generate_test_pkl(environment, model, seed, timesteps, max_ep_length):
    path = 'output/' + environment + '/' + args.m + '/TimeSteps_' + str(timesteps) + '/seed_' + str(seed) + '/maxEpLen_' + str(max_ep_length)
    env = gym.make(environment)
    env.seed(seed)
    obs, reward, d = env.reset(), 0, False
    trajectories = []
    append = trajectories.append
    for i in range(0, timesteps):
        action = model.act(torch.as_tensor(obs, dtype=torch.float32))
        append((obs, action, reward, d))
        obs2, reward, d, _ = env.step(action)
        obs = obs2

        if d:  # reset env if done
            obs = env.reset()
    np.save(path + '/trajectories_test.npy', np.asarray(trajectories))


def train_shadow_model(model, environment, seed, timesteps, max_ep_length):
    if not os.path.exists('output'):
        os.mkdir('output')
    output_model(model, environment, seed, timesteps, max_ep_length)
    trained_model = torch.load('output/' + environment + '/' + model + '/TimeSteps_' + str(timesteps) + '/seed_' + str(seed) + '/maxEpLen_' + str(max_ep_length) + '/pyt_save/model.pt')
    generate_test_pkl(environment, trained_model, seed, timesteps, max_ep_length)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', help="the environment you are in", default="HalfCheetah-v2")
    parser.add_argument('-m', help="the DRL model you wish to use", default="sac")
    parser.add_argument('--timesteps', type=int)
    parser.add_argument('--seeds', nargs='+')
    parser.add_argument('--max_ep_length', default = 1000)
    args = parser.parse_args()

    for seed in args.seeds:
        train_shadow_model(args.m, args.e, int(seed), args.timesteps, args.max_ep_length)
