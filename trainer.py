import os
import torch
import gym
import argparse
import numpy as np
from sac.sac import sac
from ddpg.ddpg import ddpg


def output_model(model, environment, seed, timesteps):
    path = './output/' + environment + '/' + str(seed) + '/trajectories'
    epoch_length = 2000
    epochs = int(timesteps / epoch_length)
    env_fn = lambda: gym.make(environment)
    logger_kwargs = dict(output_dir='output/' + environment + '/' + str(seed),
                         exp_name=environment + '_shadow_' + str(seed))
    if model == 'sac':
        sac(trajectory_output_path=path, env_fn=env_fn, logger_kwargs=logger_kwargs, seed=seed, epochs=epochs,
            steps_per_epoch=epoch_length)

    elif model == 'ddpg':
        ddpg(trajectory_output_path=path, env_fn=env_fn, logger_kwargs=logger_kwargs, seed=seed, epochs=epochs,
             steps_per_epoch=epoch_length)

    else:
        print('could not find model')
        exit(-1)


def generate_test_pkl(environment, model, seed, timesteps):
    path = 'output/' + environment + '/' + str(seed)
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


def train_shadow_model(model, environment, seed, timesteps):
    if not os.path.exists('output'):
        os.mkdir('output')
    output_model(model, environment, seed, timesteps)
    trained_model = torch.load('output/' + environment + '/' + str(seed) + '/pyt_save/model.pt')
    generate_test_pkl(environment, trained_model, seed, timesteps)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', help="the environment you are in", default="HalfCheetah-v2")
    parser.add_argument('-m', help="the DRL model you wish to use", default="sac")
    parser.add_argument('--timesteps', type=int)
    parser.add_argument('--seeds', nargs='+')
    args = parser.parse_args()

    for seed in args.seeds:
        train_shadow_model(args.m, args.e, int(seed), args.timesteps)
