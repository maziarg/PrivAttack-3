import argparse
from utils.configs import *

import argparse
import gym
import numpy as np
import os
import torch

import BCQ
import DDPG
import BCQutils

from workers import attack, experiment

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--env" , help="the environment you are in", default="Hopper-v3")           # OpenAI gym environment name
    parser.add_argument("--seed", nargs=4)                                              # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--buffer_name", default="Robust")                                          # Prepends name to filename

    parser.add_argument("--eval_freq", default=5e3, type=float)                                     # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=1e6,
                        type=int)                           # Max time steps to run environment or train for (this defines buffer size)
    parser.add_argument("--start_timesteps", default=25e3,
                        type=int) # Time steps initial random policy is used before training behavioral
    parser.add_argument("--rand_action_p", default=0.3,
                        type=float)  # Probability of selecting random action during batch generation
    parser.add_argument("--gaussian_std", default=0.3,
                        type=float)  # Std of Gaussian exploration noise (Set to 0.1 if DDPG trains poorly)
    parser.add_argument("--batch_size", default=100, type=int)  # Mini batch size for networks
    parser.add_argument("--discount", default=0.99)  # Discount factor
    parser.add_argument("--tau", default=0.005)  # Target network update rate
    parser.add_argument("--lmbda", default=0.75)  # Weighting for clipped double Q-learning in BCQ
    parser.add_argument("--phi", default=0.05)  # Max perturbation hyper-parameter for BCQ
    parser.add_argument("--train_behavioral", action="store_true")  # If true, train behavioral (DDPG)
    parser.add_argument("--train_policy", action="store_true")  # If true, train policy (BCQ)
    parser.add_argument("--generate_buffer", action="store_true")  # If true, generate buffer
    parser.add_argument("--attack_thresholds", nargs='*', type=float)  # Threshold for attack training
    parser.add_argument("--attack_sizes", nargs='*', type=float)  # Attack training size

    parser.add_argument('--just_one', default='no', choices=["yes", "no"], help="just run one experiment", type=str)
    parser.add_argument('--all', default='no', choices=["yes", 'no'], help="run all tests")
    parser.add_argument('--fix_num_models', default='no')
    parser.add_argument('--num_models', default=3, help="number of models to use", type=int)
    parser.add_argument('--attack_model_size', default=1000, type=int,
                        help="size of the training set for the attack model")
    parser.add_argument('--run_multiple', default='no', help="choose a variable attribute with all others fixed")
    parser.add_argument('--model', default='sac', help="model used to train the shadow_models")
    parser.add_argument('--trajectory_length' , nargs='*', default= 1000, type = int) #Must be equal to the max_
    # ep_length in trainer.py

    args = parser.parse_args()

    print("---------------------------------------")
    if args.train_behavioral:
        print(f"Setting: Training behavioral, Env: {args.env}, Seed: {args.seed}")
    elif args.generate_buffer:
        print(f"Setting: Generating buffer, Env: {args.env}, Seed: {args.seed}")
    else:
        print(f"Setting: Training BCQ, Env: {args.env}, Seed: {args.seed}")
    print("---------------------------------------")

    if args.train_behavioral and args.generate_buffer:
        print("Train_behavioral and generate_buffer cannot both be true.")
        exit()

    attack_path = f"{args.env}/{args.max_timesteps}/{args.seed}"

    env = gym.make(args.env)

    env.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    experiment.run_experiments_v2(attack_path, state_dim, action_dim, max_action, device, args)



    #training_iters = 0

    # timestep_seeds = None
    # dimension = 0
    # print(args)
    #
    # if args.env == "HalfCheetah-v2":
    #     dimension = 24
    # if args.env == "Humanoid-v2":
    #     dimension = 394
    # if args.env == "Hopper-v2":
    #     dimension = 15
    #
    # if args.fix_num_models != 'no':
    #     num_models = int(args.fix_num_models)
    #     print("runnning with a fixed number of models")
    #     run_experiments_v2(args.e, args.seed, args.threshold_arr, attack_model_size, num_predictions=50,
    #                        dimension=dimension, number_shadow_models=[num_models],
    #                        model = args.model, timesteps=args.timesteps, max_ep_length = args.trajectory_length)
    # else:
    #     run_experiments_v2(args.e, args.seed, args.threshold_arr, attack_model_size, num_predictions=500,
    #                        dimension=dimension, number_shadow_models=num_shadow_models,
    #                        model = args.model, timesteps=args.timesteps, max_ep_length = args.trajectory_length)
