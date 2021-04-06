import argparse
import os
import datetime
import time
import logging

# import yaml

import BCQ
import BCQutils
import DDPG
from workers import attack, experiment
from utils.configs import *
from utils.helpers import str2bool
logger = logging.getLogger(__name__)

import gym
import numpy as np
import torch
from utils.configs import CORRELATED, DECORRELATED, SEMI_CORRELATED, CORRELATION_MAP

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--attack_final_results', default=os.path.expanduser('~') + '/attack_output',
                        help='output path for files produced by the attack agent')
    parser.add_argument("--env", help="the environment you are in", default="Hopper-v3")  # OpenAI gym environment name
    parser.add_argument("--seed", nargs=4, type=int)                          # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument('--num_models', default=1, help="number of shadow models", type=int)
    # parser.add_argument("--shadow_seeds", nargs='+', type=int, help="At least two integers")
    # parser.add_argument("--target_seeds", nargs=2, type=int)  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--buffer_name", default="Robust")          # Prepends name to filename

    parser.add_argument("--eval_freq", default=5e3, type=float)     # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=1e6,
                        type=int)   # Max time steps to run environment or train for (this defines buffer size)
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
    parser.add_argument("--attack_thresholds", nargs='+', type=float)  # Threshold for attack training
    parser.add_argument("--attack_sizes", nargs='*', type=int)  # Attack training size
    parser.add_argument('--out_traj_size', default=10, type=int)  # This is used to bound the number of test trajecotries
    parser.add_argument('--in_traj_size', default=10, type=int)  # This is used to bound the number of train trajectories
    parser.add_argument('--ratio_size_prediction', default=0.25, type=float, help="determines the ratio of out- and "
                                                                                  "in-traj-sizes for making prediction "
                                                                                  "pairs")

    parser.add_argument('--just_one', default='no', choices=["yes", "no"], help="just run one experiment", type=str)
    parser.add_argument('--all', default='no', choices=["yes", 'no'], help="run all tests")
    parser.add_argument('--fix_num_models', default='no')
    parser.add_argument('--attack_model_size', default=1000, type=int,
                        help="size of the training set for the attack model")
    parser.add_argument('--run_multiple', default='no', help="choose a variable attribute with all others fixed")
    # parser.add_argument('--model', default='sac', help="model used to train the shadow_models")
    parser.add_argument('--trajectory_length', nargs='*', default=1000, type=int)  #Must be equal to the max_ep_length in trainer.py
    parser.add_argument('--max_traj_len', default=1000, type=int)
    parser.add_argument('--correlation', default='c', choices=["c", 'd', 's'], help="Activate semi/de/correlated mode.")
    parser.add_argument('--max_depth', default=20, type=int, help="xgboost maximum depth of the decision tree.")
    parser.add_argument('--xgb_n_rounds', default=150, type=int, help="xgboost number of decision trees")
    parser.add_argument('--bcq_max_timesteps', default=1000, type=int)

    args = parser.parse_args()

    # reading the parameter from a yaml file instead of the command line arguments!
    # This is to automate the entire process!
    # run_conf = None
    # if not os.path.exists("config.yaml"):
    #     raise FileNotFoundError("Config Yaml file not found!")
    # with open("config.yaml", 'r') as yaml_f:
    #     try:
    #         run_conf = yaml.safe_load(yaml_f)
    #     except yaml.YAMLError as exc:
    #         print(exc)
    #         raise

    print("---------------------------------------")
    print(f"Setting: Training Attack, Env: {args.env}, Seed: {args.seed}, Max Trajectory Length: {args.max_traj_len}")

    attack_path = f"{args.env}/{args.max_timesteps}"

    # *********************************** Logging Config ********************************************
    file_path_results = args.attack_final_results + f"/{args.env}/MaxT_{args.max_timesteps}/" \
                                                    f"bcqMaxT_{args.bcq_max_timesteps}/MaxTraj_{args.max_traj_len}/" \
                                                    f"{CORRELATION_MAP.get(args.correlation)}"
    if not os.path.exists(file_path_results):
        os.makedirs(file_path_results)
    logging.basicConfig(level=logging.DEBUG, filename=file_path_results + "/" +
                                                      str(datetime.datetime.now()).replace(" ", "_") + "_log.txt")
    logging.getLogger().addHandler(logging.StreamHandler())

    header = "===================== Experiment configuration ========================"
    logger.info(header)
    args_keys = list(vars(args).keys())
    args_keys.sort()
    max_k = len(max(args_keys, key=lambda x: len(x)))
    for k in args_keys:
        s = k + '.' * (max_k - len(k)) + ': %s' % repr(getattr(args, k))
        logger.info(s + ' ' * max((len(header) - len(s), 0)))
    logger.info("=" * len(header))

    env = gym.make(args.env)
    # Note the used of single seed here.
    # Though the usage of a single seed does not affect state_dim, action_dim, max_action.
    # TODO: would this usage model affect other parts of the code?
    env.seed(args.seed[0])
    # Bounding the maximum allowed trajectory length in the environment.
    # This is modified here for consistency with runner_v2. It seems that it is not affecting attack_trainer!
    env._max_episode_steps = args.max_traj_len
    torch.manual_seed(args.seed[0])
    np.random.seed(args.seed[0])

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    # Remove as it is not used in this step
    max_action = float(env.action_space.high[0])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    experiment.run_experiments_v2(attack_path, file_path_results, state_dim, action_dim, device, args)


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
