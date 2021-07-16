import argparse
import os
import datetime
import time
import logging
import gym
import numpy as np
import torch
from utils.configs import CORRELATED, DECORRELATED, SEMI_CORRELATED, CORRELATION_MAP

# import yaml

import BCQ
import BCQutils
import DDPG
from workers import attack, experiment
from utils.configs import *
from utils.helpers import str2bool
logger = logging.getLogger(__name__)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--attack_final_results', default=os.path.expanduser('~') + '/attack_output',
                        help='output path for files produced by the attack agent')
    parser.add_argument("--env", help="the environment you are in", default="Hopper-v3")  # OpenAI gym environment name
    # parser.add_argument("--seed", nargs=4, type=int)                          # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument('--num_models', default=1, help="number of shadow models", type=int)
    parser.add_argument("--shadow_seeds", nargs='+', type=int, help="two integers (for num_models 1 and 2). "
                                                                    "for num_models > 2: "
                                                                    "the number of shadow_seeds = num_models") # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--target_seeds", nargs=2, type=int)
    parser.add_argument("--env_seeds", nargs='+', type=int, help="Number of inputs = num_models + 1")
    parser.add_argument("--buffer_name", default="Robust")          # Prepends name to filename

    parser.add_argument("--eval_freq", default=5e3, type=float)     # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=1e6,
                        type=int)   # Max time steps to run environment or train for (this defines buffer size)
    parser.add_argument("--generatebuffer_max_timesteps", default=1e6, type=int)
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
    parser.add_argument("--create_pairs", action="store_true")  # If true, creates positive and negative pairs
    parser.add_argument("--train_policy", action="store_true")  # If true, train policy (BCQ)
    parser.add_argument("--generate_buffer", action="store_true")  # If true, generate buffer
    parser.add_argument("--attack_thresholds", nargs='+', type=float)  # Threshold for attack training
    parser.add_argument("--attack_size", default=1000, type=int)  # Attack prediction size
    parser.add_argument("--train_size", default=20000, type=int)  # Attack train size for label 1 or 0
    parser.add_argument('--out_traj_size', default=10, type=int) # This is used to bound the number of test trajectories
    parser.add_argument('--in_traj_size', default=10, type=int) # This is used to bound the number of train trajectories
    parser.add_argument('--ratio_size_prediction', default=0.25, type=float, help="determines the ratio of out- and "
                                                                                  "in-traj-sizes for making prediction "
                                                                                  "pairs")
    parser.add_argument("--truncate_traj", action="store_true")  # If true, truncates the trajectory instead of padding
    parser.add_argument('--padding_size', default=25,
                        type=int)  # This is used if --truncate_traj is True

    parser.add_argument('--just_one', default='no', choices=["yes", "no"], help="just run one experiment", type=str)
    parser.add_argument('--all', default='no', choices=["yes", 'no'], help="run all tests")
    parser.add_argument('--fix_num_models', default='no')
    parser.add_argument('--attack_model_size', default=1000, type=int,
                        help="size of the training set for the attack model")
    parser.add_argument('--run_multiple', default='no', help="choose a variable attribute with all others fixed")
    # parser.add_argument('--model', default='sac', help="model used to train the shadow_models")
    parser.add_argument('--trajectory_length', nargs='*', default=1000, type=int)  # Must be equal to the
    # max_ep_length in trainer.py
    parser.add_argument('--max_traj_len', default=1000, type=int)
    parser.add_argument('--correlation', default='c', choices=["c", 'd', 's'], help="Activate semi/de/correlated mode.")
    parser.add_argument('--bcq_max_timesteps', default=1000, type=int)
    parser.add_argument('--pairing_mode', default='horizontal', choices=['horizontal', 'vertical'],
                        help='the action sequences are paired either horizontally or vertically')

    # xgboost initial parameter values or fixed parameter values in the case which we do not want to tune the parameters
    parser.add_argument('--early_stopping_rounds', default=10, type=int, help="xgboost early stopping rounds")
    parser.add_argument('--max_depth', default=2, type=int, help="xgboost maximum depth of the decision tree.")
    parser.add_argument('--xg_eta', default=0.1, type=float, help="xgboost learning rate")
    parser.add_argument('--xgb_n_rounds', default=1000, type=int, help="xgboost maximum n_estimators")
    parser.add_argument('--min_child_weight', default=1, type=int, help="xgboost min_child_weight")
    parser.add_argument('--gamma', default=0, type=float, help="xgboost gamma")
    parser.add_argument('--subsample', default=0.8, type=float, help="xgboost subsample")
    parser.add_argument('--colsample_bytree', default=0.8, type=float, help="xgboost colsample_bytree")
    parser.add_argument('--reg_alpha', default=0, type=float, help="xgboost reg_alpha")

    # xgboost hyper parameter tuning (Leave any parameter that doesn't need to be tuned empty)
    parser.add_argument("--cv_tune_xgb", action="store_true")  # If true, it tunes the xgb hyper parameters

    parser.add_argument('--max_depth_vector', nargs='+', type=int, help="Typically between 3 and 10, but could be "
                                                                        "higher.  e.g.: 2 4 6 8 10")
    parser.add_argument('--min_child_weight_vector', nargs='+', type=int, help='Default is 1, and can change between'
                                                                               ' 0 and infinity. e.g.: 1 3 5')

    parser.add_argument('--gamma_vector', nargs='+', type=float, help="e.g.: 0 0.1 0.2 0.3 0.4 0.5")

    parser.add_argument('--subsample_vector', nargs='+', type=float, help="Typically between 0.5 and 1."
                                                                          "e.g.: 0.6 0.7 0.75 0.8 0.9")

    parser.add_argument('--colsample_bytree_vector', nargs='+', type=float, help="Typically between 0.5 and 1."
                                                                          "e.g.: 0.6 0.7 0.75 0.8 0.9")

    parser.add_argument("--reg_alpha_vector", nargs='+', type=float, help="e.g: 1e-5 1e-3 0.005 1e-2 0.1 1 10 100")

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
    print(f"Setting: Training Attack, Env: {args.env}, Shadow Seeds: {args.shadow_seeds}, "
          f"Target Seeds: {args.target_seeds}Max Trajectory Length: {args.max_traj_len}")

    attack_path = os.path.expanduser('~') + f"/projects/rrg-dprecup/samin/learning_output/{args.env}/{args.max_timesteps}/" \
                                            f"{args.generatebuffer_max_timesteps}"

    # *********************************** Logging Config ********************************************
    file_path_results = args.attack_final_results + f"/{args.env}/MaxT_{args.max_timesteps}/" \
                                                    f"genT_{args.generatebuffer_max_timesteps}/" \
                                                    f"bcqMaxT_{args.bcq_max_timesteps}/MaxTraj_{args.max_traj_len}/" \
                                                    f"{CORRELATION_MAP.get(args.correlation)}"

    if not os.path.exists(file_path_results):
        os.makedirs(file_path_results)

    pair_path_results = file_path_results + f"/pairs/{args.pairing_mode}/train_NumModel_{args.num_models}_" \
                                            f"ShSeed_{args.shadow_seeds}_TaSeed_{args.target_seeds}_" \
                                            f"EnvSeed_{args.env_seeds}"

    if not os.path.exists(pair_path_results):
        os.makedirs(pair_path_results)

    if args.create_pairs:
        logging.basicConfig(level=logging.DEBUG, filename=file_path_results + "/" +
                                                          str(datetime.datetime.now()).replace(" ", "_") + "_log.txt")
    else:
        logging.basicConfig(level=logging.DEBUG, filename=pair_path_results + "/" +
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
    env.seed(args.shadow_seeds[0])
    # Bounding the maximum allowed trajectory length in the environment.
    # This is modified here for consistency with runner_v2. It seems that it is not affecting attack_trainer!
    env._max_episode_steps = args.max_traj_len
    torch.manual_seed(args.shadow_seeds[0])
    np.random.seed(args.shadow_seeds[0])

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    # Remove as it is not used in this step
    max_action = float(env.action_space.high[0])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.create_pairs:
        experiment.run_experiments_v2(attack_path, file_path_results, pair_path_results, state_dim, action_dim, device, args)
    else:
        experiment.run_classifier(attack_path, file_path_results, pair_path_results, state_dim, action_dim, device, args)