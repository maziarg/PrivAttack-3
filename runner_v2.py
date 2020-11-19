import argparse
from utils.configs import *

from workers.experiment import run_experiment_v2, run_experiments_v2

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', help="the environment you are in", default="HalfCheetah-v2")
    parser.add_argument('--timesteps', type=int)
    parser.add_argument('--just_one', default='no', choices=["yes", "no"], help="just run one experiment", type=str)
    parser.add_argument('--all', default='no', choices=["yes", 'no'], help="run all tests")
    parser.add_argument('--fix_num_models', default='no')
    #parser.add_argument('--threshold', default=0.5, type=float)
    parser.add_argument('--threshold_arr', nargs= '*', type=float)
    parser.add_argument('--num_models', default=3, help="number of models to use", type=int)
    #parser.add_argument('--traj_length', default=1000, help="length of trajectory", type=int)
    parser.add_argument('--attack_model_size', default=1000, type=int,
                        help="size of the training set for the attack model")
    parser.add_argument('--run_multiple', default='no', help="choose a variable attribute with all others fixed")
    parser.add_argument('--seeds', nargs='*', default='none', type=int)
    parser.add_argument('--model', default='sac', help="model used to train the shadow_models")
    parser.add_argument('--trajectory_length' , nargs= '*', type = int) #Must be equal to the max_ep_length in trainer.py

    args = parser.parse_args()
    timestep_seeds = None
    dimension = 0
    print(args)

    if args.e == "HalfCheetah-v2":
        dimension = 24
    if args.e == "Humanoid-v2":
        dimension = 394
    if args.e == "Hopper-v2":
        dimension = 15

    if args.fix_num_models != 'no':
        num_models = int(args.fix_num_models)
        print("runnning with a fixed number of models")
        run_experiments_v2(args.e, args.seeds, args.threshold_arr, args.trajectory_length, attack_model_size, num_predictions=500,
                           timesteps=args.timesteps, dimension=dimension, number_shadow_models=[num_models])
    else:
        run_experiments_v2(args.e, args.seeds, args.threshold_arr, args.trajectory_length, attack_model_size, num_predictions=500,
                           timesteps=args.timesteps, dimension=dimension, number_shadow_models=num_shadow_models)
