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


# Handles interactions with the environment, i.e. train behavioral or generate buffer
def interact_with_environment(attack_path, env, state_dim, action_dim, max_action, device, args):
    # For saving files
    setting = f"{args.env}_{args.seed}"
    buffer_name = f"{args.buffer_name}_{setting}"

    # Initialize and load policy
    policy = DDPG.DDPG(state_dim, action_dim, max_action, device)  # , args.discount, args.tau)
    if args.generate_buffer: policy.load(f"{attack_path}/models/behavioral_{setting}")

    # Initialize buffer
    replay_buffer = BCQutils.ReplayBuffer(state_dim, action_dim, device, max_size=args.max_timesteps)

    evaluations = []

    state, done = env.reset(), False
    replay_buffer.initial_state.append(state)
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0

    # Interact with the environment for max_timesteps
    for t in range(int(args.max_timesteps)):

        episode_timesteps += 1

        # Select action with noise
        if (
                (args.generate_buffer and np.random.uniform(0, 1) < args.rand_action_p) or
                (args.train_behavioral and t < args.start_timesteps)
        ):
            action = env.action_space.sample()
        else:
            action = (
                    policy.select_action(np.array(state))
                    + np.random.normal(0, max_action * args.gaussian_std, size=action_dim)
            ).clip(-max_action, max_action)

        # Perform action
        next_state, reward, done, _ = env.step(action)
        # TODO: check if we need this line. This is because, we set max_episode step when we instantiate the env. Susan: I don't think we need it. I checked it and it works with different max_traj_length
        # Then, env should know it has reached the absorbing state and return done=True.
        # In fact the code in gym, seems to be doing that.
        # done_bool = float(done) if episode_timesteps < env._max_episode_steps else 0

        # Store data in replay buffer
        replay_buffer.add(state, action, next_state, reward, float(done))

        state = next_state
        episode_reward += reward

        # Train agent after collecting sufficient data
        if args.train_behavioral and t >= args.start_timesteps:
            policy.train(replay_buffer, args.batch_size)

        if done:
            # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
            print(
                f"Total T: {t + 1} Episode Num: {episode_num + 1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
            # Reset environment
            state, done = env.reset(), False
            replay_buffer.initial_state.append(state)
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1

        # Evaluate episode
        if args.train_behavioral and (t + 1) % args.eval_freq == 0:
            evaluations.append(eval_policy(policy, args.env, args.seed, max_episode_step=args.max_traj_len))
            np.save(f"{attack_path}/results/behavioral_{setting}", evaluations)
            policy.save(f"{attack_path}/models/behavioral_{setting}")

    # Save final policy
    if args.train_behavioral:
        policy.save(f"{attack_path}/models/behavioral_{setting}")

    # Save final buffer and performance
    else:
        evaluations.append(eval_policy(policy, args.env, args.seed, max_episode_step=args.max_traj_len))
        np.save(f"{attack_path}/results/buffer_performance_{setting}", evaluations)
        replay_buffer.save(f"{attack_path}/buffers/{buffer_name}")


# Trains BCQ offline
def train_BCQ(attack_path, state_dim, action_dim, max_action, device, args):
    buffer_name = f"{args.buffer_name}_{args.env}_{args.seed}"
    # For saving files
    setting = f"{args.env}_{args.seed}_{args.bcq_max_timesteps}"

    # Initialize policy
    policy = BCQ.BCQ(state_dim, action_dim, max_action, device, args.discount, args.tau, args.lmbda, args.phi)

    # Load buffer
    replay_buffer = BCQutils.ReplayBuffer(state_dim, action_dim, device, max_size=args.max_timesteps)
    replay_buffer.load(f"{attack_path}/buffers/{buffer_name}")

    evaluations = []
    episode_num = 0
    done = True
    training_iters = 0

    while training_iters < args.bcq_max_timesteps:
        policy.train(replay_buffer, iterations=int(args.eval_freq), batch_size=args.batch_size)

        evaluations.append(eval_policy(policy, args.env, args.seed, max_episode_step=args.max_traj_len))
        np.save(f"{attack_path}/results/BCQ_{setting}", evaluations)

        training_iters += args.eval_freq
        print(f"Training iterations: {training_iters}")
    policy.save(f"{attack_path}/models/target_{setting}")


# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(policy, env_name, seed, eval_episodes=10, max_episode_step=None):
    eval_env = gym.make(env_name)
    # eval_env.seed(seed + 100)
    eval_env.seed(seed)
    # Bounding the maximum allowed trajectory length in the environment
    if max_episode_step:
        eval_env._max_episode_steps = max_episode_step

    avg_reward = 0.
    for _ in range(eval_episodes):
        state, done = eval_env.reset(), False
        while not done:
            action = policy.select_action(np.array(state))
            state, reward, done, _ = eval_env.step(action)
            avg_reward += reward

    avg_reward /= eval_episodes

    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
    print("---------------------------------------")
    return avg_reward


# Handles policy interactions with the environment, i.e. generate test buffer
def policy_interact_with_environment(attack_path, env, state_dim, action_dim, max_action, device, args):
    # For saving files
    setting = f"{args.env}_{args.seed}_{args.bcq_max_timesteps}"
    buffer_name = f"target_{args.buffer_name}_{setting}"

    train_states = np.load(f"{attack_path}/buffers/{args.buffer_name}_{args.env}_{args.seed}_state.npy")
    traj_end_index = np.load(f"{attack_path}/buffers/{args.buffer_name}_{args.env}_{args.seed}_trajectory_end_index.npy")
    train_initial_states = [train_states[0, :] if i == -1 else train_states[traj_end_index[i]+1, :]
                      for i in range(-1, len(traj_end_index) - 1)]

    # Initialize and load policy
    policy = BCQ.BCQ(state_dim, action_dim, max_action, device, args.discount, args.tau, args.lmbda, args.phi)
    policy.load(f"{attack_path}/models/target_{setting}")

    # Initialize buffer
    replay_buffer = BCQutils.ReplayBuffer(state_dim, action_dim, device, max_size=len(train_initial_states))
    evaluations = []

    # Env initialization
    env = gym.make(args.env)

    env.seed(args.seed)
    # Bounding the maximum allowed trajectory length in the environment
    env._max_episode_steps = args.max_traj_len
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    # state = env.reset()

    episode_num = 0
    total_t = 0
    # Interact with the environment for max_timesteps
    for i in range(len(train_initial_states)):
        state, done = env.reset(), False
        if not np.array_equal(state, train_initial_states[i].ravel()):
            raise ValueError('The initial state is not the same as that in the training data')
        replay_buffer.initial_state.append(state)
        episode_reward = 0
        episode_timesteps = 0
        episode_num += 1
        # Select action using the target policy
        while not done:
            episode_timesteps += 1
            total_t += 1
            action = policy.select_action(np.array(state)).clip(-max_action, max_action)

        # Perform action
            next_state, reward, done, _ = env.step(action)
        # TODO: check if we need this line. This is because, we set max_episode step when we instantiate the env. Susan: No need to double check "done". The max_episode step is taking care of it.
        # Then, env should know we have reached that state and return done=True. In fact the code in gym, seems to be doing that.
        # done_bool = float(done) if episode_timesteps < env._max_episode_steps else 0

        # Store data in replay buffer
            replay_buffer.add(state, action, next_state, reward, float(done))

            state = next_state
            episode_reward += reward
            # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
        print(
             f"Total T: {total_t + 1}, Episode Num: {episode_num}, Episode T: {episode_timesteps}, "
             f"Reward: {episode_reward:.3f}")


    # Save final buffer and performance
    evaluations.append(eval_policy(policy, args.env, args.seed, max_episode_step=args.max_traj_len))
    np.save(f"{attack_path}/results/target_buffer_performance_{setting}", evaluations)
    replay_buffer.save(f"{attack_path}/buffers/{buffer_name}_compatible")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--env" , help="the environment you are in", default="Hopper-v3")           # OpenAI gym environment name
    parser.add_argument("--seed", type=int)                                              # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--buffer_name", default="Robust")                                          # Prepends name to filename

    parser.add_argument("--eval_freq", default=5e3, type=float)                                     # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=int(1e6),
                        type=int)                           # Max time steps to run environment or train for (this defines buffer size)

    # TODO: FixMe: should we tweak start_timesteps for different max_timesteps? Susan: Even if such a thing is needed, it will be taken care of during the hyper parameter tuning
    parser.add_argument("--start_timesteps", default=int(25e3),
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
    parser.add_argument("--attack_threshold", default=0.75)  # Threshold for attack training #TODO: Not needed in this file
    parser.add_argument("--attack_training_size", default=0.75)  # Attack training size #TODO: Not needed in this file. Also, the default must be different. 0.75 does not make sense.

    #parser.add_argument('--timesteps', type=int)
    parser.add_argument('--just_one', default='no', choices=["yes", "no"], help="just run one experiment", type=str) #TODO: ToMySelf: Check what it does
    parser.add_argument('--all', default='no', choices=["yes", 'no'], help="run all tests") #TODO: ToMySelf: Check what it does
    parser.add_argument('--fix_num_models', default='no') #TODO: Not needed in this file
    #parser.add_argument('--threshold', default=0.5, type=float)
    parser.add_argument('--threshold_arr', nargs= '*', type=float) #TODO: ToMySelf: Check what it does
    parser.add_argument('--num_models', default=3, help="number of models to use", type=int) #TODO: Not needed in this file. May not be needed in the other file either.
    #parser.add_argument('--traj_length', default=1000, help="length of trajectory", type=int)
    parser.add_argument('--attack_model_size', default=1000, type=int,
                        help="size of the training set for the attack model") #TODO: Not needed in this file
    parser.add_argument('--run_multiple', default='no', help="choose a variable attribute with all others fixed") #TODO: ToMySelf: Check what it does
    parser.add_argument('--model', default='sac', help="model used to train the shadow_models") #TODO: probably must be removed
    parser.add_argument('--trajectory_length' , nargs='*', default= 1000, type = int) #Must be equal to the max_ep_length in trainer.py #TODO: The comment must be removed. We probably do not need the argument
    parser.add_argument('--max_traj_len', default=1000, type=int)
    parser.add_argument('--bcq_max_timesteps', default=1000, type=int)

    args = parser.parse_args()

    ######BCQ Implementation Starts Here#########

    print(50*"-")
    if args.train_behavioral:
        print(f"Setting: Training behavioral, Env: {args.env}, Seed: {args.seed}, Max Trajectory Length: {args.max_traj_len}")
    elif args.generate_buffer:
        print(f"Setting: Generating buffer, Env: {args.env}, Seed: {args.seed}, Max Trajectory Length: {args.max_traj_len}")
    else:
        print(f"Setting: Training BCQ, Env: {args.env}, Seed: {args.seed}, Max Trajectory Length: {args.max_traj_len}")
    print(50*"-")

    if args.train_behavioral and args.generate_buffer:
        print("Train_behavioral and generate_buffer cannot both be true.")
        exit()

    attack_path = f"{os.path.expanduser('~')}/learning_output/{args.env}/{args.max_timesteps}/{args.seed}/{args.max_traj_len}"

    if not os.path.exists(f"{attack_path}/results"):
        os.makedirs(f"{attack_path}/results")

    if not os.path.exists(f"{attack_path}/models"):
        os.makedirs(f"{attack_path}/models")

    if not os.path.exists(f"{attack_path}/buffers"):
        os.makedirs(f"{attack_path}/buffers")

    # if not os.path.exists(f"./{attack_path}/attack_outputs"):
    #     os.makedirs(f"./{attack_path}/attack_outputs")
    # if not os.path.exists("./attack_outputs"):
    #     os.makedirs(f"./attack_outputs")

    env = gym.make(args.env)

    env.seed(args.seed)
    # Bounding the maximum allowed trajectory length in the environment
    env._max_episode_steps = args.max_traj_len
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    state_dim = env.observation_space.shape[0]  # for Hopper-v3, state_dim == 11
    action_dim = env.action_space.shape[0]  # for Hopper-v3, action_dim == 3
    max_action = float(env.action_space.high[0])  # for Hopper-v3, max_action == 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.train_behavioral or args.generate_buffer:
        interact_with_environment(attack_path, env, state_dim, action_dim, max_action, device, args)
    elif args.train_policy:
        train_BCQ(attack_path, state_dim, action_dim, max_action, device, args)
        policy_interact_with_environment(attack_path, env, state_dim, action_dim, max_action, device, args)
    else:
        raise NotImplementedError

