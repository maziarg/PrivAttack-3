# Privacy Leakage in Deep Reinforcement Learning

## Objectives
Demonstrate privacy issues inherit to models commonly used in the field of DeepRL through
membership inference attacks 

## Supported Features
   The application currently supports 2 DeepRL models: `sac` and `ddpg`
   
   Feel free to use any Mujoco GYM Environment you like but keep in mind only `HalfCheetah-v2`, `Hopper-v2` and
   `Humanoid-v2` have been tested. Visit: [OpenAI GYM MuJuCo](https://gym.openai.com/envs/#mujoco) for more details
## Installation
   This project uses Python 3.7.4, `pip` to manage packages and dependencies.
   Make sure to upgrade `pip` before use. Create and activate your virtualenv using  `python3 -m virtualenv venv` 
   then `source venv/bin/activate`. Then run \
   `pip install -r requirement-cc.txt` 
## Running Experiments
   Note: You must have a Mujoco license to run these experiments using `mujoco200` you can download the packages are [here](https://www.roboti.us/index.html) 
   afterwards add your license key to the folder `.mujoco`
   To train shadow models for a specific timestep amount run: \
   `python trainer.py -e {environment_name} -m {model_name} --timesteps {number_of_timesteps} --seeds {seed_1} {seed_2} ... {seed_n} ` \
   Once models are trained for a specific timestep length you can run the entire experiment suite: \
   `python runner_v2.py -e {environment_name} --timesteps {number_of_timesteps} --seeds {seed_1} {seed_2} ... {seed_n}`
   
   To define your own configuration of experiments to run edit the values in the file `utils/configs.py` 
   
   The model's train and test experiences are stored under `output/environment_name/seed/` and results are stored at `output/results/`
   
   A IPython notebook is provided for ease of use, to run install jupyter:
   `pip install ipython`