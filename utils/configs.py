# configurations are of format model_timesteps and contain the seeds used on the model that pertains
model_sac = "sac"
env_cheetah = "HalfCheetah-v2"
CORRELATED = "CORRELATED"
DECORRELATED = "DECORRELATED"
SEMI_CORRELATED = "SEMI_CORRELATED"
CORRELATION_MAP = {"c": CORRELATED, "d": DECORRELATED, "s": SEMI_CORRELATED}
# general experiment configs
#trajectory_length = [1000, 500, 50]
attack_model_size = [1000]
num_shadow_models = [5]
#threshold_arr = [0.1, 0.3, 0.5, 0.7, 0.9]

