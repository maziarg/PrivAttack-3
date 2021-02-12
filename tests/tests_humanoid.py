import unittest
from trainer import train_shadow_model
from workers.attack import train_attack_model_v2


class HumanoidTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.model = "sac"
        self.model_ddpg = "ddpg"
        self.env = "Humanoid-v2"
        self.env_half = "HalfCheetah-v2"
        self.seed1 = 8
        self.seed2 = 9
        self.seed1_d = 88
        self.seed2_d = 99
        self.seeds = [self.seed1, self.seed2]
        self.seeds_d = [self.seed1_d, self.seed2_d]
        self.timesteps = 2000
        self.dim = 394
        self.obs_dim = 376
        self.act_dim = 17
        self.threshold = 0.5
        self.trajectory_length = 200
        self.attack_model_size = 500

    def train_ddpg_halfcheet(self):
        train_shadow_model(self.model, self.env_half, self.seed1, self.timesteps)
        train_shadow_model(self.model, self.env_half, self.seed2, self.timesteps)

        baseline, false_negatives_bl, false_positives_bl, RMSE_e_i, accuracy, false_negatives, false_positives = train_attack_model_v2(
            self.env_half, self.threshold, self.trajectory_length,
            self.seeds,
            self.attack_model_size, test_size=50, timesteps=self.timesteps, dimension=self.dim)
        self.assertTrue(0 <= baseline <= 1)
        self.assertTrue(0 <= accuracy <= 1)


if __name__ == '__main__':
    unittest.main()
