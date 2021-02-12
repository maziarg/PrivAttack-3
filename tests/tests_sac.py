import unittest
from utils.helpers import *
from workers.attack import train_attack_model_v2
from trainer import train_shadow_model


class SacTestSuite(unittest.TestCase):
    def setUp(self) -> None:
        # test variables
        self.model = "sac"
        self.seed1 = 0
        self.seed2 = 2
        self.seeds = [0, 2]
        self.sac_dim = 24  # dimension is 17 obs
        self.dim = 24
        self.threshold = .5
        self.timesteps = 2000
        self.trajectory_length = 100
        self.attack_model_size = 5000
        self.env_cheetah = 'HalfCheetah-v2'

    def train_ddpg_halfcheet(self):
        train_shadow_model(self.model, self.env_cheetah, self.seed1, self.timesteps)
        train_shadow_model(self.model, self.env_cheetah, self.seed2, self.timesteps)

        baseline, false_negatives_bl, false_positives_bl, RMSE_e_i, accuracy, false_negatives, false_positives = train_attack_model_v2(
            self.env_cheetah, self.threshold, self.trajectory_length,
            self.seeds,
            self.attack_model_size, test_size=50, timesteps=self.timesteps, dimension=self.dim)
        self.assertTrue(0 <= baseline <= 1)
        self.assertTrue(0 <= accuracy <= 1)

    def tearDown(self) -> None:
        if os.path.exists('tmp_plks'):
            shutil.rmtree('tmp_plks')

    if __name__ == '__main__':
        unittest.main()
